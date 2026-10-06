import { useContext, useEffect, useRef, useState, useEffectEvent } from "react";
import { VadLifecycle } from "../audio/VadLifecycle";
import { MicVAD } from "@ricky0123/vad-web";
import { AudioCtx } from "../context/AudioContextProvider";
import { Instructions } from "./Instructions";
import { SpeechToText } from "../SpeechToText";

type SpeechPhase = "ready" | "recording" | "transcribing" | "delivering";

interface VoiceInputProps {
  onInterrupt: () => void;
  onInstructions?: (text: string) => void;
  enabled?: boolean;
}

/**
 * Uses Silero VAD with the shared mic stream while retaining the explicit
 * 650 ms preroll ring used by both speech backends. This covers the VAD's
 * 400 ms speech confirmation delay and audio processing latency.
 */
export function VoiceInput({ onInterrupt, onInstructions, enabled = false }: VoiceInputProps) {
  const { audioContext, micStream } = useContext(AudioCtx);
  const [phase, setPhase] = useState<SpeechPhase>("ready");
  const [vad, setVad] = useState<VadLifecycle | null>(null);
  const [listening, setListening] = useState(false);
  const [vadError, setVadError] = useState<string | null>(null);
  const [speechProbability, setSpeechProbability] = useState(0);
  const [instructions, setInstructions] = useState("");

  const sttRef = useRef<SpeechToText | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const prerollBufferRef = useRef<Float32Array>(new Float32Array(31200));
  const prerollWriteIdxRef = useRef(0);

  const micEnabled = Boolean(audioContext && micStream);
  const canTrigger = micEnabled && enabled && listening && phase === "ready" && !vadError;

  const reportVadError = useEffectEvent((err: unknown) => {
    console.error("VAD transition failed", err);
    setVadError((err as Error).message || "VAD transition failed");
    setListening(false);
  });

  const appendInstruction = (text: string) => {
    setInstructions((prev) => (prev ? `${prev}\n${text}` : text));
  };

  const emitInstructions = useEffectEvent((full: string) => {
    onInstructions?.(full);
  });

  const handleTransportStop = useEffectEvent(() => {
    sttRef.current = null;
    setPhase("delivering");
  });

  const beginSpeech = useEffectEvent(() => {
    if (!canTrigger || sttRef.current || !audioContext) return;
    setPhase("recording");
    setInstructions("");
    const preroll = linearizeRing(prerollBufferRef.current, prerollWriteIdxRef.current);
    const stt = new SpeechToText({
      sampleRate: audioContext.sampleRate,
      onFinal: appendInstruction,
      onStop: handleTransportStop,
    });
    sttRef.current = stt;
    stt.queue_chunk(preroll, true);
    onInterrupt();
  });

  const finishSpeech = useEffectEvent(() => {
    if (!sttRef.current) return;
    setPhase("transcribing");
    sttRef.current.finish();
  });

  useEffect(() => {
    if (enabled) return;
    sttRef.current?.close();
    sttRef.current = null;
    setPhase("ready");
  }, [enabled]);

  const shouldListen = enabled && (phase === "ready" || phase === "recording");
  useEffect(() => {
    let active = true;
    setListening(false);
    if (vad) {
      void vad.setEnabled(shouldListen).then(() => {
        if (active) setListening(shouldListen);
      }).catch((err: unknown) => { if (active) reportVadError(err); });
    }
    return () => { active = false; };
  }, [vad, shouldListen]);

  useEffect(() => {
    if (phase !== "delivering") return;
    emitInstructions(instructions);
    // Even an empty transcript needs a fresh segmentation state. The change
    // through transcribing/delivering pauses VAD before this rearm can run.
    if (!instructions.trim()) setPhase("ready");
    // A nonempty instruction waits for the editor's disabled/enabled cycle.
  }, [phase, instructions]);

  useEffect(() => {
    if (!audioContext || !micStream) return;
    let cancelled = false;
    let lifecycle: VadLifecycle | null = null;
    setVad(null);
    setVadError(null);

    void MicVAD.new({
      model: "v6",
      audioContext,
      startOnLoad: false,
      minSpeechMs: 400,
      preSpeechPadMs: 0,
      baseAssetPath: "/vad/",
      onnxWASMBasePath: "/vad/",
      getStream: async () => micStream,
      pauseStream: async () => {},
      resumeStream: async () => micStream,
      onFrameProcessed: ({ isSpeech }) => setSpeechProbability(isSpeech),
      onSpeechRealStart: beginSpeech,
      onSpeechEnd: finishSpeech,
      onVADMisfire: finishSpeech,
    }).then(async (vad) => {
      if (cancelled) {
        await vad.destroy();
        return;
      }
      lifecycle = new VadLifecycle(vad);
      setVad(lifecycle);
    }).catch((err: unknown) => {
      if (cancelled) return;
      console.error("Failed to initialize VAD", err);
      setVadError((err as Error).message || "Failed to initialize VAD");
      setListening(false);
    });

    return () => {
      cancelled = true;
      if (lifecycle) void lifecycle.destroy().catch((err: unknown) => console.error("Failed to destroy VAD", err));
    };
  }, [audioContext, micStream]);

  useEffect(() => {
    if (!audioContext || !micStream || processorRef.current) return;
    const source = audioContext.createMediaStreamSource(micStream);
    sourceRef.current = source;
    const prerollSamples = Math.max(1, Math.round(audioContext.sampleRate * 0.65));
    prerollBufferRef.current = new Float32Array(prerollSamples);
    prerollWriteIdxRef.current = 0;

    const processor = audioContext.createScriptProcessor(1024, 1, 1);
    processor.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0);
      sttRef.current?.queue_chunk(input, false);

      const buffer = prerollBufferRef.current;
      let writeIndex = prerollWriteIdxRef.current;
      for (let i = 0; i < input.length; i++) {
        buffer[writeIndex] = input[i];
        writeIndex = (writeIndex + 1) % buffer.length;
      }
      prerollWriteIdxRef.current = writeIndex;
    };
    source.connect(processor);
    const gain = audioContext.createGain();
    gain.gain.value = 0;
    processor.connect(gain);
    gain.connect(audioContext.destination);
    processorRef.current = processor;

    return () => {
      processor.disconnect();
      source.disconnect();
      processorRef.current = null;
      sourceRef.current = null;
      sttRef.current?.close();
      sttRef.current = null;
    };
  }, [audioContext, micStream]);

  const controlsDisabled = !canTrigger;
  const vadState = vadError ? "error" : !micEnabled ? "microphone unavailable"
    : !enabled ? "paused while editor is unavailable"
    : phase === "transcribing" ? "paused while transcribing"
    : phase === "delivering" ? "paused while delivering instructions"
    : phase === "recording" ? "recording current utterance"
    : !vad ? "loading" : !listening ? "arming" : "listening";

  return (
    <section style={{ border: "1px solid #ccc", padding: "0.5rem" }}>
      <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
        <button onClick={onInterrupt} disabled={controlsDisabled}>
          Interrupt now
        </button>
        <span>VAD status: {vadState}; {canTrigger ? "speech will trigger" : "new speech will not trigger"} (speech {speechProbability.toFixed(3)})</span>
      </div>
      {vadError && <div style={{ color: "red" }}>{vadError}</div>}
      <div aria-live="off" style={{ position: "absolute", left: "-9999px" }}>
        VAD {vadState} speech probability {speechProbability.toFixed(3)}
      </div>
      <Instructions onChange={setInstructions} instructions={instructions} />
    </section>
  );
}

function linearizeRing(buffer: Float32Array, writeIdx: number): Float32Array {
  const out = new Float32Array(buffer.length);
  const firstLen = buffer.length - writeIdx;
  out.set(buffer.subarray(writeIdx), 0);
  if (writeIdx > 0) out.set(buffer.subarray(0, writeIdx), firstLen);
  return out;
}
