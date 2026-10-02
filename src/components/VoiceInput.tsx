import { useContext, useEffect, useRef, useState, useEffectEvent } from "react";
import { MicVAD } from "@ricky0123/vad-web";
import { AudioCtx } from "../context/AudioContextProvider";
import { Instructions } from "./Instructions";
import { SpeechToText } from "../SpeechToText";

type VadState = "idle" | "loading" | "listening" | "triggered" | "waiting" | "error";

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
  const [vadState, setVadState] = useState<VadState>("idle");
  const [vadError, setVadError] = useState<string | null>(null);
  const [speechProbability, setSpeechProbability] = useState(0);
  const [instructions, setInstructions] = useState("");
  const [sttCompleted, setSttCompleted] = useState(false);

  const vadRef = useRef<MicVAD | null>(null);
  const sttRef = useRef<SpeechToText | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const enabledRef = useRef(enabled);
  const triggeredRef = useRef(false);
  const prerollBufferRef = useRef<Float32Array>(new Float32Array(31200));
  const prerollWriteIdxRef = useRef(0);

  const appendInstruction = (text: string) => {
    setInstructions((prev) => (prev ? `${prev}\n${text}` : text));
  };

  const emitInstructions = useEffectEvent((full: string) => {
    onInstructions?.(full);
  });

  const handleTransportStop = useEffectEvent(() => {
    sttRef.current = null;
    triggeredRef.current = false;
    setVadState("waiting");
    setSttCompleted(true);
  });

  const beginSpeech = useEffectEvent(() => {
    if (!enabledRef.current || triggeredRef.current || !audioContext) return;
    triggeredRef.current = true;
    setVadState("triggered");
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
    sttRef.current?.finish();
  });

  useEffect(() => {
    enabledRef.current = enabled;
    if (!enabled) {
      sttRef.current?.close();
      sttRef.current = null;
      triggeredRef.current = false;
      setVadState("idle");
      setSttCompleted(false);
      void vadRef.current?.pause();
      return;
    }

    setInstructions("");
    setSttCompleted(false);
    prerollBufferRef.current.fill(0);
    prerollWriteIdxRef.current = 0;
    if (vadRef.current) {
      setVadState("listening");
      void vadRef.current.start();
    } else if (audioContext && micStream) {
      setVadState("loading");
    }
  }, [enabled, audioContext, micStream]);

  useEffect(() => {
    if (!sttCompleted) return;
    emitInstructions(instructions);
    setSttCompleted(false);
  }, [sttCompleted, instructions, emitInstructions]);

  useEffect(() => {
    if (!audioContext || !micStream) return;
    let cancelled = false;
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
      vadRef.current = vad;
      if (enabledRef.current) {
        setVadState("listening");
        await vad.start();
      }
    }).catch((err: unknown) => {
      if (cancelled) return;
      console.error("Failed to initialize VAD", err);
      setVadError((err as Error).message || "Failed to initialize VAD");
      setVadState("error");
    });

    return () => {
      cancelled = true;
      const vad = vadRef.current;
      vadRef.current = null;
      if (vad) void vad.destroy();
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
      if (!enabledRef.current) return;
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

  const micEnabled = Boolean(audioContext && micStream);
  const controlsDisabled = Boolean(!micEnabled || !enabled);

  return (
    <section style={{ border: "1px solid #ccc", padding: "0.5rem" }}>
      <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
        <button onClick={onInterrupt} disabled={controlsDisabled}>
          Interrupt now
        </button>
        <span>VAD status: {vadState} (speech {speechProbability.toFixed(3)})</span>
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
