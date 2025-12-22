import { useContext, useEffect, useRef, useState, useEffectEvent } from "react";
import { AudioCtx } from "../context/AudioContextProvider";
import { Instructions } from "./Instructions";
import { SpeechToText } from "../SpeechToText";

type VadState = "idle" | "listening" | "triggered"|"waiting";

interface VoiceInputProps {
  onInterrupt: () => void;
  onInstructions?: (text: string) => void;
  enabled?: boolean;
}

/**
  * Handles mic permission, simple VAD, and a 250ms preroll ring buffer.
  * Decoupled from playback; emits onInterrupt when VAD fires or button pressed.
  */
export function VoiceInput({ onInterrupt, onInstructions, enabled = false }: VoiceInputProps) {
  const { audioContext, micStream } = useContext(AudioCtx);
  const [vadState, setVadState] = useState<VadState>("idle");
  const [vadThreshold, setVadThreshold] = useState(0.05);
  const [rms, setRms] = useState(0);
  const [instructions, setInstructions] = useState("");
  const sttRef = useRef<SpeechToText | null>(null);

  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const rmsRef = useRef(0);
  const vadThresholdRef = useRef(vadThreshold);
  const aboveCountRef = useRef(0);
  const triggeredRef = useRef(false);

  // Preroll ring buffer (~250ms)
  const prerollBufferRef = useRef<Float32Array>(new Float32Array(12000));
  const prerollWriteIdxRef = useRef(0);

  const VAD_MIN_FRAMES = 4; // frames above threshold required

  const appendInstruction = (text: string) => {
    setInstructions((prev) => (prev ? `${prev}\n${text}` : text));
  };

  const emitInstructions = useEffectEvent((full: string) => {
    onInstructions?.(full);
  });

  const handleStop = useEffectEvent(() => {
  if (sttRef.current)
     sttRef.current.close();
    sttRef.current = null;
    triggeredRef.current = false;
    aboveCountRef.current = 0;
    setVadState("waiting");
    emitInstructions(instructions);
    console.log("instructions:", instructions);
  });

  const emitInterrupt = useEffectEvent(() => {
    onInterrupt();
  });

  useEffect(() => {
    vadThresholdRef.current = vadThreshold;
  }, [vadThreshold]);

  useEffect(() => {
    if (!enabled) {
      triggeredRef.current = false;
      aboveCountRef.current = 0;
      setVadState("idle");
      return;
    }
    if (enabled && audioContext && micStream) {
      setVadState("listening");
      triggeredRef.current = false;
      setInstructions("");
    }
  }, [enabled, audioContext, micStream]);

  useEffect(() => {
    if (!audioContext || !micStream) return;
    if (processorRef.current) return;

    const source = audioContext.createMediaStreamSource(micStream);
    sourceRef.current = source;

    // Resize preroll to ~250ms at current sample rate
    const prerollSamples = Math.max(1, Math.round(audioContext.sampleRate * 0.25));
    prerollBufferRef.current = new Float32Array(prerollSamples);
    prerollWriteIdxRef.current = 0;

    const processor = audioContext.createScriptProcessor(1024, 1, 1);
    processor.onaudioprocess = (event) => {
      if (!enabled) return;
      const input = event.inputBuffer.getChannelData(0);
      // rms / VAD
      let sum = 0;
      for (let i = 0; i < input.length; i++) sum += input[i] * input[i];
      const frameRms = Math.sqrt(sum / input.length);
      setRms(frameRms);
      rmsRef.current = frameRms;

      const isAbove = frameRms >= vadThresholdRef.current;
      if (isAbove) aboveCountRef.current += 1;
      else aboveCountRef.current = Math.max(0, aboveCountRef.current - 1);

      let stt = sttRef.current;
      const justTriggered = !triggeredRef.current && aboveCountRef.current >= VAD_MIN_FRAMES;

      if (justTriggered) {
        triggeredRef.current = true;
        setVadState("triggered");
        if (!stt) {
          const prerollCopy = linearizeRing(
            prerollBufferRef.current,
            prerollWriteIdxRef.current
          );
          stt = sttRef.current = new SpeechToText({
            sampleRate: audioContext.sampleRate,
            getRms: () => rmsRef.current,
            vadThreshold: vadThresholdRef.current,
            onFinal: (t) => appendInstruction(t),
            onStop: () => handleStop(),
          });
          stt.queue_chunk(prerollCopy, true);
          emitInterrupt();
        }
      }

      // Send current frame if STT is active (including the frame that triggered it)
      if (stt) {
        stt.queue_chunk(input, false);
      }

      // update preroll ring AFTER using it to avoid including current frame twice
      const buffer = prerollBufferRef.current;
      let w = prerollWriteIdxRef.current;
      for (let i = 0; i < input.length; i++) {
        buffer[w] = input[i];
        w = (w + 1) % buffer.length;
      }
      prerollWriteIdxRef.current = w;
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
      if (sttRef.current) {
        sttRef.current.close();
        sttRef.current = null;
      }
    };
  }, [audioContext, micStream, enabled]);

  const micEnabled = Boolean(audioContext && micStream);
  const controlsDisabled = Boolean(!micEnabled || !enabled);

  return (
    <section style={{ border: "1px solid #ccc", padding: "0.5rem" }}>
      <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
        <button onClick={onInterrupt} disabled={controlsDisabled}>
          Interrupt now
        </button>
        <span>
          VAD status: {vadState} (RMS {rms.toFixed(3)} / thr {vadThreshold.toFixed(3)})
        </span>
      </div>
      <label style={{ display: "inline-block", marginTop: "0.35rem" }}>
        VAD threshold:
        <input
          type="number"
          step="0.001"
          min="0"
          value={vadThreshold}
          onChange={(e) => setVadThreshold(Number(e.target.value))}
          style={{ marginLeft: "0.35rem" }}
        />
      </label>
      <div aria-live="off" style={{ position: "absolute", left: "-9999px" }}>
        VAD {vadState} RMS {rms.toFixed(3)} threshold {vadThreshold.toFixed(3)}
      </div>
      <Instructions onChange={setInstructions} instructions={instructions} />
    </section>
  );
}

  function linearizeRing(buffer: Float32Array, writeIdx: number): Float32Array {
  // Oldest data is at writeIdx
  const out = new Float32Array(buffer.length);
  const firstLen = buffer.length - writeIdx;
  out.set(buffer.subarray(writeIdx), 0);
  if (writeIdx > 0) {
    out.set(buffer.subarray(0, writeIdx), firstLen);
  }
  return out;
}
