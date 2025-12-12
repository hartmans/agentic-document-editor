import { useContext, useEffect, useRef, useState } from "react";
import { AudioCtx } from "../context/AudioContextProvider";

type VadState = "idle" | "listening" | "triggered";

interface VoiceInputProps {
  onInterrupt: () => void;
  onSpeechStart?: (payload: { preroll: Float32Array; mediaStream: MediaStream; sampleRate: number }) => void;
  disabled?: boolean;
}

/**
  * Handles mic permission, simple VAD, and a 250ms preroll ring buffer.
  * Decoupled from playback; emits onInterrupt when VAD fires or button pressed.
  */
export function VoiceInput({ onInterrupt, onSpeechStart, disabled }: VoiceInputProps) {
  const { audioContext, setAudioContext } = useContext(AudioCtx);
  const [micEnabled, setMicEnabled] = useState(false);
  const [micError, setMicError] = useState<string | null>(null);
  const [vadState, setVadState] = useState<VadState>("idle");
  const [vadThreshold, setVadThreshold] = useState(0.005);
  const [rms, setRms] = useState(0);

  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const aboveCountRef = useRef(0);
  const triggeredRef = useRef(false);

  // Preroll ring buffer (~250ms)
  const prerollBufferRef = useRef<Float32Array>(new Float32Array(12000));
  const prerollWriteIdxRef = useRef(0);

  const VAD_MIN_FRAMES = 4; // frames above threshold required

  useEffect(() => {
    return () => {
      processorRef.current?.disconnect();
      audioContext?.close();
    };
  }, [audioContext]);

  async function enableMic() {
    setMicError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1,
        },
      });
      const ctx =
        audioContext ?? new AudioContext({ latencyHint: "interactive" });
      await ctx.resume();
      const source = ctx.createMediaStreamSource(stream);
      streamRef.current = stream;
      // Resize preroll to ~250ms at current sample rate
      const prerollSamples = Math.max(1, Math.round(ctx.sampleRate * 0.25));
      prerollBufferRef.current = new Float32Array(prerollSamples);
      prerollWriteIdxRef.current = 0;

      const processor =
        processorRef.current || ctx.createScriptProcessor(1024, 1, 1);
      processor.onaudioprocess = (event) => {
        const input = event.inputBuffer.getChannelData(0);
        // preroll ring
        const buffer = prerollBufferRef.current;
        let w = prerollWriteIdxRef.current;
        for (let i = 0; i < input.length; i++) {
          buffer[w] = input[i];
          w = (w + 1) % buffer.length;
        }
        prerollWriteIdxRef.current = w;

        // rms
        let sum = 0;
        for (let i = 0; i < input.length; i++) sum += input[i] * input[i];
        const frameRms = Math.sqrt(sum / input.length);
        setRms(frameRms);

        if (!micEnabled) {
          aboveCountRef.current = 0;
          triggeredRef.current = false;
          setVadState("idle");
          return;
        }
        setVadState("listening");
        const isAbove = frameRms >= vadThreshold;
        if (isAbove) aboveCountRef.current += 1;
        else aboveCountRef.current = Math.max(0, aboveCountRef.current - 1);

        if (!triggeredRef.current && aboveCountRef.current >= VAD_MIN_FRAMES) {
          triggeredRef.current = true;
          setVadState("triggered");
          if (onSpeechStart) {
            onSpeechStart({
              preroll: prerollBufferRef.current.slice(),
              mediaStream: stream,
              sampleRate: ctx.sampleRate,
            });
          }
          onInterrupt();
        }
      };
      source.connect(processor);
      const gain = ctx.createGain();
      gain.gain.value = 0;
      processor.connect(gain);
      gain.connect(ctx.destination);

      processorRef.current = processor;
      setMicEnabled(true);
      setAudioContext(ctx);
    } catch (err) {
      setMicError((err as Error).message || "Failed to enable microphone");
      setMicEnabled(false);
    }
  }

  return (
    <section style={{ border: "1px solid #ccc", padding: "0.5rem" }}>
      <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
        <button onClick={enableMic} disabled={micEnabled || disabled}>
          {micEnabled ? "Microphone enabled" : "Enable microphone"}
        </button>
        <button onClick={onInterrupt} disabled={!micEnabled}>
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
      {micError && <div style={{ color: "red" }}>{micError}</div>}
      <div aria-live="off" style={{ position: "absolute", left: "-9999px" }}>
        VAD {vadState} RMS {rms.toFixed(3)} threshold {vadThreshold.toFixed(3)}
      </div>
    </section>
  );
}
