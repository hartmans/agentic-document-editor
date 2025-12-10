import { useEffect, useRef, useState } from "react";

type VadState = "idle" | "listening" | "triggered";

interface VoiceInputProps {
  onInterrupt: () => void;
  onMicReady: (ctx: AudioContext) => void;
  disabled?: boolean;
}

/**
 * VoiceInput handles mic permission, simple VAD, and maintains a preroll ring buffer
 * (currently unused externally, but ready for STT wiring). It signals interruptions
 * via onInterrupt when VAD fires or when the user presses the button.
 */
export function VoiceInput({ onInterrupt, onMicReady, disabled }: VoiceInputProps) {
  const [micEnabled, setMicEnabled] = useState(false);
  const [micError, setMicError] = useState<string | null>(null);
  const [vadState, setVadState] = useState<VadState>("idle");
  const [vadThreshold, setVadThreshold] = useState(0.005);
  const [rms, setRms] = useState(0);

  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const aboveCountRef = useRef(0);
  const triggeredRef = useRef(false);

  // Preroll ring buffer (~250ms)
  const prerollBufferRef = useRef<Float32Array>(new Float32Array(48000)); // 1s at 48k
  const prerollWriteIdxRef = useRef(0);

  const VAD_MIN_FRAMES = 4;
  const VAD_FRAME_MS = 23;

  useEffect(() => {
    return () => {
      processorRef.current?.disconnect();
      audioContextRef.current?.close();
    };
  }, []);

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
        audioContextRef.current || new AudioContext({ latencyHint: "interactive" });
      await ctx.resume();
      const source = ctx.createMediaStreamSource(stream);
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

        const sustainedMs = aboveCountRef.current * VAD_FRAME_MS;
        if (!triggeredRef.current && sustainedMs >= VAD_MIN_FRAMES * VAD_FRAME_MS) {
          triggeredRef.current = true;
          setVadState("triggered");
          onInterrupt();
        }
      };
      source.connect(processor);
      // Keep processor running with silent gain
      const gain = ctx.createGain();
      gain.gain.value = 0;
      processor.connect(gain);
      gain.connect(ctx.destination);

      audioContextRef.current = ctx;
      processorRef.current = processor;
      setMicEnabled(true);
      onMicReady(ctx);
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
      {/* Live region, no aria-live so screen reader users can opt-in */}
      <div aria-live="off" style={{ position: "absolute", left: "-9999px" }}>
        VAD {vadState} RMS {rms.toFixed(3)} threshold {vadThreshold.toFixed(3)}
      </div>
    </section>
  );
}
