import { useContext, useEffect, useRef, useState } from "react";
import { DocumentEditor } from "./components/DocumentEditor";
import { AudioCtx } from "./context/AudioContextProvider";

export default function App() {
  const { audioContext, setAudioContext, micStream, setMicStream } = useContext(AudioCtx);
  const [micError, setMicError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const keepAwakeRef = useRef(false);
  const wakeLockRef = useRef<WakeLockSentinel | null>(null);

  async function requestWakeLock() {
    if (!("wakeLock" in navigator) || document.visibilityState !== "visible") return;
    if (wakeLockRef.current && !wakeLockRef.current.released) return;

    try {
      const wakeLock = await navigator.wakeLock.request("screen");
      if (!keepAwakeRef.current) {
        await wakeLock.release();
        return;
      }
      wakeLockRef.current = wakeLock;
      wakeLock.addEventListener("release", () => {
        if (wakeLockRef.current === wakeLock) wakeLockRef.current = null;
      });
    } catch (err) {
      console.warn("Unable to keep the screen awake", err);
    }
  }

  async function startAudio() {
    if (audioContext && micStream) return;
    setMicError(null);
    setStarting(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1,
        },
      });
      const ctx = audioContext ?? new AudioContext({ latencyHint: "interactive" });
      await ctx.resume();
      setAudioContext(ctx);
      setMicStream(stream);
      keepAwakeRef.current = true;
      await requestWakeLock();
    } catch (err) {
      setMicError((err as Error).message || "Failed to enable microphone");
    } finally {
      setStarting(false);
    }
  }

  useEffect(() => {
    const handleVisibilityChange = () => {
      if (keepAwakeRef.current && document.visibilityState === "visible") {
        void requestWakeLock();
      }
    };
    document.addEventListener("visibilitychange", handleVisibilityChange);

    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      keepAwakeRef.current = false;
      void wakeLockRef.current?.release();
    };
  }, []);

  useEffect(() => {
    return () => {
      micStream?.getTracks().forEach((track) => track.stop());
      audioContext?.close();
    };
  }, [audioContext, micStream]);

  const micReady = Boolean(audioContext && micStream);

  return (
    <main style={{ fontFamily: "sans-serif", padding: "1rem" }}>
      <h1>Agentic Document Editor</h1>
      <div style={{ display: "flex", gap: "0.75rem", alignItems: "center" }}>
        <button onClick={startAudio} disabled={micReady || starting}>
          {micReady ? "Started" : "Start"}
        </button>
        {micError && <span style={{ color: "red" }}>{micError}</span>}
      </div>
      <DocumentEditor />
    </main>
  );
}
