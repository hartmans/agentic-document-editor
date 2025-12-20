import { useEffect, useRef, useState } from "react";
import {
  DocumentResponse,
  InterruptionRequest,
  fetchDocument,
  fetchStatus,
  postInterruption,
  ResumePoint,
} from "../api";
import { VoiceInput } from "./VoiceInput";
import {
  DocumentPlayback,
  DocumentPlaybackHandle,
} from "./DocumentPlayback";

export function DocumentEditor() {
  const [doc, setDoc] = useState<DocumentResponse | null>(null);
  const [statusText, setStatusText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [resumePoint, setResumePoint] = useState<ResumePoint | null>(null);
  const playbackRef = useRef<DocumentPlaybackHandle | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const [status, document] = await Promise.all([fetchStatus(), fetchDocument()]);
        setDoc(document);
        setStatusText(
          `Voice ${status.voice_id}, chunks ${status.chunk_count}, strategy ${status.chunking_strategy}`
        );
      } catch (err) {
        setError((err as Error).message);
      }
    })();
  }, []);

  const handlePlay = async () => {
    try {
      if (!playbackRef.current) {
        setError("Playback not ready. Enable microphone and wait a moment.");
        return;
      }
      await playbackRef.current.play(resumePoint ?? undefined);
      setPlaying(true);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const handleStop = () => {
    const info = playbackRef.current?.stop();
    if (info) {
      setPlaying(false);
      setResumePoint({
        chunkId: info.chunkId ?? 0,
        offsetMs: info.offsetMs,
        docCharPosition: info.docCharPosition ?? undefined,
      });
    }
  };

  const handleInterrupt = () => {
    const info = playbackRef.current?.stop();
    if (!info || info.chunkId == null) return;
    sendInterruption({
      chunk_id: info.chunkId,
      elapsed_ms: info.offsetMs,
    });
  };

  async function sendInterruption(payload: InterruptionRequest) {
    try {
      const resp = await postInterruption(payload);
      setPlaying(false);
      setResumePoint(resp);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  if (!doc) return <div>Loading document...</div>;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
      <div>{statusText}</div>
      <VoiceInput onInterrupt={handleInterrupt} />
      <DocumentPlayback
        ref={playbackRef}
        documentText={doc.document}
        chunks={doc.chunks}
        onPlayingChange={setPlaying}
      />
      <div style={{ display: "flex", gap: "0.5rem" }}>
        <button onClick={handlePlay} disabled={playing}>
          Play
        </button>
        <button onClick={handleStop} disabled={!playing}>
          Stop
        </button>
        <button onClick={handlePlay} disabled={!resumePoint || playing}>
          Continue
        </button>
      </div>
      {error && <div style={{ color: "red" }}>{error}</div>}
    </div>
  );
}
