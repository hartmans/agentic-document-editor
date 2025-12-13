import { useEffect, useRef, useState } from "react";
import {
  DocumentResponse,
  InterruptionRequest,
  InterruptionResponse,
  fetchDocument,
  fetchStatus,
  postInterruption,
} from "../api";
import { VoiceInput } from "./VoiceInput";
import {
  DocumentPlayback,
  DocumentPlaybackHandle,
  ResumePoint,
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
        offsetMs: info.elapsedMs,
        docCharPosition: info.docCharPosition ?? undefined,
      });
    }
  };

  const handleInterrupt = () => {
    const info = playbackRef.current?.stop();
    if (!info || info.chunkId == null) return;
    sendInterruption({
      chunk_id: info.chunkId,
      elapsed_ms: info.elapsedMs,
    });
  };

  async function sendInterruption(payload: InterruptionRequest) {
    try {
      const resp: InterruptionResponse = await postInterruption(payload);
      setPlaying(false);
      setResumePoint({
        chunkId: resp.chunk_id,
        offsetMs: resp.elapsed_ms,
        docCharPosition: resp.doc_char_position,
      });
    } catch (err) {
      setError((err as Error).message);
    }
  }

  function handleSpeechStart(payload: {
    preroll: Float32Array;
    mediaStream: MediaStream;
    sampleRate: number;
  }) {
    // handled inside VoiceInput now; no-op
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
