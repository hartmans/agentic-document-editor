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
import { DocumentPlayback } from "./DocumentPlayback";

export function DocumentEditor() {
  const [doc, setDoc] = useState<DocumentResponse | null>(null);
  const [audioContext, setAudioContext] = useState<AudioContext | null>(null);
  const [playing, setPlaying] = useState(false);
  const [resumePoint, setResumePoint] = useState<{ chunkId: number; offsetMs: number } | null>(
    null
  );
  const [interruptionPosition, setInterruptionPosition] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [statusText, setStatusText] = useState<string>("");
  const interruptHandlerRef = useRef<() => void>();

  useEffect(() => {
    (async () => {
      try {
        const [status, document] = await Promise.all([fetchStatus(), fetchDocument()]);
        setStatusText(
          `Voice ${status.voice_id}, chunks ${status.chunk_count}, strategy ${status.chunking_strategy}`
        );
        setDoc(document);
      } catch (err) {
        setError((err as Error).message);
      }
    })();
  }, []);

  function registerInterruptHandler(fn: () => void) {
    interruptHandlerRef.current = fn;
  }

  function handleVoiceInterrupt() {
    interruptHandlerRef.current?.();
  }

  async function handleInterruptRequest(payload: InterruptionRequest) {
    try {
      const resp: InterruptionResponse = await postInterruption(payload);
      setResumePoint({ chunkId: resp.chunk_id, offsetMs: resp.elapsed_ms });
      setInterruptionPosition(resp.doc_char_position);
      setPlaying(false);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  function handleContinue() {
    if (!resumePoint) return;
    setPlaying(true);
  }

  if (!doc) {
    return <div>Loading document...</div>;
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
      <div>{statusText}</div>
      <VoiceInput onInterrupt={handleVoiceInterrupt} onMicReady={setAudioContext} />
      <DocumentPlayback
        documentText={doc.document}
        chunks={doc.chunks}
        audioContext={audioContext}
        playing={playing}
        onPlayingChange={setPlaying}
        onInterruptRequest={handleInterruptRequest}
        onRegisterInterrupt={registerInterruptHandler}
        resumePoint={resumePoint}
        interruptionPosition={interruptionPosition}
      />
      <div style={{ display: "flex", gap: "0.5rem" }}>
        <button onClick={() => setPlaying(true)} disabled={playing}>
          Play
        </button>
        <button onClick={() => setPlaying(false)} disabled={!playing}>
          Stop
        </button>
        <button onClick={handleContinue} disabled={!resumePoint || playing}>
          Continue
        </button>
      </div>
      {error && <div style={{ color: "red" }}>{error}</div>}
    </div>
  );
}
