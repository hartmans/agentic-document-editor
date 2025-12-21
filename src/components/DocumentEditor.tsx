import { useEffect, useRef, useState } from "react";
import {
  DocumentResponse,
  InterruptionRequest,
  MessageDict,
  fetchDocument,
  fetchStatus,
  postInstructions,
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
  const [documentId, setDocumentId] = useState<string | null>(null);
  const [responseDoc, setResponseDoc] = useState<DocumentResponse | null>(null);
  const [responseDocumentId, setResponseDocumentId] = useState<string | null>(null);
  const [statusText, setStatusText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [voiceMode, setVoiceMode] = useState<"idle" | "armed" | "waiting">("idle");
  const [resumePoint, setResumePoint] = useState<ResumePoint | null>(null);
  const [messages, setMessages] = useState<MessageDict[]>([]);
  const playbackRef = useRef<DocumentPlaybackHandle | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const status = await fetchStatus();
        setDoc(status.document);
        setDocumentId(status.documentId);
        setStatusText(
          `Voice ${status.voiceId}, chunks ${status.chunkCount}, strategy ${status.chunkingStrategy}`
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
      setVoiceMode("armed");
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
    setResponseDoc(null);
    setResponseDocumentId(null);
    setVoiceMode("idle");
  };

  const handleInterrupt = () => {
    const info = playbackRef.current?.stop();
    if (!info || info.chunkId == null) return;
    setResponseDoc(null);
    setResponseDocumentId(null);
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

  const handleInstructions = async (instructions: string) => {
    const trimmed = instructions.trim();
    if (!trimmed) {
      setVoiceMode((prev) => (prev === "idle" ? prev : "armed"));
      return;
    }
    const safeResumePoint = {
      chunkId: resumePoint?.chunkId ?? 0,
      offsetMs: resumePoint?.offsetMs ?? 0,
      docCharPosition: resumePoint?.docCharPosition ?? 0,
    };
    try {
      setVoiceMode((prev) => (prev === "idle" ? prev : "waiting"));
      const resp = await postInstructions({
        messages,
        instructions: trimmed,
        resumePoint: safeResumePoint,
      });
      setMessages(resp.messages);
      setResumePoint(resp.resumePoint);
      setResponseDocumentId(resp.documentId);
      const responseDocument = await fetchDocument(resp.documentId);
      setResponseDoc(responseDocument);
      setVoiceMode((prev) => (prev === "waiting" ? "armed" : prev));
    } catch (err) {
      setError((err as Error).message);
      setVoiceMode((prev) => (prev === "waiting" ? "armed" : prev));
    }
  };

  if (!doc || !documentId) return <div>Loading document...</div>;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
      <div>{statusText}</div>
      <VoiceInput
        enabled={voiceMode === "armed"}
        onInterrupt={handleInterrupt}
        onInstructions={handleInstructions}
      />
      {responseDoc && responseDocumentId && (
        <DocumentPlayback
          title="Response"
          documentId={responseDocumentId}
          documentText={responseDoc.document}
          chunks={responseDoc.chunks}
          autoPlay
          onPlayingChange={(isPlaying) => {
            if (isPlaying) setVoiceMode((prev) => (prev === "idle" ? prev : "armed"));
          }}
        />
      )}
      <DocumentPlayback
        ref={playbackRef}
        title="Document"
        documentId={documentId}
        documentText={doc.document}
        chunks={doc.chunks}
        onPlayingChange={(isPlaying) => {
          setPlaying(isPlaying);
          if (isPlaying) setVoiceMode((prev) => (prev === "idle" ? prev : "armed"));
        }}
      />
      <div style={{ display: "flex", gap: "0.5rem" }}>
        <button onClick={handlePlay} disabled={playing}>
          Play
        </button>
        <button onClick={handleStop} disabled={voiceMode !== "armed"}>
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
