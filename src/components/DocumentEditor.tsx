import { useEffect, useRef, useState } from "react";
import {
  ApiError,
  DocumentResponse,
  MessageDict,
  fetchStatus,
  postInstructions,
  postInterruption,
  ResumePoint,
} from "../api";
import { VoiceInput } from "./VoiceInput";
import { DocumentPlayback, DocumentPlaybackHandle } from "./DocumentPlayback";
import { instructionsSentUrl, play } from "../audio/sounds";

interface DocumentSnapshot {
  documentId: string;
  document: DocumentResponse;
}

interface ResponsePlayback extends DocumentSnapshot {
  intent: number;
}

export function DocumentEditor() {
  const [snapshot, setSnapshot] = useState<DocumentSnapshot | null>(null);
  const [response, setResponse] = useState<ResponsePlayback | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [starting, setStarting] = useState(false);
  const [instructionsPending, setInstructionsPending] = useState(false);
  const [voiceMode, setVoiceMode] = useState<"idle" | "armed" | "waiting">("idle");
  const [resumePoint, setResumePoint] = useState<ResumePoint | null>(null);
  const [messages, setMessages] = useState<MessageDict[]>([]);
  const playbackRef = useRef<DocumentPlaybackHandle | null>(null);
  const responsePlaybackRef = useRef<DocumentPlaybackHandle | null>(null);
  const mountedRef = useRef(false);
  const instructionRef = useRef<object | null>(null);
  const intentRef = useRef(0);
  const startingRef = useRef(false);
  const interruptionRef = useRef<Promise<ResumePoint> | null>(null);

  useEffect(() => {
    mountedRef.current = true;
    let cancelled = false;
    void fetchStatus().then((status) => {
      if (!cancelled) setSnapshot(status);
    }).catch((err: Error) => {
      if (!cancelled) setError(err.message);
    });
    return () => {
      cancelled = true;
      mountedRef.current = false;
      instructionRef.current = null;
      intentRef.current += 1;
    };
  }, []);

  const clearResponse = () => {
    responsePlaybackRef.current?.stop();
    setResponse(null);
  };

  const handlePlay = async () => {
    if (instructionRef.current || startingRef.current || playing) return;
    const intent = ++intentRef.current;
    startingRef.current = true;
    setStarting(true);
    clearResponse();
    try {
      // A preceding interruption may still be resolving its word position.
      const resume = interruptionRef.current
        ? await interruptionRef.current
        : resumePoint;
      if (!mountedRef.current || intent !== intentRef.current) return;
      if (!playbackRef.current) throw new Error("Playback not ready. Enable microphone and wait a moment.");
      setVoiceMode("armed");
      await playbackRef.current.play(resume ?? undefined);
      // The player reports actual starts. A cancelled play must not set playing.
    } catch (err) {
      if (mountedRef.current && intent === intentRef.current) setError((err as Error).message);
    } finally {
      if (mountedRef.current && intent === intentRef.current) {
        startingRef.current = false;
        setStarting(false);
      }
    }
  };

  function rememberInterruption(info: ResumePoint | undefined, intent: number) {
    if (!snapshot || !info || info.chunkId == null) return;
    const request = postInterruption({
      documentId: snapshot.documentId,
      chunk_id: info.chunkId,
      elapsed_ms: info.offsetMs,
    });
    interruptionRef.current = request;
    void request.then((point) => {
      if (mountedRef.current && intent === intentRef.current) setResumePoint(point);
    }).catch((err: Error) => {
      if (mountedRef.current && intent === intentRef.current) setError(err.message);
    });
  }

  const handleStop = () => {
    const intent = ++intentRef.current;
    const info = playbackRef.current?.stop();
    if (playing || startingRef.current) rememberInterruption(info, intent);
    startingRef.current = false;
    setStarting(false);
    clearResponse();
    setVoiceMode("idle");
  };

  const handleInterrupt = () => {
    if (instructionRef.current) return;
    const intent = ++intentRef.current;
    const info = playbackRef.current?.stop();
    if (playing || startingRef.current) rememberInterruption(info, intent);
    startingRef.current = false;
    setStarting(false);
    clearResponse();
  };

  const handleInstructions = async (instructions: string) => {
    if (instructionRef.current || !snapshot) return;
    const trimmed = instructions.trim();
    if (!trimmed) return;
    const operation = {};
    instructionRef.current = operation;
    const intent = intentRef.current;
    const isCurrent = () => mountedRef.current && instructionRef.current === operation;
    setInstructionsPending(true);
    setVoiceMode("waiting");
    setError(null);
    try {
      const point = interruptionRef.current ? await interruptionRef.current : resumePoint;
      if (!isCurrent()) return;
      void play(instructionsSentUrl);
      const result = await postInstructions({
        baseDocumentId: snapshot.documentId,
        messages,
        instructions: trimmed,
        resumePoint: {
          documentId: snapshot.documentId,
          chunkId: point?.chunkId ?? 0,
          offsetMs: point?.offsetMs ?? 0,
          docCharPosition: point?.docCharPosition ?? 0,
        },
      });
      if (!isCurrent()) return;
      // Install the revision and its resume point together; no status refetch.
      setSnapshot({ documentId: result.documentId, document: result.document });
      setMessages(result.messages);
      setResumePoint(result.resumePoint);
      interruptionRef.current = null;
      if (intent === intentRef.current) {
        setResponse({
          documentId: result.responseDocumentId,
          document: result.responseDocument,
          intent,
        });
        setVoiceMode("armed");
      }
    } catch (err) {
      if (!isCurrent()) return;
      if (err instanceof ApiError && err.status === 409) {
        // Another tab edited this revision. Refresh, but do not replay stale
        // instructions or carry their conversation/position into the new text.
        try {
          const status = await fetchStatus();
          if (!isCurrent()) return;
          setSnapshot(status);
          setResumePoint(null);
          setMessages([]);
          interruptionRef.current = null;
          setError("The document changed in another session. Loaded the latest revision; please repeat your instruction.");
        } catch (refreshError) {
          if (isCurrent()) setError((refreshError as Error).message);
        }
        if (isCurrent()) setVoiceMode("idle");
      } else {
        setError((err as Error).message);
        if (intent === intentRef.current) setVoiceMode("armed");
      }
    } finally {
      if (isCurrent()) {
        instructionRef.current = null;
        setInstructionsPending(false);
      }
    }
  };

  const handleResponseEnd = () => {
    if (response && response.intent === intentRef.current) void handlePlay();
  };

  useEffect(() => {
    // Reposition instructions can intentionally have no spoken response.
    if (response && response.document.chunks.length === 0) handleResponseEnd();
  }, [response]);

  if (!snapshot) return <div>{error ?? "Loading document..."}</div>;
  const { document: doc, documentId } = snapshot;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
      <div>{`Voice ${doc.voice_id}, chunks ${doc.chunks.length}, strategy ${doc.chunking_strategy}`}</div>
      {instructionsPending && <div role="status">Processing instructions…</div>}
      <VoiceInput
        enabled={voiceMode === "armed" && !instructionsPending}
        onInterrupt={handleInterrupt}
        onInstructions={handleInstructions}
      />
      {response && (
        <DocumentPlayback
          ref={responsePlaybackRef}
          title="Response"
          documentId={response.documentId}
          documentText={response.document.document}
          chunks={response.document.chunks}
          autoPlay
          onNaturalEnd={handleResponseEnd}
          key={`response/${response.documentId}/${response.intent}`}
        />
      )}
      <DocumentPlayback
        ref={playbackRef}
        title="Document"
        documentId={documentId}
        documentText={doc.document}
        chunks={doc.chunks}
        key={documentId}
        onPlayingChange={setPlaying}
      />
      <div style={{ display: "flex", gap: "0.5rem" }}>
        <button onClick={handlePlay} disabled={playing || starting || instructionsPending}>
          Play
        </button>
        <button onClick={handleStop} disabled={voiceMode === "idle" && !starting && !response}>
          Stop
        </button>
        <button onClick={handlePlay} disabled={!resumePoint || playing || starting || instructionsPending}>
          Continue
        </button>
      </div>
      {error && <div style={{ color: "red" }}>{error}</div>}
    </div>
  );
}
