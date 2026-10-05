import { useEffect, useRef, useState } from "react";
import {
  ApiError,
  DocumentResponse,
  fetchStatus,
  postInstructions,
} from "../api";
import { VoiceInput } from "./VoiceInput";
import { DocumentPlayback, DocumentPlaybackHandle } from "./DocumentPlayback";
import type { PlaybackPosition } from "../audio/speechMarks";
import { instructionsSentUrl, play } from "../audio/sounds";

interface DocumentSnapshot {
  documentId: string;
  document: DocumentResponse;
}

interface ResponsePlayback extends DocumentSnapshot {
  intent: number;
  // Resuming an interrupted reply updates its intent without remounting its player.
  playbackKey: number;
  disposition: "play" | "pause" | "stop";
}

export function DocumentEditor() {
  const [snapshot, setSnapshot] = useState<DocumentSnapshot | null>(null);
  const [response, setResponse] = useState<ResponsePlayback | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [starting, setStarting] = useState(false);
  const [instructionsPending, setInstructionsPending] = useState(false);
  const [voiceMode, setVoiceMode] = useState<"idle" | "armed" | "waiting">("idle");
  const [resumePoint, setResumePoint] = useState<PlaybackPosition | null>(null);
  const playbackRef = useRef<DocumentPlaybackHandle | null>(null);
  const responsePlaybackRef = useRef<DocumentPlaybackHandle | null>(null);
  const mountedRef = useRef(false);
  const instructionRef = useRef<object | null>(null);
  const intentRef = useRef(0);
  const startingRef = useRef(false);
  const responseActiveRef = useRef(false);
  const interruptedRef = useRef<{
    intent: number;
    source: "document" | "response";
    position?: PlaybackPosition;
  } | null>(null);
  const resumePointRef = useRef<PlaybackPosition | null>(null);

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
    responseActiveRef.current = false;
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
      const resume = resumePointRef.current;
      if (!mountedRef.current || intent !== intentRef.current) return;
      if (!playbackRef.current) throw new Error("Playback not ready. Enable microphone and wait a moment.");
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

  function rememberInterruption(info: PlaybackPosition | undefined, intent: number) {
    if (!snapshot || !info || info.documentId !== snapshot.documentId) return;
    if (mountedRef.current && intent === intentRef.current) {
      resumePointRef.current = info;
      setResumePoint(info);
    }
  }

  const handleStop = () => {
    interruptedRef.current = null;
    const intent = ++intentRef.current;
    const info = playbackRef.current?.stop();
    if (playing || startingRef.current) rememberInterruption(info, intent);
    startingRef.current = false;
    setStarting(false);
    clearResponse();
    setVoiceMode("idle");
  };

  const handlePause = () => {
    interruptedRef.current = null;
    const intent = ++intentRef.current;
    const info = playbackRef.current?.stop();
    if (playing || startingRef.current) rememberInterruption(info, intent);
    startingRef.current = false;
    setStarting(false);
    clearResponse();
    setVoiceMode("armed");
  };

  const handleInterrupt = () => {
    if (instructionRef.current || interruptedRef.current) return;
    const intent = ++intentRef.current;
    const documentActive = playing || startingRef.current;
    const info = playbackRef.current?.stop();
    if (documentActive) rememberInterruption(info, intent);
    const responseActive = responseActiveRef.current;
    const responsePosition = responsePlaybackRef.current?.stop();
    interruptedRef.current = responseActive
      ? { intent, source: "response", position: responsePosition }
      : documentActive ? { intent, source: "document", position: info } : null;
    startingRef.current = false;
    setStarting(false);
  };

  const handleInstructions = async (instructions: string) => {
    if (instructionRef.current || !snapshot) return;
    const trimmed = instructions.trim();
    const interrupted = interruptedRef.current;
    interruptedRef.current = null;
    if (!trimmed) {
      if (interrupted?.intent !== intentRef.current) return;
      if (interrupted.source === "document") {
        void handlePlay();
      } else {
        setResponse((current) => current && { ...current, intent: interrupted.intent });
        void responsePlaybackRef.current?.play(interrupted.position).catch((err: Error) => {
          if (mountedRef.current && interrupted.intent === intentRef.current) setError(err.message);
        });
      }
      return;
    }
    clearResponse();
    const operation = {};
    instructionRef.current = operation;
    const intent = intentRef.current;
    const isCurrent = () => mountedRef.current && instructionRef.current === operation;
    setInstructionsPending(true);
    setVoiceMode("waiting");
    setError(null);
    try {
      const captured = resumePointRef.current;
      const player = playbackRef.current;
      if (captured && !player) throw new Error("Playback not ready");
      const point = captured ? await player!.resolvePosition(captured) : null;
      if (!isCurrent()) return;
      void play(instructionsSentUrl);
      const result = await postInstructions({
        baseDocumentId: snapshot.documentId,
        instructions: trimmed,
        resumePoint: {
          documentId: snapshot.documentId,
          docCharPosition: point?.docCharPosition ?? 0,
        },
      });
      if (!isCurrent()) return;
      // Install the revision and its resume point together; no status refetch.
      setSnapshot({ documentId: result.documentId, document: result.document });
      setResumePoint(result.resumePoint);
      resumePointRef.current = result.resumePoint;
      if (intent === intentRef.current) {
        responseActiveRef.current = true;
        setResponse({
          documentId: result.responseDocumentId,
          document: result.responseDocument,
          intent,
          playbackKey: intent,
          disposition: result.disposition,
        });
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
          resumePointRef.current = null;
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
    if (!response || response.intent !== intentRef.current) return;
    responseActiveRef.current = false;
    if (response.disposition === "play") {
      void handlePlay();
    } else if (response.disposition === "pause") {
      setVoiceMode("armed");
    } else {
      setVoiceMode("idle");
    }
  };

  const handleDocumentPlayingChange = (isPlaying: boolean) => {
    setPlaying(isPlaying);
    if (isPlaying) setVoiceMode("armed");
  };

  const handleResponsePlayingChange = (isPlaying: boolean) => {
    if (isPlaying && response?.disposition !== "stop") setVoiceMode("armed");
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
          onPlayingChange={handleResponsePlayingChange}
          onNaturalEnd={handleResponseEnd}
          key={`response/${response.documentId}/${response.playbackKey}`}
        />
      )}
      <DocumentPlayback
        ref={playbackRef}
        title="Document"
        documentId={documentId}
        documentText={doc.document}
        chunks={doc.chunks}
        key={documentId}
        onPlayingChange={handleDocumentPlayingChange}
      />
      <div style={{ display: "flex", gap: "0.5rem" }}>
        <button onClick={handlePlay} disabled={playing || starting || instructionsPending}>
          Play
        </button>
        <button onClick={handlePause} disabled={!playing && !starting && !response}>
          Pause
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
