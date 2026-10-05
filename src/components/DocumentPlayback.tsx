import { useContext, useEffect, useEffectEvent, useMemo, useState } from "react";
import { AudioEngine } from "../audio/AudioEngine";
import { AudioCtx } from "../context/AudioContextProvider";
import { WordMark, chunkAudioUrl } from "../api";
import { chunkForPosition, loadSpeechMarks, markAtTime, timeAtPosition, type PlaybackPosition } from "../audio/speechMarks";
import { marksForStart } from "../audio/playbackReadiness";

export interface ChunkRef {
  doc_char_start: number;
  doc_char_end: number;
  has_audio?: boolean;
}

interface Props {
  title?: string;
  documentText: string;
  chunks: ChunkRef[];
  documentId: string;
  // A fixed starting point while playback is wanted; null means paused.
  position: PlaybackPosition | null;
  onPositionCaptured?: (position: PlaybackPosition) => void;
  onPlayingChange?: (playing: boolean) => void;
  onNaturalEnd?: () => void;
  onPlaybackError?: (error: Error) => void;
}

export function DocumentPlayback({ title = "Document", documentText, chunks, documentId,
  position, onPositionCaptured, onPlayingChange, onNaturalEnd, onPlaybackError }: Props) {
  const { audioContext } = useContext(AudioCtx);
  const [highlight, setHighlight] = useState<WordMark | null>(null);
  const [buffering, setBuffering] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const reportPosition = useEffectEvent((point: PlaybackPosition) => onPositionCaptured?.(point));
  const reportPlaying = useEffectEvent((playing: boolean) => onPlayingChange?.(playing));
  const reportEnd = useEffectEvent(() => onNaturalEnd?.());
  const reportError = useEffectEvent((err: Error) => onPlaybackError?.(err));

  useEffect(() => {
    if (!position || !audioContext) {
      setBuffering(false);
      reportPlaying(false);
      return;
    }
    const engine = new AudioEngine();
    let active = true;
    let frame = 0;
    let point = position;
    // These belong to this execution, not to the component's playback intent.
    const marksCache = new Map<number, WordMark[]>();
    const loadMarks = (chunkId: number): Promise<WordMark[]> => {
      const cached = marksCache.get(chunkId);
      if (cached) return Promise.resolve(cached);
      return loadSpeechMarks(documentId, chunkId).then((marks) => {
        marksCache.set(chunkId, marks);
        return marks;
      });
    };
    const capture = (): PlaybackPosition => {
      const info = engine.stop();
      if (info.playingChunkId === null) return point;
      const marks = marksCache.get(info.playingChunkId) ?? [];
      return { documentId, chunkId: info.playingChunkId,
        docCharPosition: markAtTime(marks, info.offsetSeconds * 1000)?.docCharStart ?? point.docCharPosition,
        offsetMs: Math.max(0, info.offsetSeconds * 1000) };
    };
    const fail = (err: Error) => {
      if (!active) return;
      setError(err.message);
      setBuffering(false);
      reportPlaying(false);
      reportError(err);
    };
    const finish = () => {
      if (!active) return;
      point = { documentId, docCharPosition: chunks[chunks.length - 1]?.doc_char_end ?? 0 };
      setBuffering(false);
      reportPlaying(false);
      reportEnd();
    };
    const startChunk = async (chunkId: number, target: PlaybackPosition) => {
      try {
        // Always yield before completion, including empty/silent documents.
        // Cleanup can invalidate this execution before it emits any event.
        await Promise.resolve();
        if (!active) return;
        while (chunkId < chunks.length && chunks[chunkId].has_audio === false) chunkId++;
        if (chunkId >= chunks.length) { finish(); return; }
        const start = chunks[chunkId].doc_char_start;
        const charPosition = Math.max(start, target.docCharPosition);
        const knownOffset = target.chunkId === chunkId ? target.offsetMs : undefined;
        point = { documentId, docCharPosition: charPosition, chunkId, offsetMs: knownOffset };
        const marksResult = loadMarks(chunkId).then(
          (marks) => ({ marks, error: null }),
          (err: Error) => ({ marks: [] as WordMark[], error: err }),
        );
        const marks = await marksForStart(charPosition, start, knownOffset, marksResult,
          marksCache.get(chunkId) ?? []);
        if (!active) return;
        const offset = knownOffset ?? (charPosition > start ? timeAtPosition(marks, charPosition) : 0);
        setHighlight(markAtTime(marks, offset) ?? null);
        void marksResult.then((result) => {
          if (active && result.error) setError(`Audio timing unavailable: ${result.error.message}`);
        });
        const started = await engine.playChunk(chunkId, chunkAudioUrl(documentId, chunkId), offset / 1000, {
          onEnded: () => {
            if (!active) return;
            const next = chunkId + 1;
            void startChunk(next, { documentId, docCharPosition: chunks[next]?.doc_char_start ?? charPosition });
          },
          onError: (err, info) => {
            point = { ...point, chunkId, offsetMs: Math.max(0, info.offsetSeconds * 1000) };
            fail(err);
          },
          onBuffering: (value) => { if (active) setBuffering(value); },
        });
        if (active && started) reportPlaying(true);
      } catch (err) { fail(err as Error); }
    };
    const tick = () => {
      if (!active) return;
      const info = engine.getCurrentPlaybackInfo();
      if (info.playingChunkId !== null) {
        const mark = markAtTime(marksCache.get(info.playingChunkId) ?? [], info.offsetSeconds * 1000);
        if (mark) setHighlight((previous) => previous === mark ? previous : mark);
      }
      frame = requestAnimationFrame(tick);
    };
    setError(null);
    setBuffering(true);
    if (position.documentId !== documentId) {
      fail(new Error("Resume point belongs to another document revision"));
    } else {
      const charPosition = Math.max(0, Math.min(position.docCharPosition, chunks[chunks.length - 1]?.doc_char_end ?? 0));
      void startChunk(chunkForPosition(chunks, charPosition), { ...position, docCharPosition: charPosition });
      frame = requestAnimationFrame(tick);
    }
    return () => {
      active = false;
      cancelAnimationFrame(frame);
      reportPosition(capture());
      // Lifecycle cleanup releases execution, never changes the requested intent.
    };
  }, [audioContext, documentId, chunks, position]);

  const characters = useMemo(() => Array.from(documentText), [documentText]);
  return <section style={{ border: "1px solid #ccc", padding: "0.75rem" }}>
    <h3>{title}</h3>
    {error && <div role="alert">{error}</div>}
    {buffering && <div role="status">Buffering audio…</div>}
    <div style={{ padding: "0.5rem", background: "#fafafa", minHeight: "160px" }}>
      {highlight ? <>{characters.slice(0, highlight.docCharStart).join("")}<mark>
        {characters.slice(highlight.docCharStart, highlight.docCharEnd).join("")}
      </mark>{characters.slice(highlight.docCharEnd).join("")}</> : documentText}
    </div>
  </section>;
}
