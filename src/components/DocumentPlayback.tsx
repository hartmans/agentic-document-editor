import {
  useContext,
  useEffect,
  useEffectEvent,
  useLayoutEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
  Ref,
} from "react";
import { AudioEngine } from "../audio/AudioEngine";
import { AudioCtx } from "../context/AudioContextProvider";
import { ResumePoint, WordMark, fetchChunkAudio, fetchChunkMarks } from "../api";
import { chunkForPosition, markAtTime, timeAtPosition, resolvePlaybackPosition, PlaybackPosition } from "../audio/speechMarks";
import { marksForStart } from "../audio/playbackReadiness";

export interface ChunkRef {
  doc_char_start: number;
  doc_char_end: number;
  has_audio?: boolean;
}

export interface DocumentPlaybackHandle {
  play: (resume?: PlaybackPosition) => Promise<void>;
  stop: () => PlaybackPosition;
  resolvePosition: (point: PlaybackPosition) => Promise<ResumePoint>;
}

let globalComponentNumber = 0;

interface Props {
  title?: string;
  documentText: string;
  chunks: ChunkRef[];
  documentId: string;
  onPlayingChange?: (playing: boolean) => void;
  onNaturalEnd?: () => void;
  autoPlay?: boolean;
}

export function DocumentPlayback({
  title = "Document",
  documentText,
  chunks,
  documentId,
  onPlayingChange,
  onNaturalEnd,
  autoPlay = false,
  ref,
}: Props & { ref?: Ref<DocumentPlaybackHandle> }) {
    const { audioContext } = useContext(AudioCtx);
    const audioEngineRef = useRef<AudioEngine | null>(null);
    const buffersRef = useRef<Map<string, AudioBuffer>>(new Map());
    const generationRef = useRef(0);
    const pendingStartRef = useRef<PlaybackPosition | null>(null);
    const marksRef = useRef<Map<string, WordMark[]>>(new Map());
    const pendingMarksRef = useRef<Map<string, Promise<WordMark[]>>>(new Map());
    const lastPositionRef = useRef<PlaybackPosition>({ documentId, docCharPosition: 0 });
    const [highlight, setHighlight] = useState<WordMark | null>(null);
    const [error, setError] = useState<string | null>(null);
    const playingRef = useRef(false);
    const componentNumber = useMemo(() => {return ++globalComponentNumber}, []);

    const emitNaturalEnd = useEffectEvent(() => onNaturalEnd?.());

    useLayoutEffect(() => {
      if (audioContext) {
        audioEngineRef.current = new AudioEngine(audioContext);
      }
      return () => {
        generationRef.current += 1;
        pendingStartRef.current = null;
        playingRef.current = false;
        audioEngineRef.current?.stop();
        audioEngineRef.current = null;
        console.log('DocumentPlayback cleanup', {componentNumber});
      };
    }, [audioContext, documentId]);

    const updateHighlight = useEffectEvent(() => {
      if (!playingRef.current) return;
      const info = audioEngineRef.current?.getCurrentPlaybackInfo();
      if (info?.playingChunkId == null || info.chunkStartTime == null) return;
      const marks = marksRef.current.get(`${documentId}/${info.playingChunkId}`) ?? [];
      const mark = markAtTime(marks, (info.contextCurrentTime - info.chunkStartTime) * 1000);
      if (mark) {
        setHighlight((previous) => previous === mark ? previous : mark);
      }
    });

    useEffect(() => {
      let frame: number;
      const tick = () => {
        updateHighlight();
        frame = requestAnimationFrame(tick);
      };
      frame = requestAnimationFrame(tick);
      return () => cancelAnimationFrame(frame);
    }, [audioContext, documentId]);

    async function loadMarks(chunkId: number): Promise<WordMark[]> {
      const key = `${documentId}/${chunkId}`;
      const cached = marksRef.current.get(key);
      if (cached) return cached;
      let pending = pendingMarksRef.current.get(key);
      if (!pending) {
        pending = fetchChunkMarks(documentId, chunkId).then((result) => {
          if (result.documentId !== documentId || result.chunkId !== chunkId) {
            throw new Error("Speech marks belong to another document chunk");
          }
          marksRef.current.set(key, result.marks);
          return result.marks;
        }).finally(() => pendingMarksRef.current.delete(key));
        pendingMarksRef.current.set(key, pending);
      }
      return pending;
    }

    async function loadBuffer(chunkId: number): Promise<AudioBuffer> {
      const key = `${documentId}/${chunkId}`;
      const cached = buffersRef.current.get(key);
      if (cached) return cached;
      if (!audioContext) throw new Error("AudioContext not ready");
      const data = await fetchChunkAudio(documentId, chunkId);
      const buf = await audioContext.decodeAudioData(data.slice(0));
      buffersRef.current.set(key, buf);
      return buf;
    }

    async function startChunk(chunkId: number, docCharPosition: number,
                              generation = generationRef.current, knownOffsetMs?: number) {
      if (!audioEngineRef.current || !audioContext) return;
      if (chunks[chunkId].has_audio === false) {
        handleNaturalEnd(chunkId, generation);
        return;
      }
      pendingStartRef.current = { documentId, docCharPosition, chunkId, offsetMs: knownOffsetMs };
      setError(null);
      try {
        // Handle rejection immediately even when audio starts without awaiting
        // timing. The animation loop picks up late marks at the current time.
        const marksResult = loadMarks(chunkId).then(
          (marks) => ({ marks, error: null }),
          (error: Error) => ({ marks: [] as WordMark[], error }),
        );
        const buffer = await loadBuffer(chunkId);
        if (generation !== generationRef.current) return;
        const needsTiming = knownOffsetMs === undefined && docCharPosition > chunks[chunkId].doc_char_start;
        const marks = await marksForStart(
          docCharPosition, chunks[chunkId].doc_char_start, knownOffsetMs, marksResult,
          marksRef.current.get(`${documentId}/${chunkId}`) ?? [],
        );
        if (generation !== generationRef.current) return;
        if (!needsTiming) {
          void marksResult.then((result) => {
            if (result.error && generation === generationRef.current
                && audioEngineRef.current?.getCurrentPlaybackInfo().playingChunkId === chunkId) {
              setError(`Audio timing unavailable: ${result.error.message}`);
            }
          });
        }
        pendingStartRef.current = null;
        if (playingRef.current || !audioEngineRef.current) return;
        const offsetMs = knownOffsetMs ?? (needsTiming ? timeAtPosition(marks, docCharPosition) : 0);
        if (offsetMs / 1000 >= buffer.duration) {
          handleNaturalEnd(chunkId, generation);
          return;
        }
        lastPositionRef.current = {
          documentId,
          docCharPosition: marks.length ? docCharPosition : chunks[chunkId].doc_char_start,
        };
        setHighlight(markAtTime(marks, offsetMs) ?? null);
        audioEngineRef.current.playChunk(chunkId, buffer, offsetMs / 1000, () => {
          if (generation !== generationRef.current) return;
          playingRef.current = false;
          handleNaturalEnd(chunkId, generation);
        });
        playingRef.current = true;
        onPlayingChange?.(true);
      } catch (err) {
        if (generation !== generationRef.current) return;
        lastPositionRef.current = { documentId, docCharPosition };
        pendingStartRef.current = null;
        playingRef.current = false;
        setError((err as Error).message);
        onPlayingChange?.(false);
      }
    }

    function handleNaturalEnd(finishedId: number, generation: number) {
      let nextId = finishedId + 1;
      while (nextId < chunks.length && chunks[nextId].has_audio === false) nextId += 1;
      if (nextId < chunks.length) {
        void startChunk(nextId, chunks[nextId].doc_char_start, generation);
      } else {
        lastPositionRef.current = { documentId, docCharPosition: chunks[chunks.length - 1].doc_char_end };
        playingRef.current = false;
        onPlayingChange?.(false);
        emitNaturalEnd();
      }
    }

    function stopPlayback(): PlaybackPosition {
      const info = audioEngineRef.current?.getCurrentPlaybackInfo();
      const pending = pendingStartRef.current;
      generationRef.current += 1;
      pendingStartRef.current = null;
      playingRef.current = false;
      audioEngineRef.current?.stop();
      let point = pending ?? lastPositionRef.current;
      if (info?.playingChunkId != null && info.chunkStartTime != null) {
        const marks = marksRef.current.get(`${documentId}/${info.playingChunkId}`) ?? [];
        const mark = markAtTime(marks, (info.contextCurrentTime - info.chunkStartTime) * 1000);
        point = {
          documentId,
          docCharPosition: mark?.docCharStart ?? lastPositionRef.current.docCharPosition,
          chunkId: info.playingChunkId,
          offsetMs: Math.max(0, (info.contextCurrentTime - info.chunkStartTime) * 1000),
        };
      }
      lastPositionRef.current = point;
      onPlayingChange?.(false);
      return point;
    }

    useImperativeHandle(ref, () => ({
      play: async (resume?: PlaybackPosition | null) => {
        if (playingRef.current) return;
        if (resume?.documentId && resume.documentId !== documentId) {
          throw new Error("Resume point belongs to another document revision");
        }
        const generation = ++generationRef.current;
        if (!chunks.length) return;
        const docPos = Math.max(0, Math.min(
          resume?.docCharPosition ?? 0, chunks[chunks.length - 1].doc_char_end,
        ));
        const targetChunk = chunkForPosition(chunks, docPos);
        const offsetMs = resume?.chunkId === targetChunk ? resume.offsetMs : undefined;
        await startChunk(targetChunk, docPos, generation, offsetMs);
      },
      stop: () => stopPlayback(),
      resolvePosition: async (point: PlaybackPosition): Promise<ResumePoint> => {
        if (point.documentId !== documentId) {
          throw new Error("Resume point belongs to another document revision");
        }
        return resolvePlaybackPosition(point, chunks, loadMarks);
      },
    }));

    useEffect(() => {
      if (!autoPlay) return;
      if (!audioContext) return;
      if (!chunks.length) return;
      console.log('DocumentPlayback autoplay', {componentNumber});
      const generation = ++generationRef.current;
      void startChunk(0, chunks[0].doc_char_start, generation);
    }, [autoPlay, audioContext, chunks, documentId]);

    // Backend offsets count Unicode code points, not JavaScript UTF-16 units.
    const characters = useMemo(() => Array.from(documentText), [documentText]);
    const renderedText = useMemo(() => {
      if (!documentText) return "Loading document...";
      if (!highlight) return documentText;
      const start = highlight.docCharStart;
      const end = highlight.docCharEnd;
      return (
        <>
          {characters.slice(0, start).join("")}
          <mark>{characters.slice(start, end).join("")}</mark>
          {characters.slice(end).join("")}
        </>
      );
    }, [documentText, characters, highlight]);

    return (
      <section style={{ border: "1px solid #ccc", padding: "0.75rem" }}>
        <h3>{title}</h3>
        {error && <div role="alert">{error}</div>}
        <div
          style={{
            padding: "0.5rem",
            background: "#fafafa",
            border: "1px solid #eee",
            minHeight: "160px",
          }}
        >
          {renderedText}
        </div>
      </section>
    );
  }
