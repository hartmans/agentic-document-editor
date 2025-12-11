import {
  useContext,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
  Ref,
} from "react";
import { AudioEngine } from "../audio/AudioEngine";
import { AudioCtx } from "../context/AudioContextProvider";
import { fetchChunkAudio } from "../api";

export interface ChunkRef {
  chunk_id: number;
  doc_char_start: number;
  doc_char_end: number;
}

export interface ResumePoint {
  chunkId: number;
  offsetMs: number;
  docCharPosition?: number;
}

export interface StopInfo {
  chunkId: number | null;
  elapsedMs: number;
  docCharPosition: number | null;
}

export interface DocumentPlaybackHandle {
  play: (resume?: ResumePoint) => Promise<void>;
  stop: () => StopInfo;
}

interface Props {
  documentText: string;
  chunks: ChunkRef[];
  onPlayingChange?: (playing: boolean) => void;
  interruptionPosition?: number | null;
}

export function DocumentPlayback({
  documentText,
  chunks,
  onPlayingChange,
  interruptionPosition,
  ref,
}: Props & { ref?: Ref<DocumentPlaybackHandle> }) {
    const { audioContext } = useContext(AudioCtx);
    const audioEngineRef = useRef<AudioEngine | null>(null);
    const buffersRef = useRef<Map<number, AudioBuffer>>(new Map());
    const [currentChunkId, setCurrentChunkId] = useState<number | null>(null);
    const [highlightRange, setHighlightRange] = useState<{ start: number; end: number }>(
      { start: 0, end: 0 }
    );
    const playingRef = useRef(false);
    const startTimeRef = useRef<number | null>(null);

    useEffect(() => {
      if (audioContext) {
        audioEngineRef.current = new AudioEngine(audioContext);
      }
    }, [audioContext]);

    useEffect(() => {
      if (interruptionPosition == null) return;
      setHighlightRange({ start: interruptionPosition, end: interruptionPosition + 1 });
    }, [interruptionPosition]);

    async function loadBuffer(chunkId: number): Promise<AudioBuffer> {
      const cached = buffersRef.current.get(chunkId);
      if (cached) return cached;
      if (!audioContext) throw new Error("AudioContext not ready");
      const data = await fetchChunkAudio(chunkId);
      const buf = await audioContext.decodeAudioData(data.slice(0));
      buffersRef.current.set(chunkId, buf);
      return buf;
    }

    async function startChunk(chunkId: number, offsetMs = 0) {
      if (!audioEngineRef.current || !audioContext) return;
      const buffer = await loadBuffer(chunkId);
      const chunk = chunks[chunkId];
      if (chunk) setHighlightRange({ start: chunk.doc_char_start, end: chunk.doc_char_end });
      audioEngineRef.current.playChunk(chunkId, buffer, offsetMs / 1000, () =>
        handleNaturalEnd(chunkId)
      );
      playingRef.current = true;
      startTimeRef.current = audioContext.currentTime - offsetMs / 1000;
      setCurrentChunkId(chunkId);
      onPlayingChange?.(true);
    }

    function handleNaturalEnd(finishedId: number) {
      const nextId = finishedId + 1;
      if (nextId < chunks.length) {
        startChunk(nextId, 0);
      } else {
        playingRef.current = false;
        startTimeRef.current = null;
        onPlayingChange?.(false);
      }
    }

    function stopPlayback(): StopInfo {
      const info = audioEngineRef.current?.getCurrentPlaybackInfo();
      audioEngineRef.current?.stop();
      const elapsedMs =
        info && info.chunkStartTime != null
          ? Math.round((info.contextCurrentTime - info.chunkStartTime) * 1000)
          : 0;
      const chunkId = info?.playingChunkId ?? currentChunkId;
      const docCharPosition =
        chunkId != null && chunkId < chunks.length
          ? chunks[chunkId].doc_char_start
          : null;
      playingRef.current = false;
      startTimeRef.current = null;
      onPlayingChange?.(false);
      return { chunkId, elapsedMs, docCharPosition };
    }

    useImperativeHandle(ref, () => ({
      play: async (resume?: ResumePoint) => {
        const targetChunk = resume?.chunkId ?? 0;
        const offset = resume?.offsetMs ?? 0;
        await startChunk(targetChunk, offset);
      },
      stop: () => stopPlayback(),
    }));

    const renderedText = useMemo(() => {
      if (!documentText) return "Loading document...";
      const { start, end } = highlightRange;
      if (start === end) return documentText;
      return (
        <>
          {documentText.slice(0, start)}
          <mark>{documentText.slice(start, end)}</mark>
          {documentText.slice(end)}
        </>
      );
    }, [documentText, highlightRange]);

    return (
      <section style={{ border: "1px solid #ccc", padding: "0.75rem" }}>
        <h3>Document</h3>
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
