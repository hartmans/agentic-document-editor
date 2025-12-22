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
import { ResumePoint, fetchChunkAudio } from "../api";

export interface ChunkRef {
  chunk_id: number;
  doc_char_start: number;
  doc_char_end: number;
}

export interface DocumentPlaybackHandle {
  play: (resume?: ResumePoint) => Promise<void>;
  stop: () => ResumePoint;
}

let globalComponentNumber = 0;

interface Props {
  title?: string;
  documentText: string;
  chunks: ChunkRef[];
  documentId: string;
  onPlayingChange?: (playing: boolean) => void;
  interruptionPosition?: number | null;
  autoPlay?: boolean;
}

export function DocumentPlayback({
  title = "Document",
  documentText,
  chunks,
  documentId,
  onPlayingChange,
  interruptionPosition,
  autoPlay = false,
  ref,
}: Props & { ref?: Ref<DocumentPlaybackHandle> }) {
    const { audioContext } = useContext(AudioCtx);
    const audioEngineRef = useRef<AudioEngine | null>(null);
    const buffersRef = useRef<Map<number, AudioBuffer>>(new Map());
    const [currentChunkId, setCurrentChunkId] = useState<number | null>(null);
    const [highlightStart, setHighlightStart] = useState<number | null>(null);
    const playingRef = useRef(false);
    const componentNumber = useMemo(() => {return ++globalComponentNumber}, []);

    useEffect(() => {
      if (audioContext) {
        audioEngineRef.current = new AudioEngine(audioContext);
      }
    }, [audioContext]);

    useEffect(() => {
      return () => {
        playingRef.current = false;
        audioEngineRef.current?.stop();
        audioEngineRef.current = null;
        console.log('DocumentPlayback cleanup', {componentNumber});
      };
    }, []);

    useEffect(() => {
      if (interruptionPosition == null) return;
    }, [interruptionPosition]);

    async function loadBuffer(chunkId: number): Promise<AudioBuffer> {
      const cached = buffersRef.current.get(chunkId);
      if (cached) return cached;
      if (!audioContext) throw new Error("AudioContext not ready");
      const data = await fetchChunkAudio(documentId, chunkId);
      const buf = await audioContext.decodeAudioData(data.slice(0));
      buffersRef.current.set(chunkId, buf);
      return buf;
    }

    async function startChunk(chunkId: number, offsetMs = 0, docCharPosition?: number | null) {
      if (!audioEngineRef.current || !audioContext) return;
      const buffer = await loadBuffer(chunkId);
      const chunk = chunks[chunkId];
      if (chunk) {
        const startPos =
          docCharPosition != null ? docCharPosition : chunk.doc_char_start;
        setHighlightStart(startPos);
      }
      if (!audioEngineRef.current)
        return;
      audioEngineRef.current.playChunk(chunkId, buffer, offsetMs / 1000, () =>
        handleNaturalEnd(chunkId)
      );
      playingRef.current = true;
      setCurrentChunkId(chunkId);
      onPlayingChange?.(true);
    }

    function handleNaturalEnd(finishedId: number) {
      if (playingRef.current == false)
        return;
      const nextId = finishedId + 1;
      if (nextId < chunks.length) {
        startChunk(nextId, 0, chunks[nextId].doc_char_start);
      } else {
        playingRef.current = false;
        onPlayingChange?.(false);
      }
    }

    function stopPlayback(): ResumePoint {
      const info = audioEngineRef.current?.getCurrentPlaybackInfo();
      playingRef.current = false;
      audioEngineRef.current?.stop();
      const offsetMs =
        info && info.chunkStartTime != null
          ? Math.round((info.contextCurrentTime - info.chunkStartTime) * 1000)
          : 0;
      const chunkId = info?.playingChunkId ?? currentChunkId;
      const docCharPosition = null;
      onPlayingChange?.(false);
      return { chunkId, offsetMs, docCharPosition };
    }

    useImperativeHandle(ref, () => ({
      play: async (resume?: ResumePoint | null) => {
        const targetChunk = resume?.chunkId ?? 0;
        const offset = resume?.offsetMs ?? 0;
        const docPos =
          resume?.docCharPosition ??
          (targetChunk < chunks.length ? chunks[targetChunk].doc_char_start : undefined);
        await startChunk(targetChunk, offset, docPos);
      },
      stop: () => stopPlayback(),
    }));

    useEffect(() => {
      if (!autoPlay) return;
      if (!audioContext) return;
      if (!chunks.length) return;
      console.log('DocumentPlayback autoplay', {componentNumber});
      void startChunk(0, 0, chunks[0].doc_char_start);
    }, [autoPlay, audioContext, chunks]);

    const renderedText = useMemo(() => {
      if (!documentText) return "Loading document...";
      const anchor = highlightStart;
      if (anchor == null) return documentText;

      // Find word starting at or after anchor.
      const isWordChar = (ch: string) => /\w/.test(ch);
      let start = anchor;
      while (start < documentText.length && !isWordChar(documentText[start])) {
        start += 1;
      }
      if (start >= documentText.length) return documentText;
      let end = start;
      while (end < documentText.length && isWordChar(documentText[end])) {
        end += 1;
      }
      return (
        <>
          {documentText.slice(0, start)}
          <mark>{documentText.slice(start, end)}</mark>
          {documentText.slice(end)}
        </>
      );
    }, [documentText, highlightStart]);

    return (
      <section style={{ border: "1px solid #ccc", padding: "0.75rem" }}>
        <h3>{title}</h3>
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
