import { useEffect, useMemo, useRef, useState } from "react";
import { fetchChunkAudio } from "../api";
import { AudioEngine } from "../audio/AudioEngine";

export interface ChunkRef {
  chunk_id: number;
  doc_char_start: number;
  doc_char_end: number;
}

interface DocumentPlaybackProps {
  documentText: string;
  chunks: ChunkRef[];
  audioContext: AudioContext | null;
  playing: boolean;
  onPlayingChange: (playing: boolean) => void;
  onInterruptRequest?: (payload: { chunkId: number; elapsedMs: number }) => void;
  onRegisterInterrupt?: (fn: () => void) => void;
  resumePoint?: { chunkId: number; offsetMs: number } | null;
  interruptionPosition?: number | null;
}

export function DocumentPlayback({
  documentText,
  chunks,
  audioContext,
  playing,
  onPlayingChange,
  onInterruptRequest,
  onRegisterInterrupt,
  resumePoint,
  interruptionPosition,
}: DocumentPlaybackProps) {
  const [currentChunkId, setCurrentChunkId] = useState<number | null>(null);
  const [highlightRange, setHighlightRange] = useState<{ start: number; end: number }>(
    { start: 0, end: 0 }
  );
  const audioEngineRef = useRef<AudioEngine | null>(null);
  const audioBuffersRef = useRef<Map<number, AudioBuffer>>(new Map());
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (audioContext) {
      audioEngineRef.current = new AudioEngine(audioContext);
      onRegisterInterrupt?.(() => emitInterrupt());
    }
  }, [audioContext, onRegisterInterrupt]);

  useEffect(() => {
    // reset highlight when document changes
    setHighlightRange({ start: 0, end: 0 });
  }, [documentText]);

  useEffect(() => {
    if (interruptionPosition == null) return;
    setHighlightRange({
      start: interruptionPosition,
      end: interruptionPosition + 1,
    });
  }, [interruptionPosition]);

  useEffect(() => {
    if (!playing || !resumePoint || !audioEngineRef.current) return;
    startPlayback(resumePoint.chunkId, resumePoint.offsetMs);
  }, [playing, resumePoint]); // eslint-disable-line react-hooks/exhaustive-deps

  async function loadBuffer(chunkId: number): Promise<AudioBuffer> {
    const cached = audioBuffersRef.current.get(chunkId);
    if (cached) return cached;
    if (!audioContext) throw new Error("AudioContext not ready");
    const data = await fetchChunkAudio(chunkId);
    const buffer = await audioContext.decodeAudioData(data.slice(0));
    audioBuffersRef.current.set(chunkId, buffer);
    return buffer;
  }

  async function startPlayback(chunkId: number, offsetMs = 0) {
    if (!audioEngineRef.current || !audioContext) return;
    setLoading(true);
    try {
      const buffer = await loadBuffer(chunkId);
      const chunk = chunks[chunkId];
      if (chunk) setHighlightRange({ start: chunk.doc_char_start, end: chunk.doc_char_end });
      audioEngineRef.current.playChunk(chunkId, buffer, offsetMs / 1000, () => {
        handleNaturalEnd(chunkId);
      });
      setCurrentChunkId(chunkId);
      onPlayingChange(true);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  function stopPlayback() {
    audioEngineRef.current?.stop();
    onPlayingChange(false);
  }

  function emitInterrupt() {
    const info = audioEngineRef.current?.getCurrentPlaybackInfo();
    if (!info || info.playingChunkId == null || info.chunkStartTime == null) return;
    const elapsedMs = Math.round(
      (info.contextCurrentTime - info.chunkStartTime) * 1000
    );
    stopPlayback();
    onInterruptRequest?.({ chunkId: info.playingChunkId, elapsedMs });
  }

  function handleNaturalEnd(chunkId: number) {
    // advance to next chunk
    const nextId = chunkId + 1;
    if (nextId < chunks.length) {
      startPlayback(nextId, 0);
    } else {
      onPlayingChange(false);
    }
  }

  useEffect(() => {
    // start from beginning when mic ready and not playing
    if (audioContext && !playing && chunks.length > 0 && currentChunkId === null) {
      setCurrentChunkId(0);
    }
  }, [audioContext, playing, chunks, currentChunkId]);

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
      <div style={{ marginTop: "0.5rem", display: "flex", gap: "0.5rem" }}>
        <button
          onClick={() => startPlayback(currentChunkId ?? 0, 0)}
          disabled={!audioContext || loading}
        >
          {playing ? "Restart playback" : "Play"}
        </button>
        <button onClick={stopPlayback} disabled={!playing}>
          Stop
        </button>
        <button onClick={emitInterrupt} disabled={!playing}>
          Interrupt now
        </button>
      </div>
    </section>
  );
}
