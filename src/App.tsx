import { useEffect, useMemo, useRef, useState } from "react";
import {
  ChunkDetail,
  ChunkMetadata,
  InterruptionResponse,
  StatusResponse,
  fetchChunkAudio,
  fetchChunkDetail,
  fetchChunks,
  fetchStatus,
  postInterruption,
} from "./api";
import { AudioEngine } from "./audio/AudioEngine";

interface ErrorState {
  message: string;
}

type ChunkDetailCache = Record<number, ChunkDetail>;

function highlightWord(
  chunkText: string,
  start: number,
  end: number
): JSX.Element {
  if (start < 0 || end > chunkText.length || start >= end) {
    return <span>{chunkText}</span>;
  }
  return (
    <span>
      {chunkText.slice(0, start)}
      <mark>{chunkText.slice(start, end)}</mark>
      {chunkText.slice(end)}
    </span>
  );
}

export default function App() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [chunks, setChunks] = useState<ChunkMetadata[]>([]);
  const [chunkDetails, setChunkDetails] = useState<ChunkDetailCache>({});
  const [selectedChunkId, setSelectedChunkId] = useState<number | null>(null);
  const [interruptionResult, setInterruptionResult] =
    useState<InterruptionResponse | null>(null);
  const [error, setError] = useState<ErrorState | null>(null);
  const [micEnabled, setMicEnabled] = useState(false);
  const [micError, setMicError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);

  const audioContextRef = useRef<AudioContext | null>(null);
  const audioEngineRef = useRef<AudioEngine | null>(null);
  const audioBuffersRef = useRef<Map<number, AudioBuffer>>(new Map());

  useEffect(() => {
    const loadInitial = async () => {
      try {
        const [statusRes, chunksRes] = await Promise.all([
          fetchStatus(),
          fetchChunks(),
        ]);
        setStatus(statusRes);
        setChunks(chunksRes);
        if (chunksRes.length > 0) {
          setSelectedChunkId(chunksRes[0].chunk_id);
        }
      } catch (err) {
        console.error(err);
        setError({ message: (err as Error).message });
      }
    };
    loadInitial();
  }, []);

  useEffect(() => {
    const fetchDetail = async () => {
      if (selectedChunkId == null || selectedChunkId in chunkDetails) return;
      try {
        const detail = await fetchChunkDetail(selectedChunkId);
        setChunkDetails((prev) => ({ ...prev, [selectedChunkId]: detail }));
      } catch (err) {
        console.error(err);
        setError({ message: (err as Error).message });
      }
    };
    fetchDetail();
  }, [selectedChunkId, chunkDetails]);

  const selectedChunkDetail = useMemo(() => {
    if (selectedChunkId == null) return null;
    return chunkDetails[selectedChunkId] || null;
  }, [selectedChunkId, chunkDetails]);

  async function enableMic() {
    setMicError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const context =
        audioContextRef.current || new AudioContext({ latencyHint: "interactive" });
      await context.resume();
      // Connect mic to context graph to keep it alive; audio not used yet.
      const source = context.createMediaStreamSource(stream);
      source.connect(context.destination);

      audioContextRef.current = context;
      audioEngineRef.current = new AudioEngine(context);
      setMicEnabled(true);
    } catch (err) {
      const message = (err as Error).message || "Failed to enable microphone";
      setMicError(message);
      setMicEnabled(false);
    }
  }

  async function loadAudioBuffer(chunkId: number): Promise<AudioBuffer> {
    const cached = audioBuffersRef.current.get(chunkId);
    if (cached) return cached;
    const context = audioContextRef.current;
    if (!context) throw new Error("AudioContext not initialized");
    const buffer = await fetchChunkAudio(chunkId).then((arrayBuffer) =>
      context.decodeAudioData(arrayBuffer.slice(0))
    );
    audioBuffersRef.current.set(chunkId, buffer);
    return buffer;
  }

  async function handlePlay() {
    if (!micEnabled || !audioEngineRef.current || !audioContextRef.current) return;
    if (selectedChunkId == null) return;
    setError(null);
    setIsLoading(true);
    try {
      const buffer = await loadAudioBuffer(selectedChunkId);
      await audioContextRef.current.resume();
      audioEngineRef.current.playChunk(selectedChunkId, buffer, () => {
        setIsPlaying(false);
      });
      setIsPlaying(true);
    } catch (err) {
      setError({ message: (err as Error).message });
    } finally {
      setIsLoading(false);
    }
  }

  function handleStop() {
    audioEngineRef.current?.stop();
    setIsPlaying(false);
  }

  async function handleInterrupt() {
    const engine = audioEngineRef.current;
    if (!engine || !isPlaying) {
      setError({ message: "No chunk is currently playing." });
      return;
    }
    const info = engine.getCurrentPlaybackInfo();
    if (info.playingChunkId == null || info.chunkStartTime == null) {
      setError({ message: "No active playback to interrupt." });
      return;
    }
    engine.stop();
    setIsPlaying(false);
    const elapsedMs = Math.round(
      (info.contextCurrentTime - info.chunkStartTime) * 1000
    );
    try {
      const result = await postInterruption({
        chunk_id: info.playingChunkId,
        elapsed_ms: elapsedMs,
      });
      setInterruptionResult(result);
    } catch (err) {
      setError({ message: (err as Error).message });
    }
  }

  const chunkList = (
    <div style={{ border: "1px solid #ccc", padding: "0.5rem", maxHeight: "50vh", overflowY: "auto" }}>
      <h3>Chunks</h3>
      <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
        {chunks.map((chunk) => (
          <li key={chunk.chunk_id} style={{ marginBottom: "0.5rem" }}>
            <button
              onClick={() => setSelectedChunkId(chunk.chunk_id)}
              disabled={isLoading}
              style={{
                width: "100%",
                textAlign: "left",
                padding: "0.5rem",
                background:
                  selectedChunkId === chunk.chunk_id ? "#e0f2ff" : "#f7f7f7",
                border: "1px solid #ccc",
                borderRadius: "4px",
              }}
            >
              <strong>Chunk {chunk.chunk_id}</strong> ({chunk.doc_char_start}–
              {chunk.doc_char_end})
              <div style={{ color: "#444", marginTop: "0.25rem" }}>
                {chunk.preview}
              </div>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );

  const chunkDetailSection = (
    <div style={{ border: "1px solid #ccc", padding: "0.75rem", flex: 1 }}>
      <h3>Active chunk</h3>
      {selectedChunkDetail ? (
        <>
          <div style={{ marginBottom: "0.5rem" }}>
            <strong>ID:</strong> {selectedChunkDetail.chunk_id} |{" "}
            <strong>Offsets:</strong> {selectedChunkDetail.doc_char_start}–
            {selectedChunkDetail.doc_char_end}
          </div>
          <div
            style={{
              padding: "0.5rem",
              background: "#fafafa",
              border: "1px solid #eee",
            }}
          >
            {interruptionResult &&
            interruptionResult.chunk_id === selectedChunkDetail.chunk_id
              ? highlightWord(
                  selectedChunkDetail.text,
                  interruptionResult.chunk_char_start,
                  interruptionResult.chunk_char_end
                )
              : selectedChunkDetail.text}
          </div>
          <div style={{ marginTop: "0.5rem", fontSize: "0.9rem" }}>
            Speech marks: {selectedChunkDetail.speech_marks.length}
          </div>
        </>
      ) : (
        <div>Select a chunk to see details.</div>
      )}
    </div>
  );

  const statusPanel = status ? (
    <div style={{ marginBottom: "1rem" }}>
      <div>
        <strong>Voice:</strong> {status.voice_id} | <strong>Chunks:</strong>{" "}
        {status.chunk_count ?? status.num_chunks ?? "?"} |{" "}
        <strong>Document chars:</strong>{" "}
        {status.document_length ?? status.document_length_chars ?? "?"}
      </div>
      <div>
        <strong>Chunking:</strong> {status.chunking_strategy}
      </div>
    </div>
  ) : (
    <div>Loading status...</div>
  );

  return (
    <div style={{ fontFamily: "sans-serif", padding: "1rem" }}>
      <h1>Agentic Document Audio</h1>
      {statusPanel}
      <div style={{ marginBottom: "1rem" }}>
        <button onClick={enableMic} disabled={micEnabled}>
          {micEnabled ? "Microphone enabled" : "Enable microphone"}
        </button>
        {micError && <span style={{ color: "red", marginLeft: "0.5rem" }}>{micError}</span>}
      </div>
      <div style={{ display: "flex", gap: "1rem" }}>
        <div style={{ width: "30%" }}>{chunkList}</div>
        <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "0.75rem" }}>
          {chunkDetailSection}
          <div style={{ display: "flex", gap: "0.5rem" }}>
            <button onClick={handlePlay} disabled={!micEnabled || isLoading}>
              Play selected chunk
            </button>
            <button onClick={handleStop} disabled={!isPlaying}>
              Stop
            </button>
            <button onClick={handleInterrupt} disabled={!isPlaying}>
              Interrupt now
            </button>
            {isPlaying && <span>Playing chunk {selectedChunkId}</span>}
          </div>
          {interruptionResult && (
            <div style={{ border: "1px solid #ddd", padding: "0.5rem" }}>
              <div><strong>Interruption result</strong></div>
              <div>
                Chunk {interruptionResult.chunk_id} @ {interruptionResult.elapsed_ms} ms
              </div>
              <div>
                Word: <strong>{interruptionResult.word_text}</strong> ({interruptionResult.doc_char_start}–
                {interruptionResult.doc_char_end})
              </div>
            </div>
          )}
          {error && <div style={{ color: "red" }}>{error.message}</div>}
        </div>
      </div>
    </div>
  );
}
