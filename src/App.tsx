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

function highlightWord(text: string, start: number, end: number): JSX.Element {
  if (start < 0 || end > text.length || start >= end) {
    return <span>{text}</span>;
  }
  return (
    <span>
      {text.slice(0, start)}
      <mark>{text.slice(start, end)}</mark>
      {text.slice(end)}
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
  const [autoPlayPending, setAutoPlayPending] = useState(false);
  const [pendingPlay, setPendingPlay] = useState<
    { chunkId: number; offsetMs: number } | null
  >(null);
  const [vadState, setVadState] = useState<"idle" | "listening" | "triggered">(
    "idle"
  );
  const [vadThreshold, setVadThreshold] = useState(0.005);
  const [vadRms, setVadRms] = useState(0);
  const [liveMessage, setLiveMessage] = useState("");

  const audioContextRef = useRef<AudioContext | null>(null);
  const audioEngineRef = useRef<AudioEngine | null>(null);
  const audioBuffersRef = useRef<Map<number, AudioBuffer>>(new Map());
  const isPlayingRef = useRef(false);
  const stopReasonRef = useRef<"natural" | "manual" | "interrupt">("natural");
  const vadProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const vadCountersRef = useRef<{ aboveCount: number }>({ aboveCount: 0 });
  const vadTriggeredRef = useRef(false);
  const vadThresholdRef = useRef(vadThreshold);

  const VAD_MIN_FRAMES = 4; // number of frames above threshold to trigger (~90ms)
  const VAD_FRAME_MS = 23; // approximate with 1024 buffer at 44.1kHz

  useEffect(() => {
    vadThresholdRef.current = vadThreshold;
  }, [vadThreshold]);

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

  // Auto-start playback once mic is enabled and we have a chunk.
  useEffect(() => {
    if (micEnabled && pendingPlay && selectedChunkId != null) {
      startPlaybackAt(pendingPlay.chunkId, pendingPlay.offsetMs);
      setPendingPlay(null);
      setAutoPlayPending(false);
    }
  }, [micEnabled, pendingPlay, selectedChunkId]);

  const selectedChunkDetail = useMemo(() => {
    if (selectedChunkId == null) return null;
    return chunkDetails[selectedChunkId] || null;
  }, [selectedChunkId, chunkDetails]);

  async function enableMic() {
    setMicError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1,
        },
      });
      const context =
        audioContextRef.current || new AudioContext({ latencyHint: "interactive" });
      await context.resume();
      // Set up simple VAD using ScriptProcessorNode.
      const source = context.createMediaStreamSource(stream);
      const processor =
        vadProcessorRef.current ||
        context.createScriptProcessor(1024, 1, 1); // ~23ms at 44.1k
      processor.onaudioprocess = (event) => {
        const input = event.inputBuffer.getChannelData(0);
        let sumSquares = 0;
        for (let i = 0; i < input.length; i++) {
          sumSquares += input[i] * input[i];
        }
        const rms = Math.sqrt(sumSquares / input.length);
        setVadRms(rms);
        const threshold = vadThresholdRef.current;
        const isAbove = rms >= threshold;
        const counters = vadCountersRef.current;
        if (!isPlayingRef.current) {
          counters.aboveCount = 0;
          vadTriggeredRef.current = false;
          setVadState("idle");
          return;
        }
        setVadState("listening");
        if (isAbove) {
          counters.aboveCount += 1;
        } else {
          counters.aboveCount = Math.max(0, counters.aboveCount - 1);
        }
        const sustainedMs = counters.aboveCount * VAD_FRAME_MS;
        if (
          !vadTriggeredRef.current &&
          sustainedMs >= VAD_MIN_FRAMES * VAD_FRAME_MS
        ) {
          vadTriggeredRef.current = true;
          setVadState("triggered");
          handleAutoInterrupt();
        }
      };
      source.connect(processor);
      // Processor needs a destination to run; use a gain node at 0 to avoid feedback.
      const gain = context.createGain();
      gain.gain.value = 0;
      processor.connect(gain);
      gain.connect(context.destination);
      vadProcessorRef.current = processor;
      audioContextRef.current = context;
      audioEngineRef.current = new AudioEngine(context);
      setMicEnabled(true);
      // Auto-play from start (or pending point) once mic is ready.
      const targetChunk =
        pendingPlay?.chunkId ??
        selectedChunkId ??
        (chunks.length ? chunks[0].chunk_id : null);
      if (targetChunk != null) {
        setPendingPlay({
          chunkId: targetChunk,
          offsetMs: pendingPlay?.offsetMs ?? 0,
        });
        setAutoPlayPending(true);
      }
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

  async function startPlaybackAt(chunkId: number, offsetMs = 0) {
    if (!micEnabled || !audioEngineRef.current || !audioContextRef.current) {
      setPendingPlay({ chunkId, offsetMs });
      setAutoPlayPending(true);
      return;
    }
    setError(null);
    setIsLoading(true);
    try {
      setSelectedChunkId(chunkId);
      stopReasonRef.current = "natural";
      const buffer = await loadAudioBuffer(chunkId);
      await audioContextRef.current.resume();
      const offsetSec = Math.max(0, offsetMs) / 1000;
      audioEngineRef.current.playChunk(chunkId, buffer, offsetSec, () => {
        handleChunkEnded(chunkId);
      });
      setIsPlaying(true);
      isPlayingRef.current = true;
      setVadState("listening");
      vadCountersRef.current.aboveCount = 0;
      vadTriggeredRef.current = false;
      setInterruptionResult(null);
    } catch (err) {
      setError({ message: (err as Error).message });
    } finally {
      setIsLoading(false);
    }
  }

  function handlePlay() {
    if (selectedChunkId == null) return;
    startPlaybackAt(selectedChunkId, 0);
  }

  function handleChunkEnded(finishedChunkId: number) {
    setIsPlaying(false);
    isPlayingRef.current = false;
    setVadState("idle");
    if (stopReasonRef.current !== "natural") {
      stopReasonRef.current = "natural";
      return;
    }
    stopReasonRef.current = "natural";
    // Auto-advance to next chunk if available.
    const nextIndex = chunks.findIndex((c) => c.chunk_id === finishedChunkId) + 1;
    if (nextIndex > 0 && nextIndex < chunks.length) {
      const nextChunkId = chunks[nextIndex].chunk_id;
      startPlaybackAt(nextChunkId, 0);
    }
  }

  function handleStop() {
    stopReasonRef.current = "manual";
    audioEngineRef.current?.stop();
    setIsPlaying(false);
    isPlayingRef.current = false;
    setVadState("idle");
  }

  async function interruptPlayback(source: "manual" | "vad") {
    const engine = audioEngineRef.current;
    if (!engine || !isPlayingRef.current) {
      setError({ message: "No chunk is currently playing." });
      return;
    }
    const info = engine.getCurrentPlaybackInfo();
    if (info.playingChunkId == null || info.chunkStartTime == null) {
      setError({ message: "No active playback to interrupt." });
      return;
    }
    stopReasonRef.current = "interrupt";
    engine.stop();
    setIsPlaying(false);
    isPlayingRef.current = false;
    setVadState(source === "vad" ? "triggered" : "idle");
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

  async function handleInterrupt() {
    stopReasonRef.current = "interrupt";
    interruptPlayback("manual");
  }

  async function handleAutoInterrupt() {
    interruptPlayback("vad");
    setLiveMessage("Auto interruption detected and sent to server.");
  }

  function handleContinue() {
    if (!micEnabled) return;
    if (!interruptionResult) {
      setError({ message: "No interruption to continue from." });
      return;
    }
    // Confirm with server (callback) then resume from interruption point.
    const payload = {
      chunk_id: interruptionResult.chunk_id,
      elapsed_ms: interruptionResult.elapsed_ms,
    };
    postInterruption(payload)
      .then(() => {
        startPlaybackAt(payload.chunk_id, payload.elapsed_ms);
      })
      .catch((err) => setError({ message: (err as Error).message }));
  }

  const documentText = useMemo(() => {
    if (!chunks.length) return "";
    // Stitch known chunks; if some details missing, use preview fallback.
    return chunks
      .map((c) => {
        const detail = chunkDetails[c.chunk_id];
        return detail?.text ?? c.preview;
      })
      .join("\n\n");
  }, [chunks, chunkDetails]);

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
      <div>
        <strong>Current chunk:</strong>{" "}
        {selectedChunkId != null ? selectedChunkId : "none"}
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
      <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
        <section
          style={{
            border: "1px solid #ccc",
            padding: "0.75rem",
            minHeight: "200px",
          }}
        >
          <h3>Document</h3>
          <div
            style={{
              padding: "0.5rem",
              background: "#fafafa",
              border: "1px solid #eee",
              minHeight: "150px",
            }}
          >
            {selectedChunkDetail &&
            interruptionResult &&
            interruptionResult.chunk_id === selectedChunkDetail.chunk_id
              ? highlightWord(
                  selectedChunkDetail.text,
                  interruptionResult.chunk_char_start,
                  interruptionResult.chunk_char_end
                )
              : documentText || "Loading document..."}
          </div>
        </section>

        <section
          style={{
            display: "flex",
            gap: "0.5rem",
            flexWrap: "wrap",
            alignItems: "center",
          }}
        >
          <button onClick={handlePlay} disabled={!micEnabled || isLoading}>
            {isPlaying ? "Restart playback" : "Play"}
          </button>
          <button onClick={handleStop} disabled={!isPlaying}>
            Stop
          </button>
          <button onClick={handleInterrupt} disabled={!isPlaying}>
            Interrupt now
          </button>
          <button
            onClick={handleContinue}
            disabled={!micEnabled || isPlaying || !interruptionResult}
          >
            Continue from last interruption
          </button>
          {isPlaying && <span>Playing chunk {selectedChunkId}</span>}
          <span>
            VAD:{" "}
            {vadState === "listening"
              ? "listening"
              : vadState === "triggered"
              ? "triggered"
              : "idle"}
            {vadState !== "idle" && (
              <span style={{ marginLeft: "0.35rem" }}>
                RMS {vadRms.toFixed(3)} / thr {vadThreshold.toFixed(3)}
              </span>
            )}
          </span>
        </section>

        <label style={{ display: "inline-block" }}>
          VAD threshold (RMS):
          <input
            type="number"
            step="0.001"
            min="0"
            value={vadThreshold}
            onChange={(e) => setVadThreshold(Number(e.target.value))}
            style={{ marginLeft: "0.5rem" }}
          />
        </label>

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
        <div aria-live="polite" style={{ position: "absolute", left: "-9999px" }}>
          {liveMessage ||
            (vadState === "triggered"
              ? "Voice activity detected."
              : `VAD state ${vadState}, RMS ${vadRms.toFixed(3)}, threshold ${vadThreshold.toFixed(3)}`)}
        </div>
      </div>
    </div>
  );
}
