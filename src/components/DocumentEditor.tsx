import { useContext, useEffect, useReducer } from "react";
import { ApiError, type DocumentResponse, type InstructionsResponse, fetchStatus, postInstructions } from "../api";
import { AudioCtx } from "../context/AudioContextProvider";
import { VoiceInput } from "./VoiceInput";
import { DocumentPlayback } from "./DocumentPlayback";
import { loadSpeechMarks, resolvePlaybackPosition, type PlaybackPosition } from "../audio/speechMarks";
import { instructionsSentUrl, play } from "../audio/sounds";

interface Snapshot { documentId: string; document: DocumentResponse }
interface Response extends Snapshot { disposition: "play" | "pause" | "stop" }
type Playing = { kind: "playing"; source: "document" | "response"; position: PlaybackPosition };
type Mode = Playing | { kind: "idle" | "paused" }
  | { kind: "capturing"; prior: Playing; next: "idle" | "paused" | "listening"; text: string | null }
  | { kind: "listening"; restore: Playing | null; fallback: "paused" | "idle"; text: string | null }
  | { kind: "processing"; override: "idle" | "paused" | null };
interface Submission { text: string; snapshot: Snapshot; position: PlaybackPosition }
interface State {
  snapshot: Snapshot | null;
  response: Response | null;
  resume: PlaybackPosition | null;
  mode: Mode;
  submission: Submission | null;
  error: string | null;
}
type Action = { type: "loaded"; snapshot: Snapshot }
  | { type: "play" } | { type: "pause" } | { type: "stop" } | { type: "interrupt" }
  | { type: "transcript"; text: string } | { type: "submit" }
  | { type: "captured"; source: Playing["source"]; position: PlaybackPosition }
  | { type: "ended"; source: Playing["source"]; documentId: string }
  | { type: "result"; result: InstructionsResponse }
  | { type: "failed"; message: string; snapshot?: Snapshot }
  | { type: "playbackFailed"; message: string; source: Playing["source"]; documentId: string };
const initial: State = { snapshot: null, response: null, resume: null, mode: { kind: "idle" }, submission: null, error: null };

function documentPlayback(state: State): Playing {
  return { kind: "playing", source: "document",
    position: state.resume ?? { documentId: state.snapshot!.documentId, docCharPosition: 0 } };
}
function afterResponse(state: State, disposition: Response["disposition"]): Mode {
  return disposition === "play" ? documentPlayback(state)
    : { kind: disposition === "pause" ? "paused" : "idle" };
}
function reducer(state: State, action: Action): State {
  const mode = state.mode;
  switch (action.type) {
    case "loaded": return { ...state, snapshot: action.snapshot };
    case "play":
      if (!state.snapshot || state.submission || mode.kind === "capturing" || mode.kind === "playing") return state;
      return { ...state, mode: documentPlayback(state), error: null };
    case "pause":
    case "stop": {
      const next = action.type === "pause" ? "paused" : "idle";
      if (mode.kind === "playing") return { ...state, mode: { kind: "capturing", prior: mode, next, text: null } };
      if (mode.kind === "capturing") return { ...state, mode: { ...mode, next, text: null } };
      if (mode.kind === "processing") return { ...state, mode: { ...mode, override: next } };
      return { ...state, mode: { kind: next } };
    }
    case "interrupt":
      if (mode.kind === "processing" || mode.kind === "capturing" || mode.kind === "listening") return state;
      return { ...state, mode: mode.kind === "playing"
        ? { kind: "capturing", prior: mode, next: "listening", text: null }
        : { kind: "listening", restore: null, fallback: mode.kind === "idle" ? "idle" : "paused", text: null } };
    case "captured": {
      // Effect replay reports snapshots too, but it never requests a pause.
      // Only the explicit capture transition can consume this callback.
      if (mode.kind !== "capturing" || mode.prior.source !== action.source
          || mode.prior.position.documentId !== action.position.documentId) return state;
      const resume = action.source === "document" ? action.position : state.resume;
      return { ...state, resume, mode: mode.next === "listening"
        ? { kind: "listening", restore: { ...mode.prior, position: action.position }, fallback: "paused", text: mode.text }
        : { kind: mode.next } };
    }
    case "transcript":
      if (mode.kind === "capturing" && mode.next === "listening") return { ...state, mode: { ...mode, text: action.text } };
      if (mode.kind === "listening") return { ...state, mode: { ...mode, text: action.text } };
      if (mode.kind === "paused" || mode.kind === "idle") return { ...state, mode: { kind: "listening", restore: null, fallback: mode.kind, text: action.text } };
      return state;
    case "submit":
      if (mode.kind !== "listening" || mode.text === null || !state.snapshot) return state;
      if (!mode.text.trim()) return { ...state, mode: mode.restore ?? { kind: mode.fallback } };
      return { ...state, error: null, response: null, mode: { kind: "processing", override: null },
        submission: { text: mode.text.trim(), snapshot: state.snapshot,
          position: state.resume ?? { documentId: state.snapshot.documentId, docCharPosition: 0 } } };
    case "result": {
      if (!state.submission || mode.kind !== "processing") return state;
      const result = action.result;
      const next = { ...state, snapshot: { documentId: result.documentId, document: result.document },
        resume: result.resumePoint, submission: null,
        response: { documentId: result.responseDocumentId, document: result.responseDocument, disposition: result.disposition } };
      const audible = result.responseDocument.chunks.some((chunk) => chunk.has_audio !== false);
      return { ...next, mode: mode.override ? { kind: mode.override }
        : audible ? { kind: "playing", source: "response", position: { documentId: result.responseDocumentId, docCharPosition: 0 } }
        : afterResponse(next, result.disposition) };
    }
    case "ended":
      if (mode.kind !== "playing" || mode.source !== action.source || mode.position.documentId !== action.documentId) return state;
      return { ...state, mode: action.source === "response" && state.response
        ? afterResponse(state, state.response.disposition) : { kind: "paused" } };
    case "playbackFailed":
      if (mode.kind !== "playing" || mode.source !== action.source || mode.position.documentId !== action.documentId) return state;
      return { ...state, error: action.message, mode: { kind: "capturing", prior: mode, next: "paused", text: null } };
    case "failed":
      return { ...state, error: action.message, submission: null, mode: { kind: action.snapshot ? "idle" : "paused" },
        ...(action.snapshot ? { snapshot: action.snapshot, resume: null, response: null } : {}) };
  }
}

export function DocumentEditor() {
  const { audioContext } = useContext(AudioCtx);
  const [state, dispatch] = useReducer(reducer, initial);
  const { snapshot, response, mode, submission, error } = state;
  useEffect(() => {
    let active = true;
    void fetchStatus().then((snapshot) => { if (active) dispatch({ type: "loaded", snapshot }); })
      .catch((err: Error) => { if (active) dispatch({ type: "failed", message: err.message }); });
    return () => { active = false; };
  }, []);
  useEffect(() => {
    if (mode.kind === "listening" && mode.text !== null) dispatch({ type: "submit" });
  }, [mode]);
  useEffect(() => {
    if (!submission) return;
    let active = true;
    void (async () => {
      try {
        const point = await resolvePlaybackPosition(submission.position, submission.snapshot.document.chunks,
          (chunkId) => loadSpeechMarks(submission.snapshot.documentId, chunkId));
        if (!active) return;
        void play(instructionsSentUrl);
        const result = await postInstructions({ baseDocumentId: submission.snapshot.documentId,
          instructions: submission.text, resumePoint: point });
        if (active) dispatch({ type: "result", result });
      } catch (err) {
        if (!active) return;
        if (err instanceof ApiError && err.status === 409) {
          try {
            const snapshot = await fetchStatus();
            if (active) dispatch({ type: "failed", snapshot,
              message: "The document changed in another session. Loaded the latest revision; please repeat your instruction." });
          } catch (refreshError) { if (active) dispatch({ type: "failed", message: (refreshError as Error).message }); }
        } else dispatch({ type: "failed", message: (err as Error).message });
      }
    })();
    return () => { active = false; };
  }, [submission]);

  if (!snapshot) return <div>{error ?? "Loading document..."}</div>;
  const armed = mode.kind !== "idle" && mode.kind !== "processing"
    && !(mode.kind === "capturing" && mode.next === "idle")
    && !(mode.kind === "playing" && mode.source === "response" && response?.disposition === "stop");
  const player = (source: Playing["source"], item: Snapshot) => <DocumentPlayback
    title={source === "document" ? "Document" : "Response"}
    documentId={item.documentId} documentText={item.document.document} chunks={item.document.chunks}
    position={mode.kind === "playing" && mode.source === source ? mode.position : null}
    onPositionCaptured={(position) => dispatch({ type: "captured", source, position })}
    onNaturalEnd={() => dispatch({ type: "ended", source, documentId: item.documentId })}
    onPlaybackError={(err) => dispatch({ type: "playbackFailed", source, documentId: item.documentId, message: err.message })}
    key={`${source}/${item.documentId}`} />;
  return <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
    <div>{`Voice ${snapshot.document.voice_id}, chunks ${snapshot.document.chunks.length}, strategy ${snapshot.document.chunking_strategy}`}</div>
    {submission && <div role="status">Processing instructions…</div>}
    <VoiceInput enabled={armed} onInterrupt={() => dispatch({ type: "interrupt" })}
      onInstructions={(text) => dispatch({ type: "transcript", text })} />
    {response && response.document.document && player("response", response)}
    {player("document", snapshot)}
    <div style={{ display: "flex", gap: "0.5rem" }}>
      <button onClick={() => dispatch({ type: "play" })} disabled={!audioContext || Boolean(submission) || mode.kind === "playing" || mode.kind === "capturing"}>Play</button>
      <button onClick={() => dispatch({ type: "pause" })} disabled={mode.kind === "paused" || mode.kind === "idle"}>Pause</button>
      <button onClick={() => dispatch({ type: "stop" })} disabled={mode.kind === "idle"}>Stop</button>
      <button onClick={() => dispatch({ type: "play" })} disabled={!audioContext || !state.resume || Boolean(submission) || mode.kind === "playing" || mode.kind === "capturing"}>Continue</button>
    </div>
    {error && <div style={{ color: "red" }}>{error}</div>}
  </div>;
}
