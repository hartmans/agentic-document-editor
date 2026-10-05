import { StrictMode, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { AudioEngine } from "../audio/AudioEngine";
import { AudioCtx } from "../context/AudioContextProvider";
import { DocumentPlayback } from "../components/DocumentPlayback";
import { loadSpeechMarks, resolvePlaybackPosition, type PlaybackPosition } from "../audio/speechMarks";

const engine = new AudioEngine();
Object.assign(window, { engine });
function Harness() {
  const [context] = useState(() => new AudioContext());
  const doc = new URLSearchParams(location.search).get("document") ?? "complete";
  const [position, setPosition] = useState<PlaybackPosition | null>(() =>
    new URLSearchParams(location.search).has("autoplay") ? { documentId: doc, docCharPosition: 0 } : null);
  const [ended, setEnded] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [resolved, setResolved] = useState("");
  const chunks = useMemo(() => [{doc_char_start: 0, doc_char_end: 11, has_audio: doc !== "silent"}], [doc]);
  return <AudioCtx value={{audioContext: context, setAudioContext: () => {}, micStream: null, setMicStream: () => {}}}>
    <DocumentPlayback documentId={doc} documentText="Hello world" chunks={chunks} position={position}
      onPlayingChange={setPlaying} onNaturalEnd={() => { setEnded((count) => count + 1); setPosition(null); }}
      onPositionCaptured={async (point) => {
        if (position !== null) return;
        setPlaying(false);
        const resolved = await resolvePlaybackPosition(point, chunks, (chunkId) => loadSpeechMarks(doc, chunkId));
        setResolved(JSON.stringify(resolved));
      }}/>
    <button onClick={() => setPosition({ documentId: doc, docCharPosition: 0 })}>Start</button>
    <button onClick={() => setPosition(null)}>Interrupt</button>
    <button onClick={() => setPosition({ documentId: doc, docCharPosition: 6 })}>Seek</button>
    <output id="ended">{ended}</output>
    <output id="playing">{String(playing)}</output>
    <output id="resolved">{resolved}</output>
  </AudioCtx>;
}
createRoot(document.getElementById("root")!).render(<StrictMode><Harness/></StrictMode>);
