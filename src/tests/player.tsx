import { useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { AudioEngine } from "../audio/AudioEngine";
import { AudioCtx } from "../context/AudioContextProvider";
import { DocumentPlayback, type DocumentPlaybackHandle } from "../components/DocumentPlayback";

const engine = new AudioEngine();
Object.assign(window, { engine });

function Harness() {
  const ref = useRef<DocumentPlaybackHandle>(null);
  const [context] = useState(() => new AudioContext());
  const [playing, setPlaying] = useState(false);
  const [resolved, setResolved] = useState("");
  const doc = new URLSearchParams(location.search).get("document") ?? "complete";
  return <AudioCtx value={{audioContext: context, setAudioContext: () => {}, micStream: null, setMicStream: () => {}}}>
    <DocumentPlayback ref={ref} documentId={doc} documentText="Hello world"
      chunks={[{doc_char_start: 0, doc_char_end: 11, has_audio: true}]}
      onPlayingChange={setPlaying}/>
    <button onClick={() => void ref.current?.play()}>Start</button>
    <button onClick={async () => {
      const point = ref.current!.stop();
      const resolved = await ref.current!.resolvePosition(point);
      setResolved(JSON.stringify(resolved));
    }}>Interrupt</button>
    <output id="playing">{String(playing)}</output>
    <output id="resolved">{resolved}</output>
  </AudioCtx>;
}
createRoot(document.getElementById("root")!).render(<Harness/>);
