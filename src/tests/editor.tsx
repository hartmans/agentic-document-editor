import { StrictMode, useState } from "react";
import { createRoot } from "react-dom/client";
import { MicVAD } from "@ricky0123/vad-web";
import { AudioCtx } from "../context/AudioContextProvider";
import { DocumentEditor } from "../components/DocumentEditor";
import { VadLifecycle } from "../audio/VadLifecycle";

function Harness() {
  const [context] = useState(() => new AudioContext());
  return <AudioCtx value={{audioContext: context, setAudioContext: () => {}, micStream: null, setMicStream: () => {}}}>
    <DocumentEditor />
  </AudioCtx>;
}

Object.assign(window, { checkVadTransitions: async () => {
  const context = new AudioContext();
  const stream = context.createMediaStreamDestination().stream;
  const vad = await MicVAD.new({
    model: "v6", audioContext: context, startOnLoad: false,
    baseAssetPath: "/vad/", onnxWASMBasePath: "/vad/",
    getStream: async () => stream, pauseStream: async () => {}, resumeStream: async () => stream,
  });
  const lifecycle = new VadLifecycle(vad);
  try {
    await lifecycle.setEnabled(true);
    await Promise.all([lifecycle.setEnabled(false), lifecycle.setEnabled(true)]);
    const resumed = { listening: vad.listening, active: (vad as any).frameProcessor.active };
    await Promise.all([lifecycle.setEnabled(false), lifecycle.setEnabled(true), lifecycle.destroy()]);
    return { resumed, destroyed: !vad.listening && !(vad as any).frameProcessor.active };
  } finally {
    stream.getTracks().forEach((track) => track.stop());
    await context.close();
  }
} });
createRoot(document.getElementById("root")!).render(<StrictMode><Harness /></StrictMode>);
