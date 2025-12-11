import { DocumentEditor } from "./components/DocumentEditor";
import { AudioContextProvider } from "./context/AudioContextProvider";

export default function App() {
  return (
    <AudioContextProvider>
      <main style={{ fontFamily: "sans-serif", padding: "1rem" }}>
        <h1>Agentic Document Editor</h1>
        <DocumentEditor />
      </main>
    </AudioContextProvider>
  );
}
