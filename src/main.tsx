import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { AudioContextProvider } from "./context/AudioContextProvider";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <AudioContextProvider>
      <App />
    </AudioContextProvider>
  </React.StrictMode>
);
