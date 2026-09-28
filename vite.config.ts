import { existsSync, realpathSync } from "node:fs";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteStaticCopy } from "vite-plugin-static-copy";

const ignoredWatchPaths = ["**/.venv/**"];
if (existsSync(".venv")) {
  ignoredWatchPaths.push(`${realpathSync(".venv")}/**`);
}

export default defineConfig({
  plugins: [
    react(),
    viteStaticCopy({
      hook: "closeBundle",
      targets: [
        {
          src: "node_modules/@ricky0123/vad-web/dist/vad.worklet.bundle.min.js",
          dest: "vad",
          rename: { stripBase: true },
        },
        {
          src: "node_modules/@ricky0123/vad-web/dist/silero_vad_v6.onnx",
          dest: "vad",
          rename: { stripBase: true },
        },
        {
          src: "node_modules/onnxruntime-web/dist/*.wasm",
          dest: "vad",
          rename: { stripBase: true },
        },
        {
          src: "node_modules/onnxruntime-web/dist/*.mjs",
          dest: "vad",
          rename: { stripBase: true },
        },
      ],
    }),
  ],
  server: {
    allowedHosts: ["konishi.lan"],
    port: 5173,
    watch: {
      ignored: ignoredWatchPaths,
    },
  },
});
