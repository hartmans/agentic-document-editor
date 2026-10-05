import { readFileSync } from "node:fs";
import { basename } from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({
  plugins: [react(), {
    name: "test-vad-assets",
    configureServer(server) {
      // Serve ONNX's dynamic module imports verbatim, before Vite transforms them.
      server.middlewares.use((request, response, next) => {
        if (!request.url?.startsWith("/vad/")) return next();
        const name = basename(request.url.split("?")[0]);
        const directory = name.startsWith("ort-") ? "onnxruntime-web" : "@ricky0123/vad-web";
        try {
          const data = readFileSync(`node_modules/${directory}/dist/${name}`);
          response.setHeader("Content-Type", name.endsWith(".wasm") ? "application/wasm"
            : name.endsWith(".onnx") ? "application/octet-stream" : "text/javascript");
          response.end(data);
        } catch (error) { next(error); }
      });
    },
  }],
  define: { "import.meta.env.VITE_API_BASE": JSON.stringify("http://127.0.0.1:8019") },
  server: { host: "127.0.0.1", port: 5179, strictPort: true, watch: { ignored: ["**/.venv/**"] } },
});
