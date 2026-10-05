import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({
  plugins: [react()],
  define: { "import.meta.env.VITE_API_BASE": JSON.stringify("http://127.0.0.1:8019") },
  server: { host: "127.0.0.1", port: 5179, strictPort: true, watch: { ignored: ["**/.venv/**"] } },
});
