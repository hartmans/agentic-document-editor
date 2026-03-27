import { existsSync, realpathSync } from "node:fs";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const ignoredWatchPaths = ["**/.venv/**"];
if (existsSync(".venv")) {
  ignoredWatchPaths.push(`${realpathSync(".venv")}/**`);
}

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    watch: {
      ignored: ignoredWatchPaths,
    },
  },
});
