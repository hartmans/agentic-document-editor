import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "src/tests",
  testMatch: "*.spec.ts",
  workers: 1,
  timeout: 20000,
  use: { baseURL: "http://127.0.0.1:5179", browserName: "chromium",
    launchOptions: { args: ["--autoplay-policy=no-user-gesture-required", "--disable-gpu"] } },
  webServer: [
    { command: `${process.env.TEST_PYTHON ?? "python3"} tests/browser/server.py`, url: "http://127.0.0.1:8019/health", timeout: 30000 },
    { command: "yarnpkg exec vite --config src/tests/vite.config.ts", url: "http://127.0.0.1:5179/src/tests/player.html", timeout: 30000 },
  ],
});
