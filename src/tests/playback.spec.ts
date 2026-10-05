import { test, expect } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  await page.goto("/src/tests/player.html");
  await page.waitForFunction(() => Boolean((window as any).engine));
});

async function start(page: import("@playwright/test").Page, mode = "complete", offset = 0) {
  await page.evaluate(({mode, offset}) => {
    const w = window as any;
    w.ended = 0; w.failure = null; w.started = null;
    w.engine.playChunk(0, `http://127.0.0.1:8019/audio?mode=${mode}`, offset, {
      onEnded: () => w.ended++, onError: (error: Error) => w.failure = error.message,
    }).then((value: boolean) => w.started = value).catch((error: Error) => w.failure = error.message);
  }, {mode, offset});
}

test("starts before response completion; pause captures stable media time", async ({ page }) => {
  await start(page, "slow");
  await expect.poll(() => page.evaluate(() => (window as any).started), {timeout: 2500}).toBe(true);
  await expect.poll(() => page.evaluate(() => (window as any).engine.getCurrentPlaybackInfo().offsetSeconds)).toBeGreaterThan(0.25);
  const point = await page.evaluate(() => (window as any).engine.stop());
  expect(point.playingChunkId).toBe(0);
  expect(point.offsetSeconds).toBeGreaterThan(0.25);
  expect(point.offsetSeconds).toBeLessThan(4);
  await page.waitForTimeout(200);
  expect(await page.evaluate(() => (window as any).ended)).toBe(0);
});

test("a mid-chunk resume waits for its buffered position", async ({ page }) => {
  await start(page, "slow", 4.5);
  await page.waitForTimeout(500);
  expect(await page.evaluate(() => (window as any).started)).toBe(null);
  await expect.poll(() => page.evaluate(() => (window as any).started)).toBe(true);
  const point = await page.evaluate(() => (window as any).engine.stop());
  expect(point.offsetSeconds).toBeGreaterThanOrEqual(4.5);
  expect(point.offsetSeconds).toBeLessThan(5);
});

test("stopping startup settles it and prevents stale completion", async ({ page }) => {
  await start(page, "pending");
  await page.evaluate(() => (window as any).engine.stop());
  await expect.poll(() => page.evaluate(() => (window as any).started)).toBe(false);
  await start(page);
  await expect.poll(() => page.evaluate(() => (window as any).started)).toBe(true);
  await page.evaluate(() => (window as any).engine.stop());
  expect(await page.evaluate(() => (window as any).ended)).toBe(0);
});

test("abrupt HTTP disconnect follows native media events", async ({ page }) => {
  await page.evaluate(() => {
    const w = window as any;
    w.nativeEnded = 0; w.nativeErrors = 0;
    document.addEventListener("ended", () => w.nativeEnded++, true);
    document.addEventListener("error", (event) => {
      if (event.target instanceof HTMLAudioElement) w.nativeErrors++;
    }, true);
    // Audio() elements are detached; track their native events directly.
    const OriginalAudio = window.Audio;
    w.Audio = function () {
      const audio = new OriginalAudio();
      audio.addEventListener("ended", () => w.nativeEnded++);
      audio.addEventListener("error", () => w.nativeErrors++);
      return audio;
    };
  });
  await start(page, "broken");
  await expect.poll(() => page.evaluate(() => (window as any).started)).toBe(true);
  await page.waitForTimeout(4500);
  const result = await page.evaluate(() => {
    const w = window as any;
    return { ended: w.ended, failure: w.failure, nativeEnded: w.nativeEnded, nativeErrors: w.nativeErrors };
  });
  expect(result.ended).toBe(result.nativeEnded);
  expect(result.failure !== null).toBe(result.nativeErrors > 0);
  await page.evaluate(() => (window as any).engine.stop());
});

test("HTTP EOF with truncated Opus propagates natural end", async ({ page }) => {
  await start(page, "truncated");
  await expect.poll(() => page.evaluate(() => (window as any).started)).toBe(true);
  await expect.poll(() => page.evaluate(() => (window as any).ended), {timeout: 10000}).toBe(1);
  expect(await page.evaluate(() => (window as any).failure)).toBe(null);
});

test("successful EOF advances naturally once", async ({ page }) => {
  await start(page, "complete", 3.7);
  await expect.poll(() => page.evaluate(() => (window as any).ended)).toBe(1);
  expect(await page.evaluate(() => (window as any).failure)).toBe(null);
});

test("DocumentPlayback starts without marks and resolves interruption after marks", async ({ page }) => {
  await page.goto("/src/tests/player.html?document=delayed");
  await page.getByRole("button", {name: "Start", exact: true}).click();
  await expect(page.locator("#playing")).toHaveText("true", {timeout: 2500});
  await page.waitForTimeout(700);
  await page.getByRole("button", {name: "Interrupt", exact: true}).click();
  await expect(page.locator("#playing")).toHaveText("false");
  expect(await page.locator("#resolved").textContent()).toBe("");
  await expect(page.locator("#resolved")).toContainText('"docCharPosition":6');
});


test("StrictMode initial playback makes one audio request and silent completion fires once", async ({ page }) => {
  const requests: string[] = [];
  page.on("request", (request) => requests.push(request.url()));
  await page.goto("/src/tests/player.html?autoplay");
  await expect(page.locator("#playing")).toHaveText("true");
  expect(requests.filter((url) => url.includes("/chunks/complete/0/audio"))).toHaveLength(1);
  await page.goto("/src/tests/player.html?document=silent&autoplay");
  await expect(page.locator("#ended")).toHaveText("1");
  await page.waitForTimeout(100);
  await expect(page.locator("#ended")).toHaveText("1");
});

test("a changed starting position seeks without an imperative handle or command token", async ({ page }) => {
  await page.getByRole("button", {name: "Start", exact: true}).click();
  await expect(page.locator("#playing")).toHaveText("true");
  await page.getByRole("button", {name: "Seek", exact: true}).click();
  await expect(page.locator("#playing")).toHaveText("true");
  await page.getByRole("button", {name: "Interrupt", exact: true}).click();
  await expect(page.locator("#resolved")).toContainText('"docCharPosition":6');
});

test("pausing a pending start prevents late audio playback", async ({ page }) => {
  await page.goto("/src/tests/player.html?document=pending");
  const request = page.waitForRequest((request) => request.url().includes("/pending/0/audio"));
  await page.getByRole("button", {name: "Start", exact: true}).click();
  await request;
  await page.getByRole("button", {name: "Interrupt", exact: true}).click();
  await expect(page.locator("#resolved")).toContainText('"docCharPosition":0');
  await page.waitForTimeout(5200);
  await expect(page.locator("#playing")).toHaveText("false");
  await expect(page.locator("#ended")).toHaveText("0");
});
