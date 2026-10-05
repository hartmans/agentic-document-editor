import { test, expect, type Page } from "@playwright/test";

async function utterance(page: Page, text: string) {
  await page.evaluate(() => (window as any).voice.onInterrupt());
  // Interruption and ASR completion occur in separate event turns.
  await page.evaluate((text) => (window as any).voice.onInstructions(text), text);
}

async function audioState(page: Page) {
  // AudioEngine creates detached elements. Capture those as they are created.
  return page.evaluate(() => (window as any).audioElements.map((audio: HTMLAudioElement) => ({
    src: audio.src, paused: audio.paused, time: audio.currentTime,
  })));
}

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    // Expose the voice boundary through React's supported devtools hook. Tests send
    // completed transcripts to the real editor without replacing any components.
    Object.assign(window, { __REACT_DEVTOOLS_GLOBAL_HOOK__: {
      supportsFiber: true,
      renderers: new Map(),
      inject(renderer: unknown) { this.renderers.set(1, renderer); return 1; },
      onCommitFiberRoot: (_id: number, root: any) => {
        const visit = (fiber: any) => {
          if (!fiber) return;
          if (fiber.type?.name === "VoiceInput") Object.assign(window, { voice: fiber.memoizedProps });
          visit(fiber.child);
          visit(fiber.sibling);
        };
        visit(root.current);
      },
      onCommitFiberUnmount: () => {},
    } });

    const OriginalAudio = window.Audio;
    (window as any).audioElements = [];
    (window as any).Audio = function (...args: ConstructorParameters<typeof Audio>) {
      const audio = new OriginalAudio(...args);
      (window as any).audioElements.push(audio);
      return audio;
    };
  });
  await page.goto("/src/tests/editor.html");
  await expect(page.getByRole("button", {name: "Play", exact: true})).toBeVisible();
});

test("ignored utterances preserve idle and explicitly paused playback", async ({ page }) => {
  await utterance(page, "");
  expect((await audioState(page)).every((audio: any) => audio.paused)).toBe(true);
  await page.getByRole("button", {name: "Play", exact: true}).click();
  await expect.poll(async () => (await audioState(page)).some((a: any) => !a.paused)).toBe(true);
  await page.getByRole("button", {name: "Pause", exact: true}).click();
  await utterance(page, "  ");
  await page.waitForTimeout(300);
  expect((await audioState(page)).every((audio: any) => audio.paused)).toBe(true);
});

test("ignored utterance resumes the interrupted document at its position", async ({ page }) => {
  await page.getByRole("button", {name: "Play", exact: true}).click();
  await expect.poll(async () => (await audioState(page)).some((a: any) => a.time > 0.8)).toBe(true);
  const before = (await audioState(page)).find((a: any) => !a.paused).time;
  await utterance(page, "");
  await expect.poll(async () => (await audioState(page)).some((a: any) => !a.paused && a.time >= before)).toBe(true);
});

for (const disposition of ["pause", "play"] as const) {
  test(`ignored utterance resumes the reply, then respects its ${disposition} disposition`, async ({ page }) => {
    await utterance(page, disposition);
    await expect.poll(async () => (await audioState(page)).some((a: any) => a.src.includes("/reply/") && a.time > 0.8)).toBe(true);
    const before = (await audioState(page)).find((a: any) => a.src.includes("/reply/") && !a.paused).time;
    await utterance(page, "");
    await expect.poll(async () => (await audioState(page)).some((a: any) => a.src.includes("/reply/") && !a.paused && a.time >= before)).toBe(true);
    await expect.poll(async () => (await audioState(page)).filter((a: any) => a.src.includes("/reply/")).every((a: any) => a.paused)).toBe(true);
    if (disposition === "play") {
      await expect.poll(async () => (await audioState(page)).some((a: any) => a.src.includes("/complete/") && !a.paused)).toBe(true);
    } else {
      await utterance(page, "");
      expect((await audioState(page)).filter((a: any) => a.src.includes("/chunks/")).every((a: any) => a.paused)).toBe(true);
    }
  });
}

test("real MicVAD remains active after overlapping pause/start requests and tears down in order", async ({ page }) => {
  const result = await page.evaluate(() => (window as any).checkVadTransitions());
  expect(result).toEqual({ resumed: { listening: true, active: true }, destroyed: true });
});
