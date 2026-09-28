import {
  TranscribeStreamingClient,
  StartStreamTranscriptionCommand,
} from "@aws-sdk/client-transcribe-streaming";
import { fetchAsrConfig, type AsrConfigResponse, type SttToken } from "./api";

export interface SpeechToTextProps {
  sampleRate: number;
  onFinal: (text: string) => void;
  onStop: () => void;
}

export type AwsAsrConfig = {
  type: "aws";
  region: string;
  stsToken: SttToken;
};

export type Gemma4AsrConfig = {
  type: "gemma4";
};

export type AsrConfig = AwsAsrConfig | Gemma4AsrConfig;

type QueuedFloatChunk = { data: Float32Array; isPreroll: boolean };

export abstract class BaseSpeechToText {
  protected readonly props: SpeechToTextProps;
  protected readonly abortController = new AbortController();
  protected readonly targetSampleRate = 16000;
  protected readonly inputGain = 1.0;

  private closedByParent = false;
  private inputFinished = false;
  private readonly maxQueuedSamples: number;
  private readonly resampler: StatefulResampler;

  private queue: QueuedFloatChunk[] = [];
  private queueSamples = 0;
  private pendingResolve: (() => void) | null = null;

  protected constructor(props: SpeechToTextProps) {
    this.props = props;
    this.resampler = new StatefulResampler(props.sampleRate, this.targetSampleRate);
    this.maxQueuedSamples = Math.floor(props.sampleRate * 5);
  }

  protected startSession() {
    void this.start();
  }

  queue_chunk(chunk: Float32Array, isPreroll = false) {
    this.queueChunk(chunk, isPreroll);
  }

  queueChunk(chunk: Float32Array, isPreroll = false) {
    if (this.closedByParent || this.inputFinished) return;
    this.queue.push({ data: chunk, isPreroll });
    this.queueSamples += chunk.length;
    if (this.queueSamples > this.maxQueuedSamples) {
      console.error("STT queue overflow (>5s); aborting session");
      this.abortController.abort();
      return;
    }
    this.wakeIterator();
  }

  close() {
    this.closedByParent = true;
    this.abortController.abort();
    this.wakeIterator();
  }

  finish() {
    if (this.closedByParent || this.inputFinished) return;
    this.inputFinished = true;
    this.wakeIterator();
  }

  protected emitFinal(text: string) {
    if (!this.closedByParent && text) {
      this.props.onFinal(text);
    }
  }

  protected isClosedByParent(): boolean {
    return this.closedByParent;
  }

  protected abstract startTransport(signal: AbortSignal): Promise<void>;
  protected abstract handlePcmChunk(chunk: Uint8Array, isPreroll: boolean): Promise<void> | void;
  protected abstract finishTransport(): Promise<void> | void;

  private async start() {
    let transportPromise: Promise<void> | null = null;
    try {
      this.abortController.signal.addEventListener("abort", () => {
        this.wakeIterator();
      });
      transportPromise = this.startTransport(this.abortController.signal);
      while (!this.closedByParent && !this.abortController.signal.aborted) {
        const next = await this.dequeue();
        if (!next) break;
        const pcm = this.normalizeChunk(next.data);
        if (pcm.byteLength === 0) continue;
        await this.handlePcmChunk(pcm, next.isPreroll);
      }
    } catch (err) {
      if (!this.closedByParent) {
        console.error("STT failed", err);
      }
    } finally {
      try {
        await this.finishTransport();
      } catch (err) {
        if (!this.closedByParent) {
          console.error("STT finish failed", err);
        }
      }
      try {
        await transportPromise;
      } catch (err) {
        if (!this.closedByParent) {
          console.error("STT transport failed", err);
        }
      }
      if (!this.closedByParent) {
        this.props.onStop();
      }
    }
  }

  private wakeIterator() {
    if (this.pendingResolve) {
      this.pendingResolve();
      this.pendingResolve = null;
    }
  }

  private dequeue(): Promise<QueuedFloatChunk | null> {
    if (this.queue.length) {
      const item = this.queue.shift()!;
      this.queueSamples -= item.data.length;
      return Promise.resolve(item);
    }
    if (this.closedByParent || this.inputFinished || this.abortController.signal.aborted) {
      return Promise.resolve(null);
    }
    return new Promise((resolve) => {
      this.pendingResolve = () => resolve(this.dequeue());
    });
  }

  private normalizeChunk(input: Float32Array): Uint8Array {
    const boosted = applyGain(input, this.inputGain);
    const resampled = this.resampler.process(boosted);
    return floatTo16BitPCM(resampled);
  }
}

export class AwsSpeechToText extends BaseSpeechToText {
  private readonly config: AwsAsrConfig;
  private client: TranscribeStreamingClient | null = null;
  private pcmQueue: Uint8Array[] = [];
  private pendingPcmResolve: (() => void) | null = null;
  private transportClosed = false;

  constructor(config: AwsAsrConfig, props: SpeechToTextProps) {
    super(props);
    this.config = config;
    this.startSession();
  }

  protected async startTransport(_signal: AbortSignal): Promise<void> {
    this.client = new TranscribeStreamingClient({
      region: this.config.region,
      credentials: {
        accessKeyId: this.config.stsToken.access_key_id,
        secretAccessKey: this.config.stsToken.secret_access_key,
        sessionToken: this.config.stsToken.session_token,
      },
    });

    const command = new StartStreamTranscriptionCommand({
      LanguageCode: "en-US",
      MediaEncoding: "pcm",
      MediaSampleRateHertz: this.targetSampleRate,
      AudioStream: this.pcmAudioIterator(),
    });

    try {
      const response = await this.client.send(command);
      for await (const evt of response.TranscriptResultStream ?? []) {
        const results = evt?.TranscriptEvent?.Transcript?.Results;
        if (!results) continue;
        for (const res of results) {
          const text = res.Alternatives?.[0]?.Transcript ?? "";
          if (!text) continue;
          if (!res.IsPartial) {
            this.emitFinal(text);
          }
        }
      }
    } catch (streamErr) {
      if (!this.isClosedByParent()) {
        console.error("AWS STT stream error", streamErr);
      }
    } finally {
      this.client?.destroy();
      this.client = null;
    }
  }

  protected handlePcmChunk(chunk: Uint8Array): void {
    this.pcmQueue.push(chunk);
    if (this.pendingPcmResolve) {
      this.pendingPcmResolve();
      this.pendingPcmResolve = null;
    }
  }

  protected async finishTransport(): Promise<void> {
    this.transportClosed = true;
    if (this.pendingPcmResolve) {
      this.pendingPcmResolve();
      this.pendingPcmResolve = null;
    }
  }

  private async *pcmAudioIterator() {
    while (true) {
      const next = await this.dequeuePcm();
      if (!next) return;
      yield { AudioEvent: { AudioChunk: next } };
    }
  }

  private dequeuePcm(): Promise<Uint8Array | null> {
    if (this.pcmQueue.length) {
      return Promise.resolve(this.pcmQueue.shift() ?? null);
    }
    if (this.transportClosed) {
      return Promise.resolve(null);
    }
    return new Promise((resolve) => {
      this.pendingPcmResolve = () => resolve(this.dequeuePcm());
    });
  }
}

export class Gemma4SpeechToText extends BaseSpeechToText {
  private socket: WebSocket | null = null;
  private openPromise: Promise<void> | null = null;
  private completionPromise: Promise<void> | null = null;
  private resolveCompletion: (() => void) | null = null;
  private rejectCompletion: ((err: unknown) => void) | null = null;
  private endSent = false;
  private resultReceived = false;

  constructor(config: Gemma4AsrConfig, props: SpeechToTextProps) {
    super(props);
    void config;
    this.startSession();
  }

  protected startTransport(_signal: AbortSignal): Promise<void> {
    const url = buildGemma4WebSocketUrl();
    this.socket = new WebSocket(url);
    this.socket.binaryType = "arraybuffer";

    this.openPromise = new Promise((resolve, reject) => {
      if (!this.socket) {
        reject(new Error("Gemma4 websocket was not created"));
        return;
      }
      this.socket.addEventListener("open", () => {
        this.socket?.send(JSON.stringify({ type: "start" }));
        resolve();
      });
      this.socket.addEventListener("error", () => {
        reject(new Error("Gemma4 websocket failed to open"));
      });
    });

    this.completionPromise = new Promise((resolve, reject) => {
      this.resolveCompletion = resolve;
      this.rejectCompletion = reject;
    });

    this.socket.addEventListener("message", (event) => {
      if (typeof event.data !== "string") {
        return;
      }
      const payload = JSON.parse(event.data) as
        | { type: "started" }
        | { type: "result"; instructions: string }
        | { type: "error"; detail?: string };

      if (payload.type === "started") {
        return;
      }
      if (payload.type === "result") {
        this.resultReceived = true;
        this.emitFinal(payload.instructions);
        this.resolveCompletion?.();
        this.resolveCompletion = null;
        this.rejectCompletion = null;
        this.socket?.close();
        return;
      }
      this.rejectCompletion?.(new Error(payload.detail || "Gemma4 websocket error"));
      this.resolveCompletion = null;
      this.rejectCompletion = null;
      this.socket?.close();
    });

    this.socket.addEventListener("close", () => {
      if (this.resolveCompletion === null && this.rejectCompletion === null) {
        return;
      }
      if (this.isClosedByParent() || this.resultReceived) {
        this.resolveCompletion?.();
      } else {
        this.rejectCompletion?.(new Error("Gemma4 websocket closed before result"));
      }
      this.resolveCompletion = null;
      this.rejectCompletion = null;
    });

    this.socket.addEventListener("error", () => {
      if (this.rejectCompletion) {
        this.rejectCompletion(new Error("Gemma4 websocket transport error"));
        this.resolveCompletion = null;
        this.rejectCompletion = null;
      }
    });

    return this.completionPromise;
  }

  protected async handlePcmChunk(chunk: Uint8Array): Promise<void> {
    await this.openPromise;
    this.socket?.send(chunk);
  }

  protected async finishTransport(): Promise<void> {
    const socket = this.socket;
    if (!socket) return;

    try {
      await this.openPromise;
    } catch {
      socket.close();
      return;
    }

    if (this.isClosedByParent()) {
      socket.close();
      return;
    }

    if (!this.endSent && socket.readyState === WebSocket.OPEN) {
      this.endSent = true;
      socket.send(JSON.stringify({ type: "end" }));
    }
  }
}

export function createSpeechToText(
  config: AsrConfig,
  props: SpeechToTextProps,
): BaseSpeechToText {
  switch (config.type) {
    case "aws":
      return new AwsSpeechToText(config, props);
    case "gemma4":
      return new Gemma4SpeechToText(config, props);
  }
}

export async function createSpeechToTextFromBackend(
  props: SpeechToTextProps,
): Promise<BaseSpeechToText> {
  const config = await fetchAsrConfig();
  return createSpeechToText(asrConfigFromResponse(config), props);
}

/**
 * Compatibility wrapper for the current frontend, which still constructs
 * `new SpeechToText(props)` directly. Once the rest of the app is refactored,
 * call `createSpeechToText` with an explicit `AsrConfig` instead.
 */
export class SpeechToText {
  private impl: BaseSpeechToText | null = null;
  private pendingChunks: QueuedFloatChunk[] = [];
  private closed = false;
  private finishRequested = false;

  constructor(private readonly props: SpeechToTextProps) {
    void this.initialize();
  }

  queue_chunk(chunk: Float32Array, isPreroll = false) {
    if (this.closed) return;
    if (!this.impl) {
      this.pendingChunks.push({ data: chunk, isPreroll });
      return;
    }
    this.impl.queue_chunk(chunk, isPreroll);
  }

  close() {
    this.closed = true;
    this.pendingChunks = [];
    this.impl?.close();
    this.impl = null;
  }

  finish() {
    if (this.closed) return;
    this.finishRequested = true;
    this.impl?.finish();
  }

  private async initialize() {
    try {
      const impl = await createSpeechToTextFromBackend(this.props);
      if (this.closed) {
        impl.close();
        return;
      }
      this.impl = impl;
      for (const chunk of this.pendingChunks) {
        this.impl.queue_chunk(chunk.data, chunk.isPreroll);
      }
      this.pendingChunks = [];
      if (this.finishRequested) this.impl.finish();
    } catch (err) {
      if (!this.closed) {
        console.error("Failed to initialize SpeechToText", err);
        this.props.onStop();
      }
    }
  }
}

function buildGemma4WebSocketUrl(): string {
  const rawBase = import.meta.env.VITE_API_BASE as string | undefined;
  if (rawBase && rawBase.trim()) {
    const base = rawBase.trim().replace(/\/+$/, "");
    if (base.startsWith("https://")) {
      return `${base.replace(/^https:\/\//, "wss://")}/ws/asr`;
    }
    if (base.startsWith("http://")) {
      return `${base.replace(/^http:\/\//, "ws://")}/ws/asr`;
    }
    if (base.startsWith("wss://") || base.startsWith("ws://")) {
      return `${base}/ws/asr`;
    }
  }
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/api/ws/asr`;
}

function asrConfigFromResponse(config: AsrConfigResponse): AsrConfig {
  switch (config.type) {
    case "aws":
      return {
        type: "aws",
        region: config.region,
        stsToken: config.stsToken as SttToken,
      };
    case "gemma4":
      return { type: "gemma4" };
  }
}

function floatTo16BitPCM(input: Float32Array): Uint8Array {
  const buffer = new ArrayBuffer(input.length * 2);
  const view = new DataView(buffer);
  let offset = 0;
  for (let i = 0; i < input.length; i++, offset += 2) {
    const sample = Math.max(-1, Math.min(1, input[i]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
  }
  return new Uint8Array(buffer);
}

class StatefulResampler {
  private readonly ratio: number;
  private phase = 0;
  private lastSample = 0;

  constructor(private readonly sourceRate: number, private readonly targetRate: number) {
    this.ratio = sourceRate / targetRate;
  }

  process(input: Float32Array): Float32Array {
    if (this.sourceRate === this.targetRate) return input;
    if (input.length === 0) return new Float32Array();

    const out: number[] = [];
    let pos = this.phase;
    const firstSample = pos < 1 ? this.lastSample : input[0];

    while (pos < input.length) {
      const i = Math.floor(pos);
      const frac = pos - i;
      const s0 = i >= 0 ? (i === 0 ? firstSample : input[i]) : firstSample;
      const s1 = i + 1 < input.length ? input[i + 1] : input[input.length - 1];
      out.push(s0 + (s1 - s0) * frac);
      pos += this.ratio;
    }

    this.phase = pos - input.length;
    this.lastSample = input[input.length - 1];
    return new Float32Array(out);
  }
}

function applyGain(input: Float32Array, gain: number): Float32Array {
  if (gain === 1) return input;
  const out = new Float32Array(input.length);
  for (let i = 0; i < input.length; i++) {
    out[i] = input[i] * gain;
  }
  return out;
}
