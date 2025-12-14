import {
  TranscribeStreamingClient,
  StartStreamTranscriptionCommand,
} from "@aws-sdk/client-transcribe-streaming";
import { fetchSttToken } from "./api";

export interface SpeechToTextProps {
  sampleRate: number;
  getRms: () => number;
  vadThreshold: number;
  onFinal: (text: string) => void;
  onStop: () => void;
}

/**
 * SpeechToText manages a single Transcribe Streaming session.
 * It consumes Float32 frames provided by the parent via queue_chunk and
 * streams them to AWS Transcribe. onStop is fired at most once when the
 * session ends naturally or due to error. Parent-initiated close() will not
 * call onStop.
 */
export class SpeechToText {
  private props: SpeechToTextProps;
  private silenceTimer: ReturnType<typeof setInterval> | null = null;
  private stopped = false;
  private closedByParent = false;
  private abortController = new AbortController();
  private lastTranscript: number | null = null;
  private readonly sessionStart = Date.now();
  private queue: { data: Float32Array; isPreroll: boolean }[] = [];
  private queueBytes = 0; // in samples *4 (Float32)
  private pendingResolve: (() => void) | null = null;
  private readonly maxQueuedSamples: number;

  private readonly resampler: StatefulResampler;
  private readonly INPUT_GAIN = 2.0;

  constructor(props: SpeechToTextProps) {
    this.props = props;
    this.resampler = new StatefulResampler(props.sampleRate, 16000);
    this.maxQueuedSamples = Math.floor(props.sampleRate * 5); // 5s safety cap
    void this.start();
  }

  /** Enqueue captured audio (Float32, source sampleRate). */
  queue_chunk(chunk: Float32Array, isPreroll = false) {
    if (this.stopped || this.closedByParent) return;
    this.queue.push({ data: chunk, isPreroll });
    this.queueBytes += chunk.length;
    if (this.queueBytes > this.maxQueuedSamples) {
      console.error("STT queue overflow (>5s); aborting session");
      this.abortController.abort();
      return;
    }
    if (this.pendingResolve) {
      this.pendingResolve();
      this.pendingResolve = null;
    }
  }

  /** Called by parent to terminate without firing onStop. */
  close() {
    this.closedByParent = true;
    this.abortController.abort();
    this.cleanupTimers();
    this.wakeIterator();
  }

  private async start() {
    try {
      const token = await fetchSttToken();
      const client = new TranscribeStreamingClient({
        region: "us-east-1",
        credentials: {
          accessKeyId: token.access_key_id,
          secretAccessKey: token.secret_access_key,
          sessionToken: token.session_token,
        },
      });

      const audioIterator = this.pcmAudioIterator(this.abortController.signal);

      this.startIdleTimers();

      const command = new StartStreamTranscriptionCommand({
        LanguageCode: "en-US",
        MediaEncoding: "pcm",
        MediaSampleRateHertz: 16000,
        AudioStream: audioIterator,
      });

      const response = await client.send(command);
      console.log("STT Started", { t: Date.now() });
      try {
        for await (const evt of response.TranscriptResultStream ?? []) {
          const results = evt?.TranscriptEvent?.Transcript?.Results;
          console.log("TranscriptEvent", {
            t: Date.now(),
            resultsLen: results?.length ?? 0,
          });
          if (!results) continue;
          for (const res of results) {
            const text = res.Alternatives?.[0]?.Transcript ?? "";
            if (!text) continue;
            this.lastTranscript = Date.now();
            if (!res.IsPartial) {
              this.props.onFinal(text);
            }
          }
        }
      } catch (streamErr) {
        console.error("STT stream error", streamErr);
      }
      console.log("STT stream ended", {
        t: Date.now(),
        metadata: response?.$metadata,
      });
    } catch (err) {
      console.error("STT failed", err);
    } finally {
      this.cleanupTimers();
      if (!this.stopped && !this.closedByParent) {
        this.stopped = true;
        this.props.onStop();
      }
    }
  }

  private startIdleTimers() {
    this.cleanupTimers();
    // Evaluate both hard (10s since last transcript) and soft (4s since last transcript & below VAD)
    this.silenceTimer = setInterval(() => this.handleIdle(), 500);
  }

  private handleIdle() {
    if (this.closedByParent || this.stopped) return;
    const now = Date.now();
    const last = this.lastTranscript ?? this.sessionStart;
    const idleMs = now - last;
    // Hard cap: 10s without any new transcript results.
    if (idleMs >= 10000) {
      this.abortController.abort();
      return;
    }
    // Soft cap: after at least one transcript, 4s idle AND below VAD threshold.
    if (this.lastTranscript && idleMs >= 4000 && this.props.getRms() < this.props.vadThreshold) {
      this.abortController.abort();
    }
  }

  private cleanupTimers() {
    if (this.silenceTimer) clearInterval(this.silenceTimer);
    this.silenceTimer = null;
  }

  private async *pcmAudioIterator(signal: AbortSignal) {
    signal.addEventListener("abort", () => {
      this.stopped = true;
      this.wakeIterator();
    });

    try {
      while (!this.stopped && !this.closedByParent && !signal.aborted) {
        const next = await this.dequeue();
        if (!next) continue;
        const boosted = applyGain(next.data, this.INPUT_GAIN);
        const resampled = this.resampler.process(boosted);
        const pcm = floatTo16BitPCM(resampled);
        yield { AudioEvent: { AudioChunk: pcm } };
      }
    } finally {
    }
  }

  private wakeIterator() {
    if (this.pendingResolve) {
      this.pendingResolve();
      this.pendingResolve = null;
    }
  }

  private dequeue(): Promise<{ data: Float32Array; isPreroll: boolean } | null> {
    if (this.queue.length) {
      const item = this.queue.shift()!;
      this.queueBytes -= item.data.length;
      return Promise.resolve(item);
    }
    if (this.stopped || this.closedByParent || this.abortController.signal.aborted) {
      return Promise.resolve(null);
    }
    return new Promise((resolve) => {
      this.pendingResolve = () => resolve(this.dequeue());
    });
  }
}

function floatTo16BitPCM(input: Float32Array): Uint8Array {
  const buffer = new ArrayBuffer(input.length * 2);
  const view = new DataView(buffer);
  let offset = 0;
  for (let i = 0; i < input.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, input[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Uint8Array(buffer);
}

/**
 * Stateful linear resampler that preserves fractional phase across calls,
 * eliminating dropped/duplicated samples at buffer boundaries.
 */
class StatefulResampler {
  private readonly ratio: number;
  private phase = 0; // fractional source index carried across buffers
  private lastSample = 0;

  constructor(private readonly sourceRate: number, private readonly targetRate: number) {
    this.ratio = sourceRate / targetRate;
  }

  process(input: Float32Array): Float32Array {
    if (this.sourceRate === this.targetRate) return input;
    if (input.length === 0) return new Float32Array();

    const out: number[] = [];
    let pos = this.phase;

    // To support interpolation at the very start, use lastSample from prior buffer if needed.
    const firstSample = pos < 1 ? this.lastSample : input[0];

    while (pos < input.length) {
      const i = Math.floor(pos);
      const frac = pos - i;
      const s0 = i >= 0 ? (i === 0 ? firstSample : input[i]) : firstSample;
      const s1 = i + 1 < input.length ? input[i + 1] : input[input.length - 1];
      const sample = s0 + (s1 - s0) * frac;
      out.push(sample);
      pos += this.ratio;
    }

    // Carry fractional position and last sample forward for continuity.
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
