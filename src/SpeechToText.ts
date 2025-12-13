import {
  TranscribeStreamingClient,
  StartStreamTranscriptionCommand,
} from "@aws-sdk/client-transcribe-streaming";
import { fetchSttToken } from "./api";

export interface SpeechToTextProps {
  mediaStream: MediaStream;
  preroll: Float32Array;
  sampleRate: number;
  getRms: () => number;
  vadThreshold: number;
  onFinal: (text: string) => void;
  onStop: () => void;
}

/**
 * SpeechToText manages a single Transcribe Streaming session.
 * It owns its AudioContext/processor and ensures onStop is fired at most once
 * when the session ends naturally or due to error. Parent-initiated close()
 * will not call onStop.
 */
export class SpeechToText {
  private props: SpeechToTextProps;
  private stopTimer: ReturnType<typeof setTimeout> | null = null;
  private silenceTimer: ReturnType<typeof setInterval> | null = null;
  private stopped = false;
  private closedByParent = false;
  private abortController = new AbortController();
  private lastTranscript: number | null = null;

  constructor(props: SpeechToTextProps) {
    this.props = props;
    void this.start();
  }

  /** Called by parent to terminate without firing onStop. */
  close() {
    this.closedByParent = true;
    this.abortController.abort();
    this.cleanupTimers();
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

      const audioIterator = this.pcmAudioIterator(
        this.props.mediaStream,
        this.props.preroll,
        this.props.sampleRate,
        this.abortController.signal
      );

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
    // Hard cap 10s; soft cap 4s after first transcript if RMS low.
    this.stopTimer = setTimeout(() => this.handleIdle(true), 10000);
    this.silenceTimer = setInterval(() => this.handleIdle(false), 500);
  }

  private handleIdle(hard: boolean) {
    if (this.closedByParent || this.stopped) return;
    const idleMs = Date.now() - (this.lastTranscript ?? Date.now());
    if (hard && idleMs >= 10000) {
      this.abortController.abort();
      return;
    }
    if (!hard && this.lastTranscript && idleMs >= 4000 && this.props.getRms() < this.props.vadThreshold) {
      this.abortController.abort();
    }
  }

  private cleanupTimers() {
    if (this.stopTimer) clearTimeout(this.stopTimer);
    if (this.silenceTimer) clearInterval(this.silenceTimer);
    this.stopTimer = null;
    this.silenceTimer = null;
  }

  private async *pcmAudioIterator(
    mediaStream: MediaStream,
    preroll: Float32Array,
    sampleRate: number,
    signal: AbortSignal
  ) {
    const audioCtx = new AudioContext({ sampleRate });
    await audioCtx.resume().catch(() => {});
    const source = audioCtx.createMediaStreamSource(mediaStream);
    const processor = audioCtx.createScriptProcessor(4096, 1, 1);
    let done = false;

    if (signal.aborted) {
      audioCtx.close().catch(() => {});
      return;
    }

    signal.addEventListener("abort", () => {
      done = true;
    });

    if (preroll.length) {
      const prerollResampled = downsampleTo16k(preroll, sampleRate);
      const prerollPcm = floatTo16BitPCM(prerollResampled);
      yield { AudioEvent: { AudioChunk: prerollPcm } };
    }

    source.connect(processor);
    processor.connect(audioCtx.destination);

    const queue: Uint8Array[] = [];
    processor.onaudioprocess = (e) => {
      if (done) return;
      const data = e.inputBuffer.getChannelData(0);
      const resampled = downsampleTo16k(data, sampleRate);
      queue.push(floatTo16BitPCM(resampled));
    };

    try {
      while (!done) {
        if (queue.length) {
          yield { AudioEvent: { AudioChunk: queue.shift()! } };
        } else {
          await new Promise((resolve) => setTimeout(resolve, 10));
        }
      }
    } finally {
      processor.disconnect();
      source.disconnect();
      audioCtx.close().catch(() => {});
    }
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

function downsampleTo16k(input: Float32Array, sourceRate: number): Float32Array {
  const targetRate = 16000;
  if (sourceRate === targetRate) return input;
  const ratio = sourceRate / targetRate;
  const newLen = Math.floor(input.length / ratio);
  const output = new Float32Array(newLen);
  let offsetResult = 0;
  let offsetBuffer = 0;
  while (offsetResult < output.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
    let accum = 0;
    let count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < input.length; i++) {
      accum += input[i];
      count++;
    }
    output[offsetResult] = count ? accum / count : 0;
    offsetResult++;
    offsetBuffer = nextOffsetBuffer;
  }
  return output;
}
