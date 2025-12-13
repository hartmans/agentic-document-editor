import { useEffect, useRef, useState, useEffectEvent} from "react";
import {
  TranscribeStreamingClient,
  StartStreamTranscriptionCommand,
} from "@aws-sdk/client-transcribe-streaming";
import { fetchSttToken } from "../api";

interface SpeechToTextProps {
  mediaStream: MediaStream;
  preroll: Float32Array;
  sampleRate: number;
  getRms: () => number;
  vadThreshold: number;
  onFinal: (text: string) => void;
  onStop: () => void;
}

/**
 * Streams audio (preroll + live) to AWS Transcribe and emits text results.
 * Stops when either:
 *  - 4s of no transcript and RMS below threshold, or
 *  - 10s of no transcript, whichever comes first.
 */
export function SpeechToText({
  mediaStream,
  preroll,
  sampleRate,
  getRms,
  vadThreshold,
  onFinal,
  onStop,
}: SpeechToTextProps) {
  const [running, setRunning] = useState(true);
  const stopTimerRef = useRef<NodeJS.Timeout | null>(null);
  const silenceTimerRef = useRef<NodeJS.Timeout | null>(null);
  const lastTranscriptRef = useRef<number | null>(null);
  const startedRef = useRef<number>(Date.now());
  const seenTranscriptEventRef = useRef(false);
  const isRunning = useEffectEvent(() => running);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const TARGET_RATE = 16000;
        const token = await fetchSttToken();
        const client = new TranscribeStreamingClient({
          region: "us-east-1",
          credentials: {
            accessKeyId: token.access_key_id,
            secretAccessKey: token.secret_access_key,
            sessionToken: token.session_token,
          },
        });

        const audioIterator = pcmAudioIterator(mediaStream, preroll, sampleRate);
        setRunning(true);
        if (!isRunning()) {
        console.error('not running');
        return;
        }
        const command = new StartStreamTranscriptionCommand({
          LanguageCode: "en-US",
          MediaEncoding: "pcm",
          MediaSampleRateHertz: TARGET_RATE,
          AudioStream: audioIterator,
        });

        const response = await client.send(command);
        console.log("STT Started", { t: Date.now() });
        try {
          for await (const evt of response.TranscriptResultStream ?? []) {
            seenTranscriptEventRef.current = true;
            const results = evt?.TranscriptEvent?.Transcript?.Results;
            console.log("TranscriptEvent", {
              t: Date.now(),
              resultsLen: results?.length ?? 0,
            });
            if (!results) continue;
            for (const res of results) {
              console.log("Got STT result", {
                isPartial: res.IsPartial,
                len: res.Alternatives?.[0]?.Transcript?.length ?? 0,
              });
              const text =
                res.Alternatives && res.Alternatives[0]
                  ? res.Alternatives[0].Transcript
                  : "";
              if (!text) continue;
              lastTranscriptRef.current = Date.now();
              if (!res.IsPartial) {
                onFinal(text);
              }
            }
            if (cancelled) {
              console.debug('STT cancelled');
              return;
            }
            if ( !isRunning()) {
            console.log('running false');
            break;
            }
          }
        } catch (streamErr) {
          console.error("STT stream error", streamErr);
        } finally {
        audioIterator.return();
        }
        const meta = response?.$metadata || {};
        console.log("STT stream ended", {
          t: Date.now(),
          httpStatus: meta.httpStatusCode,
          requestId: meta.requestId,
          attempts: meta.attempts,
          totalRetryDelay: meta.totalRetryDelay,
          rawMetadata: meta,
        });
      } catch (err) {
        console.error("STT failed", err);
      } finally {
        console.debug("STT finished", {
          t: Date.now(),
          seenTranscript: seenTranscriptEventRef.current,
        });
        cleanupTimers();
        setRunning(false);
        onStop();
      }
    })();

    startStopTimers();

    return () => {
        console.debug('stt unmounted');
      cancelled = true;
      cleanupTimers();
    };
  }, []);

  function startStopTimers() {
    cleanupTimers();
    stopTimerRef.current = setTimeout(() => endIfIdle(true), 10000);
    silenceTimerRef.current = setInterval(() => endIfIdle(false), 500);
  }

  function endIfIdle(isHardLimit: boolean) {
    const idleMs = Date.now() - (lastTranscriptRef.current || startedRef.current);
      if (isHardLimit && idleMs >= 10000) {
          console.debug('hard limit');
      cleanupTimers();
      setRunning(false);
      onStop();
      return;
    }
    if (!lastTranscriptRef.current) return;
    if (!isHardLimit && idleMs >= 4000 && getRms() < vadThreshold) {
      console.debug('soft limit');
      cleanupTimers();
      setRunning(false);
      onStop();
    }
  }

  function cleanupTimers() {
    if (stopTimerRef.current) clearTimeout(stopTimerRef.current);
    if (silenceTimerRef.current) clearInterval(silenceTimerRef.current);
    stopTimerRef.current = null;
    silenceTimerRef.current = null;
  }

  return null;
}

async function* pcmAudioIterator(
  mediaStream: MediaStream,
  preroll: Float32Array,
  sampleRate: number
) {
  const audioCtx = new AudioContext({ sampleRate });
  await audioCtx.resume().catch(() => {});
  const source = audioCtx.createMediaStreamSource(mediaStream);
  const processor = audioCtx.createScriptProcessor(4096, 1, 1);
  let done = false;

  // First yield preroll
  if (preroll.length) {
    const prerollResampled = downsampleTo16k(preroll, sampleRate);
    const prerollPcm = floatTo16BitPCM(prerollResampled);
    yield { AudioEvent: { AudioChunk: prerollPcm } };
    console.debug("Yielded preroll pcm");
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
    while (true) {
      if (done) break;
      if (queue.length) {
        yield { AudioEvent: { AudioChunk: queue.shift()! } };
        console.debug("yielded audio");
      } else {
        await new Promise((resolve) => setTimeout(resolve, 10));
      }
    }
  } finally {
    done = true;
    processor.disconnect();
    source.disconnect();
    audioCtx.close().catch(() => {});
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
    output[offsetResult] = accum / count;
    offsetResult++;
    offsetBuffer = nextOffsetBuffer;
  }
  return output;
}
