import { useEffect, useRef, useState } from "react";
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
  const [running, setRunning] = useState(false);
  const stopTimerRef = useRef<NodeJS.Timeout | null>(null);
  const silenceTimerRef = useRef<NodeJS.Timeout | null>(null);
  const lastTranscriptRef = useRef<number|null>((null));

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
        const command = new StartStreamTranscriptionCommand({
          LanguageCode: "en-US",
          MediaEncoding: "pcm",
          MediaSampleRateHertz: TARGET_RATE,
          AudioStream: audioIterator,
        });

        const response = await client.send(command);
        console.log('STT Started');
        for await (const evt of response.TranscriptResultStream ?? []) {
          const results = evt?.TranscriptEvent?.Transcript?.Results;
          if (!results) continue;
          for (const res of results) {
            console.log('Got STT result');
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
          if (cancelled || !running) break;
        }
      } catch (err) {
        console.error("STT failed", err);
      } finally {
        cleanupTimers();
        setRunning(false);
        onStop();
      }
    })();

    startStopTimers();

    return () => {
      cancelled = true;
      cleanupTimers();
      setRunning(false);
      onStop();
    };
  }, []);

  function startStopTimers() {
    cleanupTimers();
    stopTimerRef.current = setTimeout(() => endIfIdle(true), 10000);
    silenceTimerRef.current = setInterval(() => endIfIdle(false), 500);
  }

  function endIfIdle(isHardLimit: boolean) {
    const idleMs = Date.now() - (lastTranscriptRef.current||started);
    if (isHardLimit && idleMs >= 10000) {
      cleanupTimers();
      setRunning(false);
      onStop();
      return;
    }
    if (!lastTranscriptRef.current)
      return;
    if (!isHardLimit && idleMs >= 4000 && getRms() < vadThreshold) {
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
  const source = audioCtx.createMediaStreamSource(mediaStream);
  const processor = audioCtx.createScriptProcessor(4096, 1, 1);

  // First yield preroll
  if (preroll.length) {
    const prerollResampled = downsampleTo16k(preroll, sampleRate);
    const prerollPcm = floatTo16BitPCM(prerollResampled);
    yield { AudioEvent: { AudioChunk: prerollPcm } };
    console.debug('Yielded preroll pcm');
  }

  source.connect(processor);
  processor.connect(audioCtx.destination);

  const queue: Uint8Array[] = [];
  processor.onaudioprocess = (e) => {
    const data = e.inputBuffer.getChannelData(0);
    const resampled = downsampleTo16k(data, sampleRate);
    queue.push(floatTo16BitPCM(resampled));
  };

  while (true) {
    if (queue.length) {
      yield { AudioEvent: { AudioChunk: queue.shift()! } };
      console.debug('yielded audio')
    } else {
      await new Promise((resolve) => setTimeout(resolve, 10));
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
    output[offsetResult] = accum / count;
    offsetResult++;
    offsetBuffer = nextOffsetBuffer;
  }
  return output;
}
