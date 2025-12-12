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
  onPartial,
  onStop,
}: SpeechToTextProps) {
  const [running, setRunning] = useState(false);
  const stopTimerRef = useRef<NodeJS.Timeout | null>(null);
  const silenceTimerRef = useRef<NodeJS.Timeout | null>(null);
  const lastTranscriptRef = useRef<number>(Date.now());

  useEffect(() => {
    let cancelled = false;
    (async () => {
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

        const audioIterator = pcmAudioIterator(mediaStream, preroll, sampleRate);
        setRunning(true);
        const command = new StartStreamTranscriptionCommand({
          LanguageCode: "en-US",
          MediaEncoding: "pcm",
          MediaSampleRateHertz: sampleRate,
          AudioStream: audioIterator,
        });

        const response = await client.send(command);
        for await (const evt of response.TranscriptResultStream ?? []) {
          const results = evt?.TranscriptEvent?.Transcript?.Results;
          if (!results) continue;
          for (const res of results) {
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
    const idleMs = Date.now() - lastTranscriptRef.current;
    if (isHardLimit && idleMs >= 10000) {
      cleanupTimers();
      setRunning(false);
      onStop();
      return;
    }
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
    const prerollPcm = floatTo16BitPCM(preroll);
    yield { AudioEvent: { AudioChunk: prerollPcm } };
  }

  source.connect(processor);
  processor.connect(audioCtx.destination);

  const queue: Uint8Array[] = [];
  processor.onaudioprocess = (e) => {
    const data = e.inputBuffer.getChannelData(0);
    queue.push(floatTo16BitPCM(data));
  };

  while (true) {
    if (queue.length) {
      yield { AudioEvent: { AudioChunk: queue.shift()! } };
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
