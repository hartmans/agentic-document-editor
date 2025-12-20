import { createContext, useMemo, useState, ReactNode } from "react";

interface AudioContextState {
  audioContext: AudioContext | null;
  setAudioContext: (ctx: AudioContext) => void;
  micStream: MediaStream | null;
  setMicStream: (stream: MediaStream | null) => void;
}

export const AudioCtx = createContext<AudioContextState>({
  audioContext: null,
  setAudioContext: () => {},
  micStream: null,
  setMicStream: () => {},
});

export function AudioContextProvider({ children }: { children: ReactNode }) {
  const [audioContext, setAudioContextState] = useState<AudioContext | null>(null);
  const [micStream, setMicStreamState] = useState<MediaStream | null>(null);

  const value = useMemo(
    () => ({
      audioContext,
      setAudioContext: (ctx: AudioContext) => setAudioContextState(ctx),
      micStream,
      setMicStream: (stream: MediaStream | null) => setMicStreamState(stream),
    }),
    [audioContext, micStream]
  );

  return <AudioCtx.Provider value={value}>{children}</AudioCtx.Provider>;
}
