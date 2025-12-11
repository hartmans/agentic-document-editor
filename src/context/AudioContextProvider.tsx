import { createContext, useMemo, useState, ReactNode } from "react";

interface AudioContextState {
  audioContext: AudioContext | null;
  setAudioContext: (ctx: AudioContext) => void;
}

export const AudioCtx = createContext<AudioContextState>({
  audioContext: null,
  setAudioContext: () => {},
});

export function AudioContextProvider({ children }: { children: ReactNode }) {
  const [audioContext, setAudioContextState] = useState<AudioContext | null>(null);

  const value = useMemo(
    () => ({
      audioContext,
      setAudioContext: (ctx: AudioContext) => setAudioContextState(ctx),
    }),
    [audioContext]
  );

  return <AudioCtx.Provider value={value}>{children}</AudioCtx.Provider>;
}
