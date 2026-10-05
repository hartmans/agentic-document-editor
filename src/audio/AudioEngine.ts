export interface PlaybackInfo {
  playingChunkId: number | null;
  offsetSeconds: number;
}

interface PlaybackCallbacks {
  onEnded?: () => void;
  onError?: (error: Error, position: PlaybackInfo) => void;
  onBuffering?: (buffering: boolean) => void;
}

interface Session {
  audio: HTMLAudioElement;
  chunkId: number;
  started: boolean;
  cancelStart: () => void;
  detach: () => void;
}

/** Each source gets its own element so queued events cannot affect a successor. */
export class AudioEngine {
  private session: Session | null = null;

  playChunk(
    chunkId: number,
    url: string,
    startOffsetSec = 0,
    callbacks: PlaybackCallbacks = {},
  ): Promise<boolean> {
    this.stop();
    const audio = new Audio();
    audio.preload = "auto";
    const listeners: Array<[string, EventListener]> = [];
    const listen = (event: string, handler: () => void) => {
      listeners.push([event, handler]);
      audio.addEventListener(event, handler);
    };
    return new Promise<boolean>((resolve, reject) => {
      const session: Session = {
        audio, chunkId, started: false,
        cancelStart: () => resolve(false),
        detach: () => listeners.forEach(([event, handler]) => audio.removeEventListener(event, handler)),
      };
      this.session = session;
      const current = () => this.session === session;
      let requestedPlay = false;
      const fail = (error: Error) => {
        if (!current()) return;
        const started = session.started;
        audio.pause();
        const position = { playingChunkId: started ? chunkId : null, offsetSeconds: audio.currentTime };
        this.release(session);
        callbacks.onBuffering?.(false);
        if (started) callbacks.onError?.(error, position);
        else reject(error);
      };
      const finish = () => {
        if (!current()) return;
        if (audio.error) {
          fail(new Error(audio.error.message || "Audio stream failed"));
          return;
        }
        this.release(session);
        resolve(false);
        callbacks.onBuffering?.(false);
        callbacks.onEnded?.();
      };
      const contains = (ranges: TimeRanges, position: number) => {
        for (let i = 0; i < ranges.length; i += 1) {
          if (ranges.start(i) <= position && position < ranges.end(i)) return true;
        }
        return false;
      };
      const start = () => {
        if (!current() || requestedPlay) return;
        if (startOffsetSec > 0) {
          if (audio.readyState === 0) return;
          if (Number.isFinite(audio.duration) && startOffsetSec >= audio.duration) {
            finish();
            return;
          }
          // A progressive response may not support HTTP range seeking. Wait
          // until the requested position is buffered instead of playing at zero.
          if (!contains(audio.seekable, startOffsetSec) && !contains(audio.buffered, startOffsetSec)) return;
          audio.currentTime = startOffsetSec;
        }
        requestedPlay = true;
        void audio.play().catch((error: Error) => fail(error));
      };
      listen("playing", () => {
        if (!current()) return;
        session.started = true;
        callbacks.onBuffering?.(false);
        resolve(true);
      });
      listen("waiting", () => { if (current()) callbacks.onBuffering?.(true); });
      listen("error", () => fail(new Error(audio.error?.message || "Audio stream failed")));
      listen("ended", finish);
      for (const event of ["loadedmetadata", "progress", "canplay", "canplaythrough", "durationchange", "suspend"]) {
        listen(event, start);
      }
      callbacks.onBuffering?.(true);
      audio.src = url;
      audio.load();
      start();
    });
  }

  private release(session: Session): void {
    session.detach();
    session.audio.pause();
    session.audio.removeAttribute("src");
    session.audio.load(); // Abort the old download and release its media buffers.
    if (this.session === session) this.session = null;
  }

  stop(): PlaybackInfo {
    const session = this.session;
    if (!session) return { playingChunkId: null, offsetSeconds: 0 };
    // pause() updates the media position. Read it before resetting the source.
    session.audio.pause();
    const info = {
      playingChunkId: session.started ? session.chunkId : null,
      offsetSeconds: session.audio.currentTime,
    };
    session.cancelStart();
    this.release(session);
    return info;
  }

  getCurrentPlaybackInfo(): PlaybackInfo {
    const session = this.session;
    return {
      playingChunkId: session?.started ? session.chunkId : null,
      offsetSeconds: session?.audio.currentTime ?? 0,
    };
  }
}
