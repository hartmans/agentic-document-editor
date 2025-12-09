export interface PlaybackInfo {
  playingChunkId: number | null;
  chunkStartTime: number | null;
  contextCurrentTime: number;
}

export class AudioEngine {
  private audioContext: AudioContext;
  private sourceNode: AudioBufferSourceNode | null = null;
  private playingChunkId: number | null = null;
  private chunkStartTime: number | null = null;
  private onEndedCallback: (() => void) | null = null;

  constructor(audioContext: AudioContext) {
    this.audioContext = audioContext;
  }

  playChunk(
    chunkId: number,
    buffer: AudioBuffer,
    startOffsetSec = 0,
    onEnded?: () => void
  ): void {
    this.stop();
    this.onEndedCallback = onEnded || null;
    const source = this.audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(this.audioContext.destination);
    source.onended = () => {
      this.sourceNode = null;
      this.playingChunkId = null;
      this.chunkStartTime = null;
      const cb = this.onEndedCallback;
      this.onEndedCallback = null;
      if (cb) cb();
    };
    source.start(0, startOffsetSec);
    this.sourceNode = source;
    this.playingChunkId = chunkId;
    // Track start time relative to the buffer timeline so elapsed calculation accounts for offsets.
    this.chunkStartTime = this.audioContext.currentTime - startOffsetSec;
  }

  stop(): void {
    if (this.sourceNode) {
      try {
        this.sourceNode.stop();
      } catch {
        // already stopped
      }
    }
    this.sourceNode = null;
    this.playingChunkId = null;
    this.chunkStartTime = null;
  }

  getCurrentPlaybackInfo(): PlaybackInfo {
    return {
      playingChunkId: this.playingChunkId,
      chunkStartTime: this.chunkStartTime,
      contextCurrentTime: this.audioContext.currentTime,
    };
  }
}
