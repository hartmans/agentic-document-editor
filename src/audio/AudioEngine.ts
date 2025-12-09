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

  constructor(audioContext: AudioContext) {
    this.audioContext = audioContext;
  }

  playChunk(
    chunkId: number,
    buffer: AudioBuffer,
    onEnded?: () => void
  ): void {
    this.stop();
    const source = this.audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(this.audioContext.destination);
    source.onended = () => {
      this.sourceNode = null;
      this.playingChunkId = null;
      this.chunkStartTime = null;
      if (onEnded) onEnded();
    };
    source.start();
    this.sourceNode = source;
    this.playingChunkId = chunkId;
    this.chunkStartTime = this.audioContext.currentTime;
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
