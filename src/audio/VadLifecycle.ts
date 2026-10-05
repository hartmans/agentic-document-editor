import type { MicVAD } from "@ricky0123/vad-web";

/** MicVAD changes flags before awaiting stream operations: transitions must not overlap. */
export class VadLifecycle {
  private pending: Promise<void> = Promise.resolve();

  constructor(private readonly vad: MicVAD) {}

  private enqueue(operation: () => Promise<void>): Promise<void> {
    const result = this.pending.then(operation);
    // Keep cleanup and later transitions possible after an operation fails.
    this.pending = result.catch(() => {});
    return result;
  }

  setEnabled(enabled: boolean): Promise<void> {
    return this.enqueue(() => enabled ? this.vad.start() : this.vad.pause());
  }

  destroy(): Promise<void> {
    return this.enqueue(() => this.vad.destroy());
  }
}
