import type { WordMark } from "../api";

export interface MarksResult {
  marks: WordMark[];
  error: Error | null;
}

// Only a character-based seek inside a chunk requires alignment before play.
export async function marksForStart(
  position: number,
  chunkStart: number,
  knownOffsetMs: number | undefined,
  pending: Promise<MarksResult>,
  cached: WordMark[],
): Promise<WordMark[]> {
  if (knownOffsetMs !== undefined || position <= chunkStart) return cached;
  const result = await pending;
  if (result.error) throw result.error;
  return result.marks;
}
