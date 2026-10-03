import type { DocumentChunk, WordMark } from "../api";

export function chunkForPosition(
  chunks: Pick<DocumentChunk, "chunk_id" | "doc_char_end">[], position: number,
): number {
  const chunk = chunks.find((chunk) => position < chunk.doc_char_end);
  return chunk?.chunk_id ?? chunks[chunks.length - 1]?.chunk_id ?? 0;
}

export function markAtTime(marks: WordMark[], timeMs: number): WordMark | undefined {
  let result = marks[0];
  for (const mark of marks) {
    if (mark.timeMs > timeMs) break;
    result = mark;
  }
  return result;
}

export function timeAtPosition(marks: WordMark[], position: number): number {
  let time = 0;
  for (const mark of marks) {
    if (mark.docCharStart > position) break;
    time = mark.timeMs;
  }
  return time;
}
