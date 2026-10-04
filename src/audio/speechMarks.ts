import type { DocumentChunk, ResumePoint, WordMark } from "../api";

export function chunkForPosition(
  chunks: Pick<DocumentChunk, "doc_char_end">[], position: number,
): number {
  const index = chunks.findIndex((chunk) => position < chunk.doc_char_end);
  return index >= 0 ? index : Math.max(0, chunks.length - 1);
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


// Audio coordinates are local playback state, never instruction API fields.
export interface PlaybackPosition extends ResumePoint {
  chunkId?: number;
  offsetMs?: number;
}

export async function resolvePlaybackPosition(
  point: PlaybackPosition,
  chunks: Pick<DocumentChunk, "doc_char_start">[],
  loadMarks: (chunkId: number) => Promise<WordMark[]>,
): Promise<ResumePoint> {
  if (point.chunkId === undefined || point.offsetMs === undefined) {
    return { documentId: point.documentId, docCharPosition: point.docCharPosition };
  }
  const marks = await loadMarks(point.chunkId);
  return {
    documentId: point.documentId,
    docCharPosition: markAtTime(marks, point.offsetMs)?.docCharStart
      ?? chunks[point.chunkId].doc_char_start,
  };
}
