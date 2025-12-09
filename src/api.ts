const rawBase = import.meta.env.VITE_API_BASE;
const API_BASE =
  rawBase && rawBase.trim()
    ? rawBase.trim().replace(/\/+$/, "")
    : "http://localhost:8000";

function join(base: string, path: string): string {
  return `${base.replace(/\/+$/, "")}/${path.replace(/^\/+/, "")}`;
}

export interface StatusResponse {
  document_length?: number;
  document_length_chars?: number;
  chunk_count?: number;
  num_chunks?: number;
  voice_id: string;
  chunking_strategy: string;
}

export interface ChunkMetadata {
  chunk_id: number;
  doc_char_start: number;
  doc_char_end: number;
  preview: string;
}

export interface SpeechMark {
  time_ms: number;
  mark_type: string;
  text: string;
  chunk_char_start: number;
  chunk_char_end: number;
}

export interface ChunkDetail {
  chunk_id: number;
  text: string;
  doc_char_start: number;
  doc_char_end: number;
  speech_marks: SpeechMark[];
}

export interface InterruptionRequest {
  chunk_id: number;
  elapsed_ms: number;
}

export interface InterruptionResponse {
  chunk_id: number;
  elapsed_ms: number;
  word_text: string;
  chunk_char_start: number;
  chunk_char_end: number;
  doc_char_start: number;
  doc_char_end: number;
}

async function handleJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Request failed (${res.status}): ${text}`);
  }
  return res.json() as Promise<T>;
}

export async function fetchStatus(): Promise<StatusResponse> {
  return handleJson(await fetch(join(API_BASE, "/status")));
}

export async function fetchChunks(): Promise<ChunkMetadata[]> {
  return handleJson(await fetch(join(API_BASE, "/chunks")));
}

export async function fetchChunkDetail(chunkId: number): Promise<ChunkDetail> {
  return handleJson(await fetch(join(API_BASE, `/chunks/${chunkId}`)));
}

export async function fetchChunkAudio(chunkId: number): Promise<ArrayBuffer> {
  const res = await fetch(join(API_BASE, `/chunks/${chunkId}/audio`));
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Audio fetch failed (${res.status}): ${text}`);
  }
  return res.arrayBuffer();
}

export async function postInterruption(
  payload: InterruptionRequest
): Promise<InterruptionResponse> {
  const res = await fetch(join(API_BASE, "/interruptions"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handleJson(res);
}
