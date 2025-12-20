const rawBase = import.meta.env.VITE_API_BASE;
const API_BASE =
  rawBase && rawBase.trim()
    ? rawBase.trim().replace(/\/+$/, "")
    : "http://localhost:8000";

function join(base: string, path: string): string {
  return `${base.replace(/\/+$/, "")}/${path.replace(/^\/+/, "")}`;
}

export interface StatusResponse {
  document_length: number;
  chunk_count: number;
  voice_id: string;
  chunking_strategy: string;
}

export interface DocumentChunk {
  chunk_id: number;
  doc_char_start: number;
  doc_char_end: number;
}

export interface DocumentResponse {
  document: string;
  chunks: DocumentChunk[];
  voice_id: string;
  chunking_strategy: string;
}

export interface ResumePoint {
  chunkId: number | null;
  offsetMs: number;
  docCharPosition?: number | null;
}

export interface InterruptionRequest {
  chunk_id: number;
  elapsed_ms: number;
}

export interface SttToken {
  access_key_id: string;
  secret_access_key: string;
  session_token: string;
  expiration: string;
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

export async function fetchDocument(): Promise<DocumentResponse> {
  return handleJson(await fetch(join(API_BASE, "/document")));
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
): Promise<ResumePoint> {
  const res = await fetch(join(API_BASE, "/interruptions"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handleJson(res);
}

export async function fetchSttToken(): Promise<SttToken> {
  return handleJson(await fetch(join(API_BASE, "/stt_token")));
}
