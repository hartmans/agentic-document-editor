const rawBase = import.meta.env.VITE_API_BASE;
const API_BASE =
  rawBase && rawBase.trim()
    ? rawBase.trim().replace(/\/+$/, "")
    : "/api";

function join(base: string, path: string): string {
  return `${base.replace(/\/+$/, "")}/${path.replace(/^\/+/, "")}`;
}

export interface StatusResponse {
  documentId: string;
  documentLength: number;
  chunkCount: number;
  voiceId: string;
  chunkingStrategy: string;
  document: DocumentResponse;
}

export interface DocumentChunk {
  chunk_id: number;
  doc_char_start: number;
  doc_char_end: number;
  has_audio: boolean;
}

export interface DocumentResponse {
  document: string;
  chunks: DocumentChunk[];
  voice_id: string;
  chunking_strategy: string;
}

// Character positions are Unicode code-point offsets, matching Python indexing.
export interface ResumePoint {
  documentId: string;
  docCharPosition: number;
  // Retain an exact audio position when alignment has not arrived yet.
  chunkId?: number;
  offsetMs?: number;
}

export interface WordMark {
  timeMs: number;
  docCharStart: number;
  docCharEnd: number;
}

export interface ChunkMarks {
  documentId: string;
  chunkId: number;
  marks: WordMark[];
}

export interface SttToken {
  access_key_id: string;
  secret_access_key: string;
  session_token: string;
  expiration: string;
}

export type AsrConfigResponse =
  | {
      type: "aws";
      region: string;
      stsToken: SttToken;
    }
  | {
      type: "gemma4";
    };

export interface InstructionsRequest {
  baseDocumentId: string;
  instructions: string;
  resumePoint: ResumePoint;
}

export interface InstructionsResponse {
  disposition: "play" | "pause" | "stop";
  resumePoint: ResumePoint;
  documentId: string;
  document: DocumentResponse;
  responseDocumentId: string;
  responseDocument: DocumentResponse;
}

export class ApiError extends Error {
  constructor(public readonly status: number, message: string) {
    super(message);
  }
}

async function handleJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    throw new ApiError(res.status, `Request failed (${res.status}): ${text}`);
  }
  return res.json() as Promise<T>;
}

export async function fetchStatus(): Promise<StatusResponse> {
  return handleJson(await fetch(join(API_BASE, "/status")));
}

export async function fetchDocument(documentId: string): Promise<DocumentResponse> {
  return handleJson(await fetch(join(API_BASE, `/document/${documentId}`)));
}

export async function fetchChunkAudio(
  documentId: string,
  chunkId: number
): Promise<ArrayBuffer> {
  const res = await fetch(join(API_BASE, `/chunks/${documentId}/${chunkId}/audio`));
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Audio fetch failed (${res.status}): ${text}`);
  }
  return res.arrayBuffer();
}

export async function fetchChunkMarks(
  documentId: string,
  chunkId: number,
): Promise<ChunkMarks> {
  return handleJson(await fetch(join(API_BASE, `/chunks/${documentId}/${chunkId}/marks`)));
}

export async function fetchAsrConfig(): Promise<AsrConfigResponse> {
  return handleJson(await fetch(join(API_BASE, "/asr")));
}

export async function postInstructions(
  payload: InstructionsRequest
): Promise<InstructionsResponse> {
  const res = await fetch(join(API_BASE, "/instructions"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handleJson(res);
}
