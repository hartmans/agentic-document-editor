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

export interface MessageDict {
  type: string;
  data: Record<string, unknown>;
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
  messages: MessageDict[];
  instructions: string;
  resumePoint: ResumePoint;
}

export interface InstructionsResponse {
  messages: MessageDict[];
  resumePoint: ResumePoint;
  documentId: string;
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
