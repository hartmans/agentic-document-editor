"""
Agentic document editor foundation: chunk text, synthesize speech with Amazon Polly,
track timing via speech marks, and expose a FastAPI server for playback and
interruption-to-text mapping.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

from agentic_editor.document import DocumentState, load_document
from agentic_editor.tts import build_document_state, map_elapsed_time_to_document
from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
import uvicorn


# ----------------------------- FastAPI models ------------------------------ #


class DocumentChunkModel(BaseModel):
    chunk_id: int
    doc_char_start: int
    doc_char_end: int


class DocumentResponseModel(BaseModel):
    document: str
    chunks: List[DocumentChunkModel]
    voice_id: str
    chunking_strategy: str


class InterruptionRequest(BaseModel):
    chunk_id: int
    elapsed_ms: int = Field(..., ge=0)


class ResumePoint(BaseModel):
    chunkId: int
    offsetMs: int
    docCharPosition: int


class StsTokenResponse(BaseModel):
    access_key_id: str
    secret_access_key: str
    session_token: str
    expiration: str


# ----------------------------- FastAPI app --------------------------------- #


def create_app(state: DocumentState) -> FastAPI:
    app = FastAPI(title="Agentic Document TTS Server")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/status")
    def status():
        return {
            "document_length": len(state.document),
            "chunk_count": len(state.chunks),
            "voice_id": state.voice_id,
            "chunking_strategy": "paragraph",
        }

    @app.get("/document", response_model=DocumentResponseModel)
    def get_document():
        return DocumentResponseModel(
            document=state.document,
            chunks=[
                DocumentChunkModel(
                    chunk_id=i,
                    doc_char_start=c.doc_char_start,
                    doc_char_end=c.doc_char_end,
                )
                for i, c in enumerate(state.chunks)
            ],
            voice_id=state.voice_id,
            chunking_strategy="paragraph",
        )

    @app.get("/chunks/{chunk_id}/audio")
    def get_chunk_audio(chunk_id: int):
        try:
            chunk = state.chunks[chunk_id]
        except IndexError:
            raise HTTPException(status_code=404, detail="Unknown chunk_id") from None
        return Response(content=chunk.audio_bytes, media_type=chunk.audio_mime_type)

    @app.post("/interruptions", response_model=ResumePoint)
    def map_interruption(request: InterruptionRequest):
        try:
            chunk = state.chunks[request.chunk_id]
        except IndexError:
            raise HTTPException(status_code=404, detail="Unknown chunk_id") from None
        mapping = map_elapsed_time_to_document(chunk, request.chunk_id, request.elapsed_ms)
        return ResumePoint(**mapping)

    @app.get("/stt_token", response_model=StsTokenResponse)
    def get_stt_token():
        if state.aws_session is None or state.region_name is None:
            raise HTTPException(status_code=500, detail="AWS session not configured")
        client = state.aws_session.client("sts", region_name=state.region_name)
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "transcribe:StartStreamTranscription",
                        "transcribe:StartStreamTranscriptionWebSocket",
                        "transcribe:StartTranscriptionJob",
                        "transcribe:GetTranscriptionJob",
                    ],
                    "Resource": "*",
                }
            ],
        }
        try:
            resp = client.get_federation_token(
                Name="agentic-editor-stt",
                Policy=json.dumps(policy),
                DurationSeconds=900,
            )
        except ClientError as exc:
            logging.error("Failed to obtain STS token: %s", exc)
            raise HTTPException(status_code=500, detail="Failed to obtain STS token")
        creds = resp.get("Credentials")
        if not creds:
            raise HTTPException(status_code=500, detail="Invalid STS response")
        return StsTokenResponse(
            access_key_id=creds["AccessKeyId"],
            secret_access_key=creds["SecretAccessKey"],
            session_token=creds["SessionToken"],
            expiration=creds["Expiration"].isoformat() if hasattr(creds["Expiration"], "isoformat") else str(creds["Expiration"]),
        )
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic document editor TTS synchronization server."
    )
    parser.add_argument(
        "--document-path",
        required=True,
        help="Path to the UTF-8 plain-text document to read.",
    )
    parser.add_argument(
        "--voice-id",
        default="Joanna",
        help="Amazon Polly VoiceId to use (default: Joanna).",
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region to use for Polly (default: us-east-1).",
    )
    parser.add_argument(
        "--cache-dir",
        default=".tts_cache",
        help="Directory to store/retrieve cached TTS assets (default: .tts_cache).",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for uvicorn to bind (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port for uvicorn (default: 8000)."
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    document_path = Path(args.document_path)

    try:
        document_text = load_document(document_path)
    except Exception as exc:
        logging.error("Failed to load document: %s", exc)
        raise SystemExit(1) from exc

    try:
        state = build_document_state(
            document_text=document_text,
            voice_id=args.voice_id,
            region_name=args.region,
            cache_dir=Path(args.cache_dir),
        )
    except Exception as exc:
        logging.error("Failed to preprocess document: %s", exc)
        raise SystemExit(1) from exc

    app = create_app(state)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
