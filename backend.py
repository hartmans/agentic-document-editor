"""
Agentic document editor foundation: chunk text, synthesize speech with Amazon Polly,
track timing via speech marks, and expose a FastAPI server for playback and
interruption-to-text mapping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List
import typing

from agentic_editor.document import (
    DocumentState,
    ResumePoint,
    apply_document_update,
    load_document,
)
from agentic_editor.graph import GraphState, build_graph
from agentic_editor.tts import (
    build_document_state,
    map_doc_char_to_offset,
    map_elapsed_time_to_document,
)
from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    messages_from_dict,
    messages_to_dict,
)
from pydantic import BaseModel, Field
import uvicorn
import yaml


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


class InstructionsRequest(BaseModel):
    messages: List[dict] = Field(default_factory=list)
    instructions: str
    resumePoint: ResumePoint


class InstructionsResponse(BaseModel):
    messages: List[dict]
    resumePoint: ResumePoint
    documentId: str


class StsTokenResponse(BaseModel):
    access_key_id: str
    secret_access_key: str
    session_token: str
    expiration: str


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_document_response(document_state: DocumentState) -> DocumentResponseModel:
    return DocumentResponseModel(
        document=document_state.document,
        chunks=[
            DocumentChunkModel(
                chunk_id=i,
                doc_char_start=c.doc_char_start,
                doc_char_end=c.doc_char_end,
            )
            for i, c in enumerate(document_state.chunks)
        ],
        voice_id=document_state.voice_id,
        chunking_strategy="paragraph",
    )


# ----------------------------- FastAPI app --------------------------------- #


def create_app(state: DocumentState, document_path: Path) -> FastAPI:
    document_id = _sha256_text(str(document_path.resolve()))
    app = FastAPI(title="Agentic Document TTS Server")
    app.state.document_state = state
    app.state.document_id = document_id
    app.state.document_path = document_path
    app.state.documents  = {document_id: state}
    app.state.graph = build_graph()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/status")
    def status():
        document_state = app.state.document_state
        document_response = _build_document_response(document_state).dict()
        return {
            "documentId": app.state.document_id,
            "documentLength": len(document_state.document),
            "chunkCount": len(document_state.chunks),
            "voiceId": document_state.voice_id,
            "chunkingStrategy": "paragraph",
            "document": document_response,
        }

    @app.get("/document/{document_id}", response_model=DocumentResponseModel)
    def get_document(document_id: str):
        document_state = app.state.documents.get(document_id)
        if not document_state:
            raise HTTPException(status_code=404, detail="Unknown document_id")
        return _build_document_response(document_state)

    @app.get("/chunks/{document_id}/{chunk_id}/audio")
    def get_chunk_audio(document_id: str, chunk_id: int):
        document_state = app.state.documents.get(document_id)
        if not document_state:
            raise HTTPException(status_code=404, detail="Unknown document_id")
        try:
            chunk = document_state.chunks[chunk_id]
        except IndexError:
            raise HTTPException(status_code=404, detail="Unknown chunk_id") from None
        try:
            audio_bytes, _ = document_state.tts_cache[chunk.text]
        except Exception as exc:
            logging.error("Failed to load audio for chunk %s: %s", chunk_id, exc)
            raise HTTPException(status_code=500, detail="Failed to synthesize audio")
        return Response(content=audio_bytes, media_type=chunk.audio_mime_type)

    @app.post("/interruptions", response_model=ResumePoint)
    def map_interruption(request: InterruptionRequest):
        document_state = app.state.document_state
        try:
            chunk = document_state.chunks[request.chunk_id]
        except IndexError:
            raise HTTPException(status_code=404, detail="Unknown chunk_id") from None
        mapping = map_elapsed_time_to_document(
            document_state.tts_cache,
            chunk,
            request.chunk_id,
            request.elapsed_ms,
        )
        return ResumePoint(**mapping)

    @app.post("/instructions", response_model=InstructionsResponse)
    async def handle_instructions(request: InstructionsRequest):
        document_state = app.state.document_state
        resume_point = request.resumePoint
        messages: list[BaseMessage]
        raw_messages = list(request.messages)
        if raw_messages and isinstance(raw_messages[0], dict):
            messages = list(messages_from_dict(raw_messages))
        else:
            messages = typing.cast(list[BaseMessage], raw_messages)
        messages.append(HumanMessage(content=request.instructions))
        graph_state = GraphState(
            document_state=document_state,
            position=resume_point.docCharPosition,
            chunk_id=resume_point.chunkId,
            elapsed_ms=resume_point.offsetMs,
            messages=messages,
        )
        result = await app.state.graph.ainvoke(graph_state)
        if isinstance(result, GraphState):
            output_state = result
        else:
            output_state = GraphState(**result)
            if isinstance(output_state.messages[-1], AIMessage):
                response_message = output_state.messages[-1]
            else:
                raise RuntimeError('No AI response')
        app.state.document_state = output_state.document_state
        app.state.documents[app.state.document_id] = output_state.document_state
        response_text = response_message.content # pyright: ignore[reportPossiblyUnboundVariable]
        if not isinstance(response_text, str):
            response_text = json.dumps(response_text, ensure_ascii=True)
        response_document_id = _sha256_text(response_text)
        response_document_state = build_document_state(
            document_text=response_text,
            tts_cache=document_state.tts_cache,
            session=document_state.aws_session,
        )
        app.state.documents[response_document_id] = response_document_state
        if output_state.position:
            updated_resume_point = map_doc_char_to_offset(
                app.state.document_state, # type: ignore
                output_state.position 
            )
        else: updated_resume_point = resume_point
        if app.state.document_state:
            app.state.document_path.write_text(app.state.document_state.document)
        return InstructionsResponse(
            messages=messages_to_dict(output_state.messages),
            resumePoint=updated_resume_point,
            documentId=response_document_id,
        )

    @app.get("/stt_token", response_model=StsTokenResponse)
    def get_stt_token():
        document_state = app.state.document_state
        if document_state.aws_session is None or document_state.region_name is None:
            raise HTTPException(status_code=500, detail="AWS session not configured")
        client = document_state.aws_session.client(
            "sts",
            region_name=document_state.region_name,
        )
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
        "--config",
        help="Optional YAML config file for TTS settings.",
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
    config: dict[str, typing.Any] = {}
    if args.config:
        config_path = Path(args.config)
        try:
            config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logging.error("Failed to load config: %s", exc)
            raise SystemExit(1) from exc
        if config_data is None:
            config_data = {}
        if not isinstance(config_data, dict):
            logging.error("Config file must contain a mapping object at top level.")
            raise SystemExit(1)
        config = config_data

    if "voice_id" not in config:
        config["voice_id"] = args.voice_id
    if "region_name" not in config:
        config["region_name"] = args.region
    if "cache_dir" not in config:
        config["cache_dir"] = args.cache_dir

    try:
        document_text = load_document(document_path)
    except Exception as exc:
        logging.error("Failed to load document: %s", exc)
        raise SystemExit(1) from exc

    try:
        state = build_document_state(
            document_text=document_text,
            config=config,
        )
    except Exception as exc:
        logging.error("Failed to preprocess document: %s", exc)
        raise SystemExit(1) from exc

    app = create_app(state, document_path)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
