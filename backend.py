"""
Agentic document editor foundation: chunk text, synthesize speech with Amazon Polly or qwen_tts,
track timing via speech marks, and expose a FastAPI server for playback and
interruption-to-text mapping.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Literal
import typing
from uuid import uuid4

from agentic_editor.document import (
    DocumentState,
    CharacterResumePoint,
    ResumePoint,
    apply_document_update,
    load_document,
)
from agentic_editor.graph import GraphState, build_graph, initial_messages, instruction_message
from agentic_editor.tts import (
    build_document_state,
    submit_document_audio,
    map_elapsed_time_to_document,
)
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
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


class WordMarkModel(BaseModel):
    timeMs: int
    docCharStart: int
    docCharEnd: int


class ChunkMarksResponse(BaseModel):
    documentId: str
    chunkId: int
    marks: List[WordMarkModel]


class InterruptionRequest(BaseModel):
    documentId: str
    chunk_id: int
    elapsed_ms: int = Field(..., ge=0)


class InstructionsRequest(BaseModel):
    baseDocumentId: str
    instructions: str
    resumePoint: CharacterResumePoint


class InstructionsResponse(BaseModel):
    resumePoint: CharacterResumePoint
    documentId: str
    document: DocumentResponseModel
    responseDocumentId: str
    responseDocument: DocumentResponseModel


class AwsStsTokenResponse(BaseModel):
    access_key_id: str
    secret_access_key: str
    session_token: str
    expiration: str


class AwsAsrResponse(BaseModel):
    type: Literal["aws"]
    region: str
    stsToken: AwsStsTokenResponse


class Gemma4AsrResponse(BaseModel):
    type: Literal["gemma4"]


AsrResponse = AwsAsrResponse | Gemma4AsrResponse


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


def _resolve_asr_type(config: dict[str, typing.Any]) -> Literal["aws", "gemma4"]:
    region_name = config.get("region_name")
    if region_name:
        return "aws"
    return "gemma4"


def _build_aws_asr_response(
    state: DocumentState,
    region_name: str,
) -> AwsAsrResponse:
    session = state.aws_session
    if session is None:
        import boto3

        session = boto3.Session()
    client = session.client("sts", region_name=region_name)
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
    resp = client.get_federation_token(
        Name="agentic-editor-stt",
        Policy=json.dumps(policy),
        DurationSeconds=900,
    )
    creds = resp.get("Credentials")
    if not creds:
        raise HTTPException(status_code=500, detail="Invalid STS response")
    return AwsAsrResponse(
        type="aws",
        region=region_name,
        stsToken=AwsStsTokenResponse(
            access_key_id=creds["AccessKeyId"],
            secret_access_key=creds["SecretAccessKey"],
            session_token=creds["SessionToken"],
            expiration=creds["Expiration"].isoformat()
            if hasattr(creds["Expiration"], "isoformat")
            else str(creds["Expiration"]),
        ),
    )


# ----------------------------- FastAPI app --------------------------------- #


def create_app(
    state: DocumentState,
    document_path: Path,
    *,
    config: dict[str, typing.Any],
    asr_receiver: object | None = None,
    gemma4_debug_audio_dir: Path | None = None,
) -> FastAPI:
    document_id = uuid4().hex
    app = FastAPI(title="Agentic Document TTS Server")
    app.state.document_state = state
    app.state.document_id = document_id
    app.state.document_path = document_path
    app.state.documents  = {document_id: state}
    app.state.instructions_lock = asyncio.Lock()
    app.state.graph = build_graph(config)
    app.state.messages = initial_messages(state.document)
    app.state.asr_type = _resolve_asr_type(config)
    app.state.asr_region_name = str(config["region_name"]) if config.get("region_name") else None
    app.state.asr_receiver = asr_receiver
    app.state.asr_config = dict(config)
    app.state.gemma4_debug_audio_dir = gemma4_debug_audio_dir

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/status")
    async def status():
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

    @app.get("/asr", response_model=AsrResponse)
    def get_asr():
        if app.state.asr_type == "aws":
            region_name = app.state.asr_region_name
            if not region_name:
                raise HTTPException(status_code=500, detail="AWS ASR region not configured")
            try:
                return _build_aws_asr_response(app.state.document_state, region_name)
            except HTTPException:
                raise
            except Exception as exc:
                logging.error("Failed to obtain AWS ASR config: %s", exc)
                raise HTTPException(status_code=500, detail="Failed to obtain AWS ASR config")
        return Gemma4AsrResponse(type="gemma4")

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

    @app.get("/chunks/{document_id}/{chunk_id}/marks", response_model=ChunkMarksResponse)
    def get_chunk_marks(document_id: str, chunk_id: int):
        document_state = app.state.documents.get(document_id)
        if document_state is None:
            raise HTTPException(status_code=404, detail="Unknown document_id")
        if chunk_id < 0 or chunk_id >= len(document_state.chunks):
            raise HTTPException(status_code=404, detail="Unknown chunk_id")
        chunk = document_state.chunks[chunk_id]
        try:
            _, marks = document_state.tts_cache[chunk.text]
        except Exception as exc:
            logging.error("Failed to load marks for chunk %s: %s", chunk_id, exc)
            raise HTTPException(status_code=500, detail="Failed to synthesize audio")
        return ChunkMarksResponse(
            documentId=document_id,
            chunkId=chunk_id,
            marks=[WordMarkModel(
                timeMs=mark.time_ms,
                docCharStart=chunk.doc_char_start + mark.chunk_char_start,
                docCharEnd=chunk.doc_char_start + mark.chunk_char_end,
            ) for mark in marks],
        )

    @app.post("/interruptions", response_model=ResumePoint)
    def map_interruption(request: InterruptionRequest):
        document_state = app.state.documents.get(request.documentId)
        if document_state is None:
            raise HTTPException(status_code=404, detail="Unknown document revision")
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
        return ResumePoint(**mapping, documentId=request.documentId)

    @app.post("/instructions", response_model=InstructionsResponse)
    async def handle_instructions(request: InstructionsRequest):
        async with app.state.instructions_lock:
            if request.baseDocumentId != app.state.document_id:
                raise HTTPException(status_code=409, detail="Document changed; reload before submitting instructions")
            if request.resumePoint.documentId not in (None, request.baseDocumentId):
                raise HTTPException(status_code=409, detail="Resume point belongs to another document revision")
            return await process_instructions(request)

    async def process_instructions(request: InstructionsRequest):
        document_state = app.state.document_state
        resume_point = request.resumePoint
        messages = [
            *app.state.messages,
            instruction_message(
                document_state.document,
                resume_point.docCharPosition,
                request.instructions,
            ),
        ]
        graph_state = GraphState(
            document_state=document_state,
            position=resume_point.docCharPosition,
            chunk_id=document_state.find_chunk_id(resume_point.docCharPosition),
            messages=messages,
        )
        result = await app.state.graph.ainvoke(graph_state)
        if isinstance(result, GraphState):
            output_state = result
        else:
            output_state = GraphState(**result)
        if output_state.response_text is None:
            raise RuntimeError('No AI response')
        updated_state = output_state.document_state
        if updated_state is None:
            raise RuntimeError("No document state returned")
        updated_document_id = (
            uuid4().hex if updated_state.document != document_state.document
            else app.state.document_id
        )
        response_text = output_state.response_text
        response_document_id = "response-" + _sha256_text(response_text)
        response_document_state = build_document_state(
            document_text=response_text,
            submit_audio=False,
            tts_cache=document_state.tts_cache,
            session=document_state.aws_session,
        )
        submit_document_audio(
            updated_state,
            position_hint=(output_state.position if output_state.position is not None
                           else resume_point.docCharPosition),
            response_state=response_document_state,
        )
        position = (output_state.position if output_state.position is not None
                    else resume_point.docCharPosition)
        updated_resume_point = CharacterResumePoint(
            docCharPosition=max(0, min(position, len(updated_state.document))),
            documentId=updated_document_id,
        )
        await run_in_threadpool(
            app.state.document_path.write_text, updated_state.document, encoding="utf-8"
        )
        # Publish together without an await. Old revisions remain addressable by
        # outstanding audio and interruption requests.
        app.state.documents[updated_document_id] = updated_state
        app.state.documents[response_document_id] = response_document_state
        app.state.document_state = updated_state
        app.state.document_id = updated_document_id
        app.state.messages = output_state.messages
        return InstructionsResponse(
            resumePoint=updated_resume_point,
            documentId=updated_document_id,
            document=_build_document_response(updated_state),
            responseDocumentId=response_document_id,
            responseDocument=_build_document_response(response_document_state),
        )

    if app.state.asr_type == "gemma4":
        @app.websocket("/ws/asr")
        async def asr_websocket(websocket: WebSocket):
            await websocket.accept()
            buffer = None
            try:
                if app.state.asr_receiver is None:
                    from agentic_editor.gemma4 import Gemma4AsrReceiver

                    app.state.asr_receiver = Gemma4AsrReceiver(
                        app.state.asr_config,
                        document=app.state.document_state.document,
                        debug_audio_dir=app.state.gemma4_debug_audio_dir,
                    )
                receiver = app.state.asr_receiver
                while True:
                    message = await websocket.receive()
                    message_type = message["type"]

                    if message_type == "websocket.disconnect":
                        return

                    if message.get("text") is not None:
                        payload = json.loads(message["text"])
                        event_type = payload.get("type")

                        if event_type == "start":
                            buffer = receiver.create_buffer()
                            await websocket.send_json({"type": "started"})
                        elif event_type == "end":
                            if buffer is None:
                                raise ValueError("Received end before start")
                            instructions = await run_in_threadpool(
                                receiver.transcribe_audio,
                                buffer.to_float32(),
                            )
                            await websocket.send_json(
                                {"type": "result", "instructions": instructions}
                            )
                            await websocket.close()
                            return
                        elif event_type == "cancel":
                            await websocket.close(code=1000)
                            return
                        else:
                            raise ValueError(f"Unknown ASR websocket event type: {event_type!r}")
                    elif message.get("bytes") is not None:
                        if buffer is None:
                            raise ValueError("Received binary audio before start")
                        buffer.append_bytes(typing.cast(bytes, message["bytes"]))
            except Exception as exc:
                logging.exception("ASR websocket failed")
                await websocket.send_json({"type": "error", "detail": str(exc)})
                await websocket.close(code=1011)
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
        default="Ruth",
        help="Amazon Polly VoiceId to use (default: Joanna).",
    )
    parser.add_argument(
        "--region",
        help="AWS region to use when configured.",
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
    parser.add_argument(
        "--debug-gemma4",
        action="store_true",
        help="Write each Gemma4 ASR audio fragment to logs/*.wav for debugging.",
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
    if args.region is not None:
        config["region_name"] = args.region
    if "cache_dir" not in config:
        config["cache_dir"] = args.cache_dir
    if "speech_driver" not in config:
        config["speech_driver"] = "poly"

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

    app = create_app(
        state,
        document_path,
        config=config,
        gemma4_debug_audio_dir=(Path("logs") if args.debug_gemma4 else None),
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
