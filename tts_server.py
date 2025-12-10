"""
Agentic document editor foundation: chunk text, synthesize speech with Amazon Polly,
track timing via speech marks, and expose a FastAPI server for playback and
interruption-to-text mapping.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

import boto3
from agentic_editor.tts_cache import TtsCache
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
import uvicorn


# ----------------------------- Data models --------------------------------- #


@dataclass
class SpeechMark:
    """Word- or sentence-level timing emitted by Polly."""

    time_ms: int
    mark_type: str
    text: str
    chunk_char_start: int
    chunk_char_end: int


@dataclass
class TextChunk:
    """Single contiguous span of the original document."""

    chunk_id: int
    text: str
    doc_char_start: int
    doc_char_end: int
    audio_bytes: bytes
    speech_marks: List[SpeechMark]
    audio_mime_type: str = "audio/mpeg"


@dataclass
class DocumentState:
    """In-memory representation of the preprocessed document."""

    original_text: str
    voice_id: str
    chunking_strategy: str
    chunks: List[TextChunk] = field(default_factory=list)


# ----------------------------- Chunking ------------------------------------ #


def chunk_document_paragraphs(document: str) -> List[tuple[int, int]]:
    """
    Split the document into contiguous spans separated by one or more blank lines.

    Returns a list of (start, end) offsets into the original document.
    """
    import re

    separator = re.compile(r"(?:\r?\n\s*){2,}")
    spans: List[tuple[int, int]] = []
    cursor = 0

    for match in separator.finditer(document):
        start = cursor
        end = match.start()
        cursor = match.end()
        if start == end:
            continue
        text = document[start:end]
        if text.strip() == "":
            continue
        spans.append((start, end))

    if cursor < len(document):
        trailing = document[cursor:]
        if trailing.strip() != "":
            spans.append((cursor, len(document)))

    # If the document was entirely whitespace or separators, fall back to one chunk.
    if not spans and document.strip() != "":
        spans.append((0, len(document)))
    return spans




# ----------------------------- Mapping logic ------------------------------- #


def _speech_marks_from_records(records: Sequence[dict]) -> List[SpeechMark]:
    marks: List[SpeechMark] = []
    for rec in records:
        try:
            marks.append(
                SpeechMark(
                    time_ms=int(rec["time_ms"]),
                    mark_type=str(rec["mark_type"]),
                    text=str(rec.get("text", "")),
                    chunk_char_start=int(rec["chunk_char_start"]),
                    chunk_char_end=int(rec["chunk_char_end"]),
                )
            )
        except (KeyError, ValueError, TypeError):
            continue
    marks.sort(key=lambda m: m.time_ms)
    return marks


def map_elapsed_time_to_document(chunk: TextChunk, elapsed_ms: int) -> dict:
    """Resolve an elapsed time within a chunk to document-level character offsets."""
    word_marks = [m for m in chunk.speech_marks if m.mark_type == "word"]
    if not word_marks:
        raise ValueError("Chunk has no word-level speech marks to map time.")
    # Ensure marks are time-sorted.
    word_marks.sort(key=lambda m: m.time_ms)

    target_mark = word_marks[0]
    for mark in word_marks:
        if mark.time_ms <= elapsed_ms:
            target_mark = mark
        else:
            break

    # Clamp to last word if elapsed exceeds final mark.
    if elapsed_ms > word_marks[-1].time_ms:
        target_mark = word_marks[-1]

    return {
        "chunk_id": chunk.chunk_id,
        "elapsed_ms": elapsed_ms,
        "doc_char_position": chunk.doc_char_start + target_mark.chunk_char_start,
    }


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


class InterruptionResponse(BaseModel):
    chunk_id: int
    elapsed_ms: int
    doc_char_position: int


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
            "document_length": len(state.original_text),
            "chunk_count": len(state.chunks),
            "voice_id": state.voice_id,
            "chunking_strategy": state.chunking_strategy,
        }

    @app.get("/document", response_model=DocumentResponseModel)
    def get_document():
        return DocumentResponseModel(
            document=state.original_text,
            chunks=[
                DocumentChunkModel(
                    chunk_id=c.chunk_id,
                    doc_char_start=c.doc_char_start,
                    doc_char_end=c.doc_char_end,
                )
                for c in state.chunks
            ],
            voice_id=state.voice_id,
            chunking_strategy=state.chunking_strategy,
        )

    @app.get("/chunks/{chunk_id}/audio")
    def get_chunk_audio(chunk_id: int):
        try:
            chunk = state.chunks[chunk_id]
        except IndexError:
            raise HTTPException(status_code=404, detail="Unknown chunk_id") from None
        return Response(content=chunk.audio_bytes, media_type=chunk.audio_mime_type)

    @app.post("/interruptions", response_model=InterruptionResponse)
    def map_interruption(request: InterruptionRequest):
        try:
            chunk = state.chunks[request.chunk_id]
        except IndexError:
            raise HTTPException(status_code=404, detail="Unknown chunk_id") from None
        mapping = map_elapsed_time_to_document(chunk, request.elapsed_ms)
        return InterruptionResponse(**mapping)

    return app


# ----------------------------- Startup flow -------------------------------- #


def load_document(path: Path) -> str:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Document path not found or unreadable: {path}")
    return path.read_text(encoding="utf-8")


def build_document_state(
    document_text: str,
    voice_id: str,
    chunking_strategy: str,
    region_name: str,
    cache_dir: Path,
) -> DocumentState:
    if chunking_strategy != "paragraph":
        raise ValueError(f"Unsupported chunking strategy: {chunking_strategy}")

    chunk_spans = chunk_document_paragraphs(document_text)
    session = boto3.Session()
    tts_cache = TtsCache(
        cache_dir=cache_dir,
        boto_session=session,
        region_name=region_name,
        voice_id=voice_id,
    )
    chunks: List[TextChunk] = []

    for chunk_id, (start, end) in enumerate(chunk_spans):
        text = document_text[start:end]
        logging.info("Synthesizing chunk %s (%s chars)", chunk_id, len(text))
        try:
            audio_bytes, speech_mark_records = tts_cache[text]
            speech_marks = _speech_marks_from_records(speech_mark_records)
        except (BotoCoreError, ClientError, RuntimeError) as exc:
            logging.error("Polly synthesis failed for chunk %s: %s", chunk_id, exc)
            raise

        chunks.append(
            TextChunk(
                chunk_id=chunk_id,
                text=text,
                doc_char_start=start,
                doc_char_end=end,
                audio_bytes=audio_bytes,
                speech_marks=speech_marks,
            )
        )

    return DocumentState(
        original_text=document_text,
        voice_id=voice_id,
        chunking_strategy=chunking_strategy,
        chunks=chunks,
    )


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
        "--chunking-strategy",
        default="paragraph",
        help="Chunking strategy to use (default: paragraph).",
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
            chunking_strategy=args.chunking_strategy,
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
