"""
Agentic document editor foundation: chunk text, synthesize speech with Amazon Polly,
track timing via speech marks, and expose a FastAPI server for playback and
interruption-to-text mapping.
"""

from __future__ import annotations

import argparse
import json
import logging
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import boto3
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

    def get_chunk(self, chunk_id: int) -> TextChunk:
        try:
            return self.chunks[chunk_id]
        except IndexError as exc:
            raise KeyError(f"Unknown chunk_id {chunk_id}") from exc


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


# ----------------------------- Polly helpers ------------------------------- #


def _call_polly(
    client,
    *,
    text: str,
    voice_id: str,
    output_format: str,
    speech_mark_types: Optional[List[str]] = None,
) -> dict:
    """Call Polly, preferring neural engine when available, falling back to standard."""
    base_params = {
        "Text": text,
        "VoiceId": voice_id,
        "OutputFormat": output_format,
    }
    if speech_mark_types:
        base_params["SpeechMarkTypes"] = speech_mark_types

    # Attempt neural engine first.
    try:
        return client.synthesize_speech(**base_params, Engine="neural")
    except ClientError as err:
        logging.warning("Neural engine unavailable, falling back to standard: %s", err)
    # Fall back to standard engine.
    return client.synthesize_speech(**base_params)


def synthesize_audio(client, text: str, voice_id: str) -> bytes:
    response = _call_polly(client, text=text, voice_id=voice_id, output_format="mp3")
    stream = response.get("AudioStream")
    if stream is None:
        raise RuntimeError("Polly response missing AudioStream for audio synthesis")
    return stream.read()


def synthesize_speech_marks(client, text: str, voice_id: str) -> List[SpeechMark]:
    response = _call_polly(
        client,
        text=text,
        voice_id=voice_id,
        output_format="json",
        speech_mark_types=["word", "sentence"],
    )
    stream = response.get("AudioStream")
    if stream is None:
        raise RuntimeError("Polly response missing AudioStream for speech marks")
    raw = stream.read().decode("utf-8")
    lines = [line for line in raw.splitlines() if line.strip()]
    return list(_parse_speech_mark_lines(lines, text))


def _byte_to_char_offset(byte_offset: int, cumulative_bytes: List[int]) -> int:
    """Translate a byte offset into a character offset using cumulative byte counts."""
    idx = bisect_right(cumulative_bytes, byte_offset) - 1
    return max(idx, 0)


def _parse_speech_mark_lines(
    lines: Iterable[str], chunk_text: str
) -> Iterable[SpeechMark]:
    cumulative_bytes: List[int] = [0]
    for ch in chunk_text:
        cumulative_bytes.append(cumulative_bytes[-1] + len(ch.encode("utf-8")))

    for line in lines:
        data = json.loads(line)
        mark_type = data.get("type")
        if mark_type not in {"word", "sentence"}:
            continue
        start_byte = int(data.get("start", 0))
        end_byte = int(data.get("end", start_byte))
        chunk_char_start = _byte_to_char_offset(start_byte, cumulative_bytes)
        chunk_char_end = _byte_to_char_offset(end_byte, cumulative_bytes)
        yield SpeechMark(
            time_ms=int(data["time"]),
            mark_type=mark_type,
            text=data.get("value", ""),
            chunk_char_start=chunk_char_start,
            chunk_char_end=chunk_char_end,
        )


# ----------------------------- Mapping logic ------------------------------- #


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
        "word_text": target_mark.text,
        "chunk_char_start": target_mark.chunk_char_start,
        "chunk_char_end": target_mark.chunk_char_end,
        "doc_char_start": chunk.doc_char_start + target_mark.chunk_char_start,
        "doc_char_end": chunk.doc_char_start + target_mark.chunk_char_end,
    }


# ----------------------------- FastAPI models ------------------------------ #


class SpeechMarkModel(BaseModel):
    time_ms: int
    mark_type: str
    text: str
    chunk_char_start: int
    chunk_char_end: int


class ChunkMetadataModel(BaseModel):
    chunk_id: int
    doc_char_start: int
    doc_char_end: int
    preview: str


class ChunkDetailModel(BaseModel):
    chunk_id: int
    text: str
    doc_char_start: int
    doc_char_end: int
    speech_marks: List[SpeechMarkModel]


class InterruptionRequest(BaseModel):
    chunk_id: int
    elapsed_ms: int = Field(..., ge=0)


class InterruptionResponse(BaseModel):
    chunk_id: int
    elapsed_ms: int
    word_text: str
    chunk_char_start: int
    chunk_char_end: int
    doc_char_start: int
    doc_char_end: int


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

    @app.get("/chunks", response_model=List[ChunkMetadataModel])
    def list_chunks():
        return [
            ChunkMetadataModel(
                chunk_id=chunk.chunk_id,
                doc_char_start=chunk.doc_char_start,
                doc_char_end=chunk.doc_char_end,
                preview=chunk.text[:80],
            )
            for chunk in state.chunks
        ]

    @app.get("/chunks/{chunk_id}", response_model=ChunkDetailModel)
    def get_chunk(chunk_id: int):
        try:
            chunk = state.get_chunk(chunk_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Unknown chunk_id")
        return ChunkDetailModel(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            doc_char_start=chunk.doc_char_start,
            doc_char_end=chunk.doc_char_end,
            speech_marks=[
                SpeechMarkModel(
                    time_ms=mark.time_ms,
                    mark_type=mark.mark_type,
                    text=mark.text,
                    chunk_char_start=mark.chunk_char_start,
                    chunk_char_end=mark.chunk_char_end,
                )
                for mark in chunk.speech_marks
            ],
        )

    @app.get("/chunks/{chunk_id}/audio")
    def get_chunk_audio(chunk_id: int):
        try:
            chunk = state.get_chunk(chunk_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Unknown chunk_id")
        return Response(content=chunk.audio_bytes, media_type=chunk.audio_mime_type)

    @app.post("/interruptions", response_model=InterruptionResponse)
    def map_interruption(request: InterruptionRequest):
        try:
            chunk = state.get_chunk(request.chunk_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Unknown chunk_id")
        mapping = map_elapsed_time_to_document(chunk, request.elapsed_ms)
        return InterruptionResponse(**mapping)

    return app


# ----------------------------- Startup flow -------------------------------- #


def load_document(path: Path) -> str:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Document path not found or unreadable: {path}")
    return path.read_text(encoding="utf-8")


def build_document_state(
    document_text: str, voice_id: str, chunking_strategy: str, region_name: str
) -> DocumentState:
    if chunking_strategy != "paragraph":
        raise ValueError(f"Unsupported chunking strategy: {chunking_strategy}")

    chunk_spans = chunk_document_paragraphs(document_text)
    polly_client = boto3.client("polly", region_name=region_name)
    chunks: List[TextChunk] = []

    for chunk_id, (start, end) in enumerate(chunk_spans):
        text = document_text[start:end]
        logging.info("Synthesizing chunk %s (%s chars)", chunk_id, len(text))
        try:
            audio_bytes = synthesize_audio(polly_client, text=text, voice_id=voice_id)
            speech_marks = synthesize_speech_marks(
                polly_client, text=text, voice_id=voice_id
            )
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
        )
    except Exception as exc:
        logging.error("Failed to preprocess document: %s", exc)
        raise SystemExit(1) from exc

    app = create_app(state)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
