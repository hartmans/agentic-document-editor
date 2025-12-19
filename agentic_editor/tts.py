from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from agentic_editor.document import chunk_document_paragraphs
from agentic_editor.tts_cache import TtsCache


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
    aws_session: boto3.Session | None = None
    region_name: str | None = None


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


def build_document_state(
    document_text: str,
    voice_id: str,
    chunking_strategy: str,
    region_name: str,
    cache_dir: Path,
    session: boto3.Session | None = None,
) -> DocumentState:
    if chunking_strategy != "paragraph":
        raise ValueError(f"Unsupported chunking strategy: {chunking_strategy}")

    chunk_spans = chunk_document_paragraphs(document_text)
    session = session or boto3.Session()
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
        aws_session=session,
        region_name=region_name,
    )


__all__ = [
    "SpeechMark",
    "TextChunk",
    "DocumentState",
    "map_elapsed_time_to_document",
    "build_document_state",
]
