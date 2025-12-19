from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Sequence

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from agentic_editor.document import DocumentState, SpeechMark, TextChunk, chunk_documents
from agentic_editor.tts_cache import TtsCache


def _speech_marks_from_records(records: Sequence[dict]) -> List[SpeechMark]:
    """Convert raw speech mark records to word-level SpeechMark objects."""
    marks: List[SpeechMark] = []
    for rec in records:
        try:
            marks.append(
                SpeechMark(
                    time_ms=int(rec["time_ms"]),
                    chunk_char_start=int(rec["chunk_char_start"]),
                    chunk_char_end=int(rec["chunk_char_end"]),
                )
            )
        except (KeyError, ValueError, TypeError):
            continue
    marks.sort(key=lambda m: m.time_ms)
    return marks


def map_elapsed_time_to_document(chunk: TextChunk, chunk_id: int, elapsed_ms: int) -> dict:
    """Resolve an elapsed time within a chunk to document-level character offsets."""
    word_marks = chunk.speech_marks
    if not word_marks:
        raise ValueError("Chunk has no word-level speech marks to map time.")
    word_marks.sort(key=lambda m: m.time_ms)

    # Before first mark: clamp to start of chunk.
    if elapsed_ms <= word_marks[0].time_ms:
        return {
            "chunk_id": chunk_id,
            "elapsed_ms": elapsed_ms,
            "doc_char_position": chunk.doc_char_start,
        }

    target_mark = word_marks[0]
    for mark in word_marks:
        if mark.time_ms <= elapsed_ms:
            target_mark = mark
        else:
            break

    if elapsed_ms > word_marks[-1].time_ms:
        target_mark = word_marks[-1]

    return {
        "chunk_id": chunk_id,
        "elapsed_ms": elapsed_ms,
        "doc_char_position": chunk.doc_char_start + target_mark.chunk_char_start,
    }


def build_document_state(
    document_text: str,
    voice_id: str,
    region_name: str,
    cache_dir: Path,
    session: boto3.Session | None = None,
) -> DocumentState:
    chunk_spans = chunk_documents(document_text)
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
                text=text,
                doc_char_start=start,
                doc_char_end=end,
                audio_bytes=audio_bytes,
                speech_marks=speech_marks,
            )
        )

    return DocumentState(
        document=document_text,
        voice_id=voice_id,
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
