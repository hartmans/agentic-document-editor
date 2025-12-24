from __future__ import annotations

import logging
from typing import List
import typing

import boto3

from agentic_editor.document import DocumentState, TextChunk, chunk_documents
from agentic_editor.tts_cache import TtsCache


def map_elapsed_time_to_document(
    tts_cache: TtsCache, chunk: TextChunk, chunk_id: int, elapsed_ms: int
) -> dict:
    """Resolve an elapsed time within a chunk to document-level character offsets."""
    _, word_marks = tts_cache[chunk.text]
    if not word_marks:
        raise ValueError("Chunk has no word-level speech marks to map time.")
    
    # Before first mark: clamp to start of chunk.
    if elapsed_ms <= word_marks[0].time_ms:
        return {
            "chunkId": chunk_id,
            "offsetMs": elapsed_ms,
            "docCharPosition": chunk.doc_char_start,
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
        "chunkId": chunk_id,
        "offsetMs": elapsed_ms,
        "docCharPosition": chunk.doc_char_start + target_mark.chunk_char_start,
    }


def build_document_state(
    document_text: str,
    *,
    tts_cache: TtsCache | None = None,
    session: boto3.Session | None = None,
    config: dict[str, typing.Any] | None = None,
    prefix: str = "",
) -> DocumentState:
    chunk_spans = chunk_documents(document_text)
    if tts_cache is None:
        session = session or boto3.Session()
        tts_cache = TtsCache(config or {}, session, prefix=prefix)
    else:
        session = session or getattr(tts_cache, "session", boto3.Session())
    chunks: List[TextChunk] = []

    for chunk_id, (start, end) in enumerate(chunk_spans):
        text = document_text[start:end]
        logging.info("Submitting chunk %s (%s chars)", chunk_id, len(text))
        chunk = TextChunk(
            text=text,
            doc_char_start=start,
            doc_char_end=end,
        )
        chunks.append(chunk)
        tts_cache.submit(text)

    return DocumentState(
        document=document_text,
        tts_cache=tts_cache,
        voice_id=tts_cache.voice_id,
        chunks=chunks,
        aws_session=session,
        region_name=tts_cache.region_name,
    )


__all__ = [
    "TextChunk",
    "DocumentState",
    "map_elapsed_time_to_document",
    "build_document_state",
]
