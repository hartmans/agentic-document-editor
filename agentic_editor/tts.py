from __future__ import annotations

import logging
from typing import List
import typing

from agentic_editor.document import DocumentState, ResumePoint, TextChunk, chunk_documents
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


def map_doc_char_to_offset(state: DocumentState, position: int) -> ResumePoint:
    """Resolve a document character offset to chunk id and audio offset."""
    chunk_id = state.find_chunk_id(position)
    if not state.chunks:
        return ResumePoint(chunkId=0, offsetMs=0, docCharPosition=0)
    chunk = state.chunks[chunk_id]
    _, word_marks = state.tts_cache[chunk.text]
    if not word_marks:
        return ResumePoint(
            chunkId=chunk_id,
            offsetMs=0,
            docCharPosition=chunk.doc_char_start,
        )

    relative_pos = max(0, position - chunk.doc_char_start)
    target_mark = None
    for mark in word_marks:
        if mark.chunk_char_start <= relative_pos:
            target_mark = mark
        else:
            break

    if target_mark is None:
        return ResumePoint(
            chunkId=chunk_id,
            offsetMs=0,
            docCharPosition=chunk.doc_char_start,
        )

    return ResumePoint(
        chunkId=chunk_id,
        offsetMs=target_mark.time_ms,
        docCharPosition=chunk.doc_char_start + target_mark.chunk_char_start,
    )


def build_document_state(
    document_text: str,
    *,
    tts_cache: TtsCache | None = None,
    session: object | None = None,
    config: dict[str, typing.Any] | None = None,
    prefix: str = "",
    position_hint: int | None = None,
) -> DocumentState:
    chunk_spans = chunk_documents(document_text)
    if tts_cache is None:
        tts_cache = TtsCache(config or {}, session, prefix=prefix)
        session = typing.cast(
            object | None,
            getattr(tts_cache, "session", session),
        )
    else:
        session = typing.cast(
            object | None,
            session or getattr(tts_cache, "session", None),
        )
    chunks: List[TextChunk] = []
    for chunk_id, (start, end) in enumerate(chunk_spans):
        text = document_text[start:end]
        chunk = TextChunk(
            text=text,
            doc_char_start=start,
            doc_char_end=end,
        )
        chunks.append(chunk)
    state = DocumentState(
        document=document_text,
        tts_cache=tts_cache,
        voice_id=tts_cache.voice_id,
        chunks=chunks,
        aws_session=session,
        region_name=tts_cache.region_name,
    )
    if position_hint is not None and chunks:
        preferred_chunk_id = state.find_chunk_id(position_hint)
        logging.info("Submitting preferred chunk %s first", preferred_chunk_id)
        tts_cache.submit(chunks[preferred_chunk_id].text)
        for chunk_id, chunk in enumerate(chunks):
            if chunk_id == preferred_chunk_id:
                continue
            logging.info("Submitting chunk %s (%s chars)", chunk_id, len(chunk.text))
            tts_cache.submit(chunk.text)
    else:
        for chunk_id, chunk in enumerate(chunks):
            logging.info("Submitting chunk %s (%s chars)", chunk_id, len(chunk.text))
            tts_cache.submit(chunk.text)

    return state


__all__ = [
    "TextChunk",
    "DocumentState",
    "map_elapsed_time_to_document",
    "map_doc_char_to_offset",
    "build_document_state",
]
