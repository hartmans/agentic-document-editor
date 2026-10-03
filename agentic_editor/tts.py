"""Build document speech metadata, then register plans as a group."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .document import DocumentState, ResumePoint, TextChunk, chunk_documents
from .speech import SpeechService


async def map_elapsed_time_to_document(
    speech: SpeechService, chunk: TextChunk, chunk_id: int, elapsed_ms: int,
) -> dict:
    marks = await speech.marks(chunk)
    position = chunk.doc_char_start
    for mark in marks:
        if mark.time_ms > elapsed_ms:
            break
        position = chunk.doc_char_start + mark.chunk_char_start
    return {"chunkId": chunk_id, "offsetMs": elapsed_ms, "docCharPosition": position}


async def map_doc_char_to_offset(state: DocumentState, position: int) -> ResumePoint:
    chunk_id = state.find_chunk_id(position)
    if not state.chunks:
        return ResumePoint(chunkId=0, offsetMs=0, docCharPosition=0)
    chunk = state.chunks[chunk_id]
    marks = await state.speech.marks(chunk) if state.speech else []
    selected = None
    for mark in marks:
        if mark.chunk_char_start > position - chunk.doc_char_start:
            break
        selected = mark
    return ResumePoint(
        chunkId=chunk_id, offsetMs=selected.time_ms if selected else 0,
        docCharPosition=chunk.doc_char_start + (selected.chunk_char_start if selected else 0),
    )


def build_document_state(
    document_text: str, *, speech: SpeechService | None = None,
    config: dict[str, Any] | None = None, document_path: Path | None = None,
) -> DocumentState:
    """Prepare text/dialogue without registering or rendering audio.

    Text-only document tools need no speech service. The server supplies a
    configured service shared by all document revisions and assistant replies.
    """
    if speech is None and config is not None:
        if document_path is None:
            raise ValueError("document_path is required to configure speech caching")
        speech = SpeechService(config, document_path)
    chunks = [TextChunk(document_text[start:end], start, end)
              for start, end in chunk_documents(document_text)]
    if speech:
        for chunk in chunks:
            speech.prepare_chunk(chunk)
    return DocumentState(
        document=document_text, speech=speech,
        voice_id=speech.voice_id if speech else "", chunks=chunks,
    )


async def submit_document_audio(
    state: DocumentState, *, position_hint: int | None = None,
    response_state: DocumentState | None = None,
) -> None:
    if state.speech is None:
        return
    replies = response_state.chunks if response_state else []
    await state.speech.plan_chunks([*replies, *state.chunks])
    first = state.find_chunk_id(position_hint) if position_hint is not None else 0
    state.speech.prefetch([*replies, *state.chunks[first:]])
