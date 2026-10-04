"""Build document speech metadata, then register plans as a group."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .document import DocumentState, TextChunk, chunk_documents
from .speech import SpeechService

if TYPE_CHECKING:
    from radio_drama import RenderResult


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
    chunks = [TextChunk(document_text[start:end])
              for start, end in chunk_documents(document_text)]
    state = DocumentState(
        document=document_text, speech=speech,
        voice_id=speech.voice_id if speech else "", chunks=chunks,
    )
    state.prepare_changed_chunks()
    return state


async def submit_document_audio(
    state: DocumentState, *,
    response_state: DocumentState | None = None, eager: bool = True,
) -> None:
    if state.speech is None:
        return
    replies = response_state.chunks if response_state else []
    await state.speech.plan_chunks([*replies, *state.chunks])
    if eager:
        state.speech.prefetch([*replies, *state.chunks])


async def prepare_document_round(
    previous: DocumentState, draft: DocumentState, position: int,
) -> DocumentState:
    """Keep no-op rounds intact; introduce a local resume boundary when needed.

    Registering changed text first discovers persistent cache hits without
    rendering. Only a missing render requires replacing the containing chunk
    with new prefix/suffix plans. Reposition-only rounds never change chunking.
    """
    if previous.document == draft.document:
        return previous
    position = max(0, min(position, len(draft.document)))
    await asyncio.to_thread(draft.prepare_changed_chunks)
    if draft.speech:
        await draft.speech.plan_chunks(draft.chunks)
    idx = draft.find_chunk_id(position)
    if (draft.chunk_start(idx) < position < draft.chunk_end(idx)
            and (draft.speech is None or not draft.speech.audio_available(draft.chunks[idx]))):
        draft.split_chunk(position)
        await asyncio.to_thread(draft.prepare_changed_chunks)
        if draft.speech:
            await draft.speech.plan_chunks(draft.chunks)
    return draft


async def stream_document_audio(state: DocumentState, chunk_id: int) -> AsyncIterator[RenderResult]:
    """Playback PCM iterator for phase 2's encoder/HTTP response.

    The next chunk belongs to this revision, not to a shared TextChunk: later
    revisions may give the very same chunk a different successor.
    """
    if state.speech is None:
        raise RuntimeError("Speech is not configured")
    if not 0 <= chunk_id < len(state.chunks):
        raise IndexError("Unknown chunk_id")
    await state.speech.plan_chunks(state.chunks)
    following = next((chunk for chunk in state.chunks[chunk_id + 1:]
                      if chunk.dialogue_lines), None)
    started = False
    async with aclosing(state.speech.stream_audio(state.chunks[chunk_id])) as stream:
        async for result in stream:
            if result.frame_count and not started:
                started = True
                if following is not None:
                    state.speech.prefetch([following])
            yield result
