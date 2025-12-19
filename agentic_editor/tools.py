"""LangChain tools for editing the in-memory document state.

This module intentionally ignores audio/chunk bookkeeping. It focuses purely
on updating the `DocumentState.document` text so the LLM can issue small edit
commands without re-running TTS.
"""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping

from langchain_core.tools import tool

from agentic_editor.document import DocumentState

SEARCH_RADIUS = 400


def _extract_document_state(state: Any) -> DocumentState:
    """Pull a DocumentState off the injected state container.

    Supports either attribute-style access (state.document) or mapping-style
    access (state["document"]). Raises a ValueError if no DocumentState is
    found so the model gets immediate feedback.
    """

    doc_state: DocumentState | None = None
    if isinstance(state, Mapping) and "document" in state:
        maybe_state = state["document"]
        if isinstance(maybe_state, DocumentState):
            doc_state = maybe_state

    if doc_state is None and hasattr(state, "document"):
        maybe_state = getattr(state, "document")
        if isinstance(maybe_state, DocumentState):
            doc_state = maybe_state

    if doc_state is None:
        raise ValueError(
            "Injected state must include a DocumentState at `state.document`."
        )
    return doc_state


@tool
def replace_text(
    offset: int,
    old_text: str,
    new_text: str,
    *,
    state: Any = None,
    context: Any = None,
) -> str:
    """Replace an exact substring in the document near a character offset.

    Model usage guidance:
    - Provide `offset` near where you expect `old_text` to appear (0-based char).
    - The tool searches up to SEARCH_RADIUS characters *forward* from offset
      first. If not found, it searches the same window *backward* from offset.
      The first exact match of `old_text` is replaced with `new_text`.
    - The editable text lives in `state.document`, which is a `DocumentState`.
      The function mutates that DocumentState in-place and also writes it back
      to mapping-based state containers when possible.
    - Chunking/audio metadata is intentionally ignored for now; this only keeps
      the document text in sync.
    - `context` is injected LangChain context (not used yet but accepted).
    """

    if state is None:
        raise ValueError("Injected state is required to locate the document.")

    doc_state = _extract_document_state(state)
    document = doc_state.document

    pivot = max(0, min(offset, len(document)))

    # Prefer searching forward from the pivot, then backward if needed.
    forward_end = min(len(document), pivot + SEARCH_RADIUS)
    index = document.find(old_text, pivot, forward_end)

    if index == -1:
        backward_start = max(0, pivot - SEARCH_RADIUS)
        index = document.rfind(old_text, backward_start, pivot)

    if index == -1:
        raise ValueError("old_text was not found near the provided offset.")

    updated = document[:index] + new_text + document[index + len(old_text) :]
    doc_state.document = updated

    # Write back to mapping-style state so downstream runnables see the change.
    if isinstance(state, MutableMapping):
        state["document"] = doc_state
    elif hasattr(state, "document"):
        setattr(state, "document", doc_state)

    return (
        f"Replaced text starting at char {index} with new_text; "
        f"new document length is {len(updated)}."
    )


__all__ = ["replace_text"]
