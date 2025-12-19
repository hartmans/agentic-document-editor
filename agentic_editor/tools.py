"""LangChain tools for editing the in-memory document state.

This module intentionally ignores audio/chunk bookkeeping. It focuses purely
on updating the `DocumentState.document` text so the LLM can issue small edit
commands without re-running TTS.
"""

from __future__ import annotations

from typing import Any, Annotated

from langchain_core.tools import InjectedToolArg, tool
from langgraph.prebuilt.tool_node import InjectedState

from agentic_editor.document import DocumentState

SEARCH_RADIUS = 400


@tool
def replace_text(
    offset: int,
    old_text: str,
    new_text: str,
    *,
    state: Annotated[DocumentState, InjectedState("document")],
    context: Annotated[Any, InjectedToolArg] = None,
) -> str:
    """Replace an exact substring close to a character offset.

    Args:
        offset: 0-based character position near where `old_text` should appear.
        old_text: Exact text to replace.
        new_text: Replacement text.

    Returns:
        Human-readable summary of the edit that was applied.

    Notes:
        - Searches up to SEARCH_RADIUS characters forward from `offset`; if not
          found, searches the same distance backward. The first exact match is
          replaced.
        - Requires injected state carrying `document: DocumentState`; only the
          text is updated (chunk/audio metadata is ignored).
    """

    doc_state = state
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

    return (
        f"Replaced text starting at char {index} with new_text; "
        f"new document length is {len(updated)}."
    )


__all__ = ["replace_text"]
