
from __future__ import annotations

from typing import Any, Annotated
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolArg, InjectedToolCallId, tool
from langgraph.types import Command
from langgraph.prebuilt.tool_node import InjectedState

from agentic_editor.document import DocumentState
from agentic_editor.graph import DocumentUpdate

SEARCH_RADIUS = 400


@tool
def replace_text(
    offset: int,
    old_text: str,
    new_text: str,
    *,
    state: Annotated[DocumentState, InjectedState("document")],
    context: Annotated[Any, InjectedToolArg] = None,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command|str:
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
        return f"'Error: Unable to find `{old_text}' near offset {offset}; around that position I actually found {document[offset:offset+SEARCH_RADIUS]}. Update your request and resubmit."
    updated = document[:index] + new_text + document[index + len(old_text) :]
    return Command(update={
        'document_update': DocumentUpdate(
            offset=index,
            length=len(old_text),
            new_text=new_text),
            'messages': [ToolMessage(
                content="Document updated. Now describe your change so the user knows what you did.",
                tool_call_id=tool_call_id,
            )],
    })

__all__ = ["replace_text"]
