
from __future__ import annotations

import typing
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
    """Edit the document. This tool replaces the text *old_text* with the text *new_text* in the document. *offset* should be the character position at the beginning of *old_text*.
    To delete text, Place the text to delete in *old_text* and include empty *new_text*.
    To add new text, Use *old_text* as an anchor for where the new text should be added, and include the contents of *old_text* in *new_text* so that text is added.
    

    Args:
        offset: 0-based character position near where `old_text` should appear.
        old_text: Exact text to replace.
        new_text: Replacement text.

    Returns:
        Instructions for the LLM.

    Notes:
        - Searches up to SEARCH_RADIUS characters forward from `offset`; if not
          found, searches the same distance backward. The first exact match is
          replaced.
    """

    document = typing.cast(str, state.document)

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
        'document': updated,
            'messages': [ToolMessage(
                content="Document updated. Now describe your change so the user knows what you did.",
                tool_call_id=tool_call_id,
            )],
    })

__all__ = ["replace_text"]
