
from __future__ import annotations

import typing
from typing import Any, Annotated
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import InjectedToolArg, InjectedToolCallId, tool
from langgraph.types import Command
from langgraph.prebuilt.tool_node import InjectedState

from agentic_editor.document import DocumentState
from agentic_editor.graph import DocumentUpdate

FOCUS_RADIUS = 400


def _search_text(document: str, text: str, offset: int) -> tuple[int | None, str]:
    pivot = max(0, min(offset, len(document)))

    # Prefer searching forward from the pivot, then backward if needed.
    forward_end = min(len(document), pivot + FOCUS_RADIUS)
    index = document.find(text, pivot, forward_end)

    if index == -1:
        backward_start = max(0, pivot - FOCUS_RADIUS)
        index = document.rfind(text, backward_start, pivot)

    snippet_start = max(0, min(offset, len(document)))
    snippet_end = min(len(document), snippet_start + FOCUS_RADIUS)
    snippet = document[snippet_start:snippet_end]
    if index == -1:
        return None, snippet
    return index, snippet


@tool(infer_schema=True)
def replace_text(
    offset: int,
    old_text: str,
    new_text: str,
    *,
    state: Annotated[DocumentState, InjectedState("document_state")],
    context: Annotated[Any, InjectedToolArg] = None,
    tool_call_id: Annotated[str, InjectedToolCallId],
    position: Annotated[int, InjectedState('position')],
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

    """

    document = typing.cast(str, state.document)

    index, snippet = _search_text(document, old_text, offset)

    if index is None:
        return f"'Error: Unable to find `{old_text}' near offset {offset}; around that position I actually found {snippet}. Update your request and resubmit."
    updated = document[:index] + new_text + document[index + len(old_text) :]
    return Command(update={
        'document': updated,
        'position': min(position, index),
            'messages': [ToolMessage(
                content="Document updated. Now describe your change so the user knows what you did.",
                tool_call_id=tool_call_id,
            )],
    })

@tool(infer_schema=True)
def reposition_playback(
    offset: int,
    text: str,
    message: str = "",
    *,
    state: Annotated[DocumentState, InjectedState("document_state")],
    context: Annotated[Any, InjectedToolArg] = None,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command | str:
    '''
    Use this tool if you are asked to reposition to a particular point in the document or asked to play from a given point in the document. Select the position at which the document continues playing back to the author. This should be the last action taken by the LLM; it ends the conversation at least until the author interrupts again.
    Select the character offset where playback should resume; include the text you find at that offset to confirm that you have selected the intended position.

    Args:
        offset: The character offset at which *text* appears in the document; you must be within {FOCUS_RADIUS} characters.
        text: Start playback immediately before an occurance  of *text* around *offset* in the document
        message: If supplied the final response to the author before playback resumes. If not supplied, playback will resume with no response.
    '''
    document = typing.cast(str, state.document)
    index, snippet = _search_text(document, text, offset)

    if index is None:
        return f"'Error: Unable to find `{text}' near offset {offset}; around that position I actually found {snippet}. Update your request and use the reposition_playback tool again with a correct offset."
    return Command(update={
        'position': index,
        'messages': [ToolMessage(
            content='success',
            tool_call_id=tool_call_id,
        ),
        AIMessage(content=message or '')],
    },
    goto='final')

__all__ = ["replace_text", "reposition_playback"]
