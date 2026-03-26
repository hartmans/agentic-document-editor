
from __future__ import annotations

import typing
from typing import Any, Annotated
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import InjectedToolArg, InjectedToolCallId, tool
from langgraph.types import Command
from langgraph.prebuilt.tool_node import InjectedState
from pydantic import Field
from agentic_editor.document import DocumentState

@tool(infer_schema=True, parse_docstring=True)
def replace_text(
    before: Annotated[str, Field(max_length=120)],
    old_text: str,
    new_text: str,
    after: Annotated[str, Field(max_length=120)],
    *,
    state: Annotated[DocumentState, InjectedState("document_state")],
    context: Annotated[Any, InjectedToolArg] = None,
    tool_call_id: Annotated[str, InjectedToolCallId],
    position: Annotated[int, InjectedState('position')],
) -> Command|str:
    """Edit the document. This tool replaces the text *old_text* with the text *new_text* in the document. *offset* should be the character position at the beginning of *old_text*.
    To delete text, Place the text to delete in *old_text* and include empty *new_text*.
    To add new text, Use *old_text* as an anchor for where the new text should be added, and include the contents of *old_text* in *new_text* so that text is added.
    This tool looks for old_text between before and after and replaces it with new_text. The combination of before followed by old_text followed by after must be unique in the document.

    Args:
        before: A few words from the document just before the old_text
        old_text: Exact text to replace.
        after: A few words just after the modified
        new_text: Replacement text.

    Returns:
        Instructions for the LLM.

    """

    document = typing.cast(str, state.document)

    result = state.search_text(before, old_text, after)
    if result.index is None:
        return result.render_error()
    updated = document[:result.index] + new_text + document[result.index + len(old_text) :]
    return Command(update={
        'document': updated,
        'position': min(position, result.index),
            'messages': [ToolMessage(
                content="Document updated. Now briefly describe your change so the user knows what you did.",
                tool_call_id=tool_call_id,
            )],
    })

@tool(infer_schema=True, parse_docstring=True)
def reposition_playback(
    before: Annotated[str, Field(max_length=120)],
    target: str,
    after: Annotated[str, Field(max_length=120)],
    message: str = "",
    *,
    state: Annotated[DocumentState, InjectedState("document_state")],
    context: Annotated[Any, InjectedToolArg] = None,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command | str:
    '''
    Use this tool if you are asked to reposition to a particular point in the document or asked to play from a given point in the document. Select the position at which the document continues playing back to the author. This should be the last action taken by the LLM; it ends the conversation at least until the author interrupts again.
    Playback resumes at target, a string that must appear in the document between *before* and *after.
    
    Args:
        before: Short excerpt of Text before target (a few words)
        target: Start playback immediately before an occurance  of **target* between *before* and *after* in the document.
        after:  Short excerpt of text appearing after *target* (a few words)
        message: If supplied the final response to the author before playback resumes. If not supplied, playback will resume with no response.
    '''
    document = typing.cast(str, state.document)
    result = state.search_text(before, target, after)
    if result.index is None:
        return result.render_error()
    return Command(update={
        'position': result.index,
        'messages': [ToolMessage(
            content='success',
            tool_call_id=tool_call_id,
        ),
        AIMessage(content=message or '')],
    },
    goto='final')

__all__ = ["replace_text", "reposition_playback"]
