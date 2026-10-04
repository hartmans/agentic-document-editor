
from __future__ import annotations

import json
import re
from typing import Any, Annotated
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import InjectedToolArg, InjectedToolCallId, tool
from langgraph.types import Command
from langgraph.prebuilt.tool_node import InjectedState
from pydantic import Field
from agentic_editor.document import DocumentState

DEFAULT_EXCERPT_LENGTH = 250


def document_excerpt(
    document: str,
    position: int,
    length: int = DEFAULT_EXCERPT_LENGTH,
) -> dict[str, int | str]:
    """Return a bounded excerpt of document text in the standard tool format."""
    bounded_position = max(0, min(position, len(document)))
    bounded_length = max(0, length)
    return {
        "position": bounded_position,
        "text": document[bounded_position:bounded_position + bounded_length],
    }


def _tool_response(document: str, position: int, length: int = DEFAULT_EXCERPT_LENGTH) -> str:
    return json.dumps(
        document_excerpt(document, position, length),
        ensure_ascii=False,
    )


@tool(infer_schema=True, parse_docstring=True)
def replace_text(
    before: Annotated[str, Field(max_length=120)],
    old_text: str,
    new_text: str,
    after: Annotated[str, Field(max_length=120)],
    *,
    state: Annotated[DocumentState, InjectedState("document_state")],
    document: Annotated[str, InjectedState("document")],
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
        JSON containing the edit position and updated document text from there.

    """

    result = state.search_text(before, old_text, after)
    if result.index is None:
        return result.render_error(target_field_name="old_text")
    state.update(new_text, result.index, result.index + len(old_text))
    updated = state.document
    # Keep nearby preceding context, but jump to edits farther ahead.
    resume_position = (
        result.index if result.index < position or result.index > position + 100
        else position
    )
    return Command(update={
        'document': updated,
        'document_state': state,
        'position': resume_position,
            'messages': [ToolMessage(
                content=_tool_response(
                    updated,
                    result.index,
                    max(len(old_text + after), len(new_text + after)),
                ),
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
    document: Annotated[str, InjectedState("document")],
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
    result = state.search_text(before, target, after)
    if result.index is None:
        return result.render_error()
    return Command(update={
        'position': result.index,
        'response_text': message or '',
        'messages': [ToolMessage(
            content=_tool_response(document, result.index),
            tool_call_id=tool_call_id,
        ),
        AIMessage(content=message or '')],
    },
    goto='final')


def _finish_playback_action(message: str, disposition: str, tool_call_id: str) -> Command:
    return Command(
        update={
            "disposition": disposition,
            "response_text": message,
            "messages": [
                ToolMessage(content=message, tool_call_id=tool_call_id),
                AIMessage(content=message),
            ],
        },
        goto="final",
    )


@tool(infer_schema=True, parse_docstring=True)
def pause_playback(
    message: str,
    *,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Pause playback, end this interaction, and present a message to the user.

    Use this tool when the editor needs the user to explicitly answer a question,
    or when the user explicitly directs you to pause playback. Voice activity
    detection remains armed while playback is paused.

    Args:
        message: The final response presented to the user.
    """
    return _finish_playback_action(message, "pause", tool_call_id)


@tool(infer_schema=True)
def stop_playback(
    *,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Stop playback, end this interaction, and announce how to resume.

    Use this tool only when the user explicitly asks to end their editing session.
    Stopping disarms voice activity detection, so the user must interact with their
    device to resume. Unless the user explicitly asks to end editing, use
    pause_playback instead.
    """
    return _finish_playback_action(
        "Playback stopped. Press Continue to resume.",
        "stop",
        tool_call_id,
    )


@tool(infer_schema=True, parse_docstring=True)
def read_document(
    position: Annotated[int, Field(ge=0)] = 0,
    length: Annotated[int, Field(ge=0)] = DEFAULT_EXCERPT_LENGTH,
    *,
    document: Annotated[str, InjectedState("document")],
) -> str:
    """Read text from the current document starting at a character position.

    Args:
        position: Character position at which to start reading.
        length: Maximum number of characters to return.
    """
    return _tool_response(document, position, length)


@tool(infer_schema=True, parse_docstring=True)
def search_document(
    search_text: Annotated[str, Field(min_length=1)],
    length: Annotated[int, Field(ge=0)] = DEFAULT_EXCERPT_LENGTH,
    starting_position: Annotated[int, Field(ge=0)] = 0,
    max_hits: Annotated[int, Field(ge=1)] = 10,
    *,
    document: Annotated[str, InjectedState("document")],
) -> str:
    """Search the current document case-insensitively and return excerpts around each hit.

    Args:
        search_text: Text to find case-insensitively in the current document.
        length: Maximum number of characters in each returned excerpt.
        starting_position: Character position at which to begin searching.
        max_hits: Maximum number of matching excerpts to return.
    """
    hits: list[dict[str, int | str]] = []
    search_position = min(starting_position, len(document))
    matches = re.finditer(
        re.escape(search_text),
        document[search_position:],
        flags=re.IGNORECASE,
    )
    for match in matches:
        if len(hits) >= max_hits:
            break
        hit = search_position + match.start()
        surrounding_chars = max(0, length - len(search_text))
        excerpt_position = hit - surrounding_chars // 2
        excerpt_position = max(0, min(excerpt_position, max(0, len(document) - length)))
        hits.append(document_excerpt(document, excerpt_position, length))
    return json.dumps(hits, ensure_ascii=False)


__all__ = [
    "document_excerpt",
    "pause_playback",
    "read_document",
    "replace_text",
    "reposition_playback",
    "search_document",
    "stop_playback",
]
