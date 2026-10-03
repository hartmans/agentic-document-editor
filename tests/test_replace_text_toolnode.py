import json

import pytest

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.types import Command

from agentic_editor.document import DocumentState
from agentic_editor.graph import DocumentUpdate
from agentic_editor.tools import (
    pause_playback,
    replace_text,
    reposition_playback,
    stop_playback,
)



@pytest.mark.parametrize(("position", "change_start", "expected_position"), [
    (200, 100, 100),  # Rewind to an earlier change.
    (200, 200, 200),
    (200, 201, 200),
    (200, 299, 200),
    (200, 300, 200),  # Exactly 100 characters ahead retains context.
    (200, 301, 301),
    (200, 500, 500),
])
def test_replace_text_toolnode_executes(position, change_start, expected_position):
    prefix = "x" * change_start
    document = prefix + "Hello world"
    doc_state = DocumentState(
        document=document,
        voice_id="Joanna",
    )
    tool_node = ToolNode([replace_text])

    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_1",
                "name": "replace_text",
                "args": {
                    "before": "",
                    "old_text": "Hello",
                    "new_text": "Hi",
                    "after": " world",
                },
            }
        ],
    )

    runtime = Runtime()

    result = tool_node.invoke(
        {
            "messages": [ai_msg],
            "document_state": doc_state,
            "document": document,
            "position": position,
        },
        runtime=runtime,
    )

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], Command)
    update = result[0].update
    # ToolNode should have appended a ToolMessage and returned a document update.
    assert isinstance(update["messages"][-1], ToolMessage)
    assert update["messages"][-1].tool_call_id == "call_1"
    assert update["document"] == prefix + "Hi world"
    assert update["position"] == expected_position
    assert json.loads(update["messages"][-1].content) == {
        "position": change_start,
        "text": "Hi world",
    }
    assert doc_state.document == document


def test_reposition_playback_closes_tool_turn_and_sets_response_text():
    document = "Start here and continue."
    doc_state = DocumentState(
        document=document,
        voice_id="Joanna",
    )
    tool_node = ToolNode([reposition_playback])
    ai_msg = AIMessage(
        content="",
        tool_calls=[{
            "id": "call_1",
            "name": "reposition_playback",
            "args": {
                "before": "",
                "target": "continue",
                "after": ".",
                "message": "Resuming there.",
            },
        }],
    )

    result = tool_node.invoke(
        {
            "messages": [ai_msg],
            "document_state": doc_state,
            "document": document,
        },
        runtime=Runtime(),
    )

    assert isinstance(result, list)
    update = result[0].update
    assert update["response_text"] == "Resuming there."
    assert isinstance(update["messages"][0], ToolMessage)
    assert json.loads(update["messages"][0].content) == {
        "position": document.index("continue"),
        "text": "continue.",
    }
    assert update["messages"][1] == AIMessage(content="Resuming there.")


def test_pause_playback_sets_message_and_disposition():
    ai_msg = AIMessage(
        content="",
        tool_calls=[{
            "id": "call_1",
            "name": pause_playback.name,
            "args": {"message": "Please answer this question."},
        }],
    )

    result = ToolNode([pause_playback]).invoke(
        {"messages": [ai_msg]},
        runtime=Runtime(),
    )

    assert isinstance(result, list)
    command = result[0]
    assert command.goto == "final"
    assert command.update["disposition"] == "pause"
    assert command.update["response_text"] == "Please answer this question."
    tool_message, assistant_message = command.update["messages"]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "Please answer this question."
    assert tool_message.tool_call_id == "call_1"
    assert assistant_message == AIMessage(content="Please answer this question.")


def test_stop_playback_takes_no_message_and_announces_how_to_resume():
    assert "message" not in stop_playback.args
    ai_msg = AIMessage(
        content="",
        tool_calls=[{"id": "call_1", "name": stop_playback.name, "args": {}}],
    )

    result = ToolNode([stop_playback]).invoke(
        {"messages": [ai_msg]},
        runtime=Runtime(),
    )

    command = result[0]
    message = "Playback stopped. Press Continue to resume."
    assert command.goto == "final"
    assert command.update["disposition"] == "stop"
    assert command.update["response_text"] == message
    tool_message, assistant_message = command.update["messages"]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == message
    assert tool_message.tool_call_id == "call_1"
    assert assistant_message == AIMessage(content=message)


def test_replace_text_searches_the_current_mutable_document():
    original = "alpha beta"
    current = "alpha beta gamma"
    doc_state = DocumentState(
        document=original,
        voice_id="Joanna",
    )
    ai_msg = AIMessage(
        content="",
        tool_calls=[{
            "id": "call_1",
            "name": "replace_text",
            "args": {
                "before": "beta ",
                "old_text": "gamma",
                "new_text": "delta",
                "after": "",
            },
        }],
    )

    result = ToolNode([replace_text]).invoke(
        {
            "messages": [ai_msg],
            "document_state": doc_state,
            "document": current,
            "position": 0,
        },
        runtime=Runtime(),
    )

    assert result[0].update["document"] == "alpha beta delta"
