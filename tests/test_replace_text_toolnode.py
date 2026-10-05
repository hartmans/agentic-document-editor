import json

import pytest

from langchain_core.messages import AIMessage, ToolMessage

from agentic_editor.document import DocumentState
from agentic_editor.tools import (
    pause_playback,
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
def test_replace_text_graph_executes(tool_graph, position, change_start, expected_position):
    prefix = "x" * change_start
    document = prefix + "Hello world"
    doc_state = DocumentState(
        document=document,
        voice_id="Joanna",
    )

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

    result = tool_graph(doc_state, [ai_msg, AIMessage(content="Done.")], position=position)
    draft = result["document_state"]
    tool_message = result["messages"][-2]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.tool_call_id == "call_1"
    assert draft.document == prefix + "Hi world"
    assert result["position"] == expected_position
    assert result["response_text"] == "Done."
    assert json.loads(tool_message.content) == {
        "position": change_start,
        "text": "Hi world",
    }
    assert doc_state.document == document


@pytest.mark.parametrize("message", ["Resuming there.", ""])
def test_reposition_playback_closes_tool_turn_and_sets_response_text(tool_graph, message):
    document = "Start here and continue."
    doc_state = DocumentState(
        document=document,
        voice_id="Joanna",
    )
    ai_msg = AIMessage(
        content="",
        tool_calls=[{
            "id": "call_1",
            "name": "reposition_playback",
            "args": {
                "before": "",
                "target": "continue",
                "after": ".",
                "message": message,
            },
        }],
    )

    update = tool_graph(doc_state, [ai_msg])

    assert update["response_text"] == message
    assert isinstance(update["messages"][-2], ToolMessage)
    assert json.loads(update["messages"][-2].content) == {
        "position": document.index("continue"),
        "text": "continue.",
    }
    assert update["messages"][-1] == AIMessage(content=message)


def test_pause_playback_sets_message_and_disposition(tool_graph):
    ai_msg = AIMessage(
        content="",
        tool_calls=[{
            "id": "call_1",
            "name": pause_playback.name,
            "args": {"message": "Please answer this question."},
        }],
    )

    result = tool_graph(DocumentState("Document", "Joanna"), [ai_msg])
    assert result["disposition"] == "pause"
    assert result["response_text"] == "Please answer this question."
    tool_message, assistant_message = result["messages"][-2:]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "Please answer this question."
    assert tool_message.tool_call_id == "call_1"
    assert assistant_message == AIMessage(content="Please answer this question.")


def test_stop_playback_takes_no_message_and_announces_how_to_resume(tool_graph):
    assert "message" not in stop_playback.args
    ai_msg = AIMessage(
        content="",
        tool_calls=[{"id": "call_1", "name": stop_playback.name, "args": {}}],
    )

    result = tool_graph(DocumentState("Document", "Joanna"), [ai_msg])
    message = "Playback stopped. Press Continue to resume."
    assert result["disposition"] == "stop"
    assert result["response_text"] == message
    tool_message, assistant_message = result["messages"][-2:]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == message
    assert tool_message.tool_call_id == "call_1"
    assert assistant_message == AIMessage(content=message)


def test_successive_edits_and_readers_use_current_draft(tool_graph):
    published = DocumentState("alpha beta", "Joanna")
    calls = [
        ("replace_text", {"before": "alpha ", "old_text": "beta", "new_text": "beta gamma", "after": ""}),
        ("replace_text", {"before": "beta ", "old_text": "gamma", "new_text": "delta", "after": ""}),
        ("read_document", {}),
        ("search_document", {"search_text": "delta"}),
        ("reposition_playback", {"before": "beta ", "target": "delta", "after": "", "message": "Resuming."}),
    ]
    responses = [AIMessage(content="", tool_calls=[{
        "id": f"call_{i}", "name": name, "args": args,
    }]) for i, (name, args) in enumerate(calls)]
    result = tool_graph(published, responses)
    messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert [message.tool_call_id for message in messages] == [f"call_{i}" for i in range(len(calls))]
    assert json.loads(messages[2].content) == {"position": 0, "text": "alpha beta delta"}
    assert json.loads(messages[3].content) == [{"position": 0, "text": "alpha beta delta"}]
    assert json.loads(messages[4].content) == {"position": 11, "text": "delta"}
    assert result["document_state"].document == "alpha beta delta"
    assert published.document == "alpha beta"
    assert result["position"] == 11
    assert result["response_text"] == "Resuming."


@pytest.mark.parametrize("content", [
    "Done. Resuming playback.",
    [{"type": "text", "text": "Done. Resuming playback."}],
])
def test_mixed_content_rejects_entire_tool_batch_before_any_edit(tool_graph, content):
    state = DocumentState("Hello world", "Joanna")
    calls = [
        {"id": "edit", "name": "replace_text", "args": {
            "before": "", "old_text": "Hello", "new_text": "Hi", "after": " world",
        }},
        {"id": "jump", "name": "reposition_playback", "args": {
            "before": "Hello ", "target": "world", "after": "",
        }},
    ]
    output = tool_graph(state, [AIMessage(content=content, tool_calls=calls), AIMessage(content="Retrying later.")])
    errors = [message for message in output["messages"] if isinstance(message, ToolMessage)]
    assert [message.tool_call_id for message in errors] == ["edit", "jump"]
    assert all(message.status == "error" and "No tools were executed" in message.content for message in errors)
    assert output["document_state"].document == "Hello world"
    assert output["position"] == 0
    assert output["response_text"] == "Retrying later."


def test_mixed_content_retry_executes_edit_once_and_puts_response_in_message(tool_graph):
    edit = {"id": "rejected", "name": "replace_text", "args": {
        "before": "", "old_text": "Hello", "new_text": "Hello again", "after": " world",
    }}
    output = tool_graph(DocumentState("Hello world", "Joanna"), [
        AIMessage(content="I edited it.", tool_calls=[edit]),
        AIMessage(content="", tool_calls=[{**edit, "id": "retry"}]),
        AIMessage(content="", tool_calls=[{"id": "jump", "name": "reposition_playback", "args": {
            "before": "", "target": "Hello again", "after": " world", "message": "Done. Resuming there.",
        }}]),
    ])
    messages = [message for message in output["messages"] if isinstance(message, ToolMessage)]
    assert messages[0].status == "error"
    assert output["document_state"].document == "Hello again world"
    assert output["response_text"] == "Done. Resuming there."


@pytest.mark.parametrize("content", [
    " \n ",
    [{"type": "reasoning", "reasoning": "Private reasoning"}],
    [{"type": "thinking", "thinking": "Private reasoning"}, {"type": "text", "text": "  "}],
])
def test_whitespace_and_reasoning_do_not_reject_tool_calls(tool_graph, content):
    output = tool_graph(DocumentState("Document", "Joanna"), [
        AIMessage(content=content, tool_calls=[{"id": "pause", "name": "pause_playback", "args": {"message": "Paused."}}]),
    ])
    assert output["disposition"] == "pause"
    assert output["response_text"] == "Paused."
