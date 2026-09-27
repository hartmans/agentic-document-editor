import pytest

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.types import Command

from agentic_editor.document import DocumentState
from agentic_editor.graph import DocumentUpdate
from agentic_editor.tools import replace_text


class StubTtsCache:
    def __getitem__(self, text: str):
        raise AssertionError("replace_text should not access TTS cache")


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
        tts_cache=StubTtsCache(),
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
    assert doc_state.document == document
