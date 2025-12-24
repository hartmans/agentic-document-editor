from types import SimpleNamespace

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from agentic_editor.document import DocumentState
from agentic_editor.graph import DocumentUpdate
from agentic_editor.tools import replace_text


class StubTtsCache:
    def __getitem__(self, text: str):
        raise AssertionError("replace_text should not access TTS cache")


def test_replace_text_toolnode_executes():
    doc_state = DocumentState(
        document="Hello world",
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
                "args": {"offset": 0, "old_text": "Hello", "new_text": "Hi"},
            }
        ],
    )

    runtime = SimpleNamespace(context=None, store=None, stream_writer=None)

    result = tool_node.invoke(
        {"messages": [ai_msg], "document": doc_state},
        runtime=runtime,
    )

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], Command)
    update = result[0].update
    # ToolNode should have appended a ToolMessage and returned a document update.
    assert isinstance(update["messages"][-1], ToolMessage)
    assert update["messages"][-1].tool_call_id == "call_1"
    assert update["document"] == "Hi world"
    assert doc_state.document == "Hello world"
