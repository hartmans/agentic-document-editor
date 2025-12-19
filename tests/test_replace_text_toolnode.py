from types import SimpleNamespace

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from agentic_editor.document import DocumentState
from agentic_editor.tools import replace_text


def test_replace_text_toolnode_executes():
    doc_state = DocumentState(document="Hello world", voice_id="Joanna")
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

    # ToolNode should have appended a ToolMessage and updated state.document
    assert isinstance(result["messages"][-1], ToolMessage)
    assert doc_state.document == "Hi world"
