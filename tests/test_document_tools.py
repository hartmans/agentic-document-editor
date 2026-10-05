import json

from langchain_core.messages import AIMessage, ToolMessage
from agentic_editor.document import DocumentState
from agentic_editor.tools import (
    DEFAULT_EXCERPT_LENGTH,
    document_excerpt,
    read_document,
    search_document,
)


def test_document_excerpt_bounds_position_and_length():
    assert document_excerpt("abcdef", 2, 3) == {"position": 2, "text": "cde"}
    assert document_excerpt("abcdef", 99) == {"position": 6, "text": ""}


def test_read_document_defaults_to_start_and_250_characters(tool_graph):
    document = "0123456789" * 30

    output = tool_graph(DocumentState(document, "Joanna"), [
        AIMessage(content="", tool_calls=[{"id": "read", "name": "read_document", "args": {}}]),
        AIMessage(content="Done."),
    ])
    message = output["messages"][-2]
    assert isinstance(message, ToolMessage)
    result = json.loads(message.content)

    assert result == {"position": 0, "text": document[:DEFAULT_EXCERPT_LENGTH]}
    assert "current document" in read_document.description.lower()


def test_search_document_defaults_and_centers_hits(tool_graph):
    document = ("a" * 200) + "needle" + ("b" * 300) + "needle" + ("c" * 200)

    output = tool_graph(DocumentState(document, "Joanna"), [
        AIMessage(content="", tool_calls=[{
            "id": "search", "name": "search_document", "args": {"search_text": "needle"},
        }]),
        AIMessage(content="Done."),
    ])
    results = json.loads(output["messages"][-2].content)

    assert len(results) == 2
    for result in results:
        assert len(result["text"]) == DEFAULT_EXCERPT_LENGTH
        assert "needle" in result["text"]
        relative_hit = result["text"].index("needle")
        assert abs(relative_hit - (DEFAULT_EXCERPT_LENGTH - len("needle")) // 2) <= 1
    assert "current document" in search_document.description.lower()


def test_search_document_honors_optional_limits(tool_graph):
    document = "hit xx hit yy hit"

    output = tool_graph(DocumentState(document, "Joanna"), [
        AIMessage(content="", tool_calls=[{
            "id": "search", "name": "search_document", "args": {
                "search_text": "hit", "length": 6, "starting_position": 4, "max_hits": 1,
            },
        }]),
        AIMessage(content="Done."),
    ])
    results = json.loads(output["messages"][-2].content)

    assert len(results) == 1
    assert results[0]["position"] <= document.index("hit", 4)
    assert "hit" in results[0]["text"]
