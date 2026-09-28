import json

from agentic_editor.graph import TOOLS
from agentic_editor.tools import (
    DEFAULT_EXCERPT_LENGTH,
    document_excerpt,
    read_document,
    search_document,
)


def test_document_excerpt_bounds_position_and_length():
    assert document_excerpt("abcdef", 2, 3) == {"position": 2, "text": "cde"}
    assert document_excerpt("abcdef", 99) == {"position": 6, "text": ""}


def test_read_document_defaults_to_start_and_250_characters():
    document = "0123456789" * 30

    result = json.loads(read_document.func(document=document))

    assert result == {"position": 0, "text": document[:DEFAULT_EXCERPT_LENGTH]}
    assert "current document" in read_document.description.lower()


def test_search_document_defaults_and_centers_hits():
    document = ("a" * 200) + "needle" + ("b" * 300) + "needle" + ("c" * 200)

    results = json.loads(search_document.func(search_text="needle", document=document))

    assert len(results) == 2
    for result in results:
        assert len(result["text"]) == DEFAULT_EXCERPT_LENGTH
        assert "needle" in result["text"]
        relative_hit = result["text"].index("needle")
        assert abs(relative_hit - (DEFAULT_EXCERPT_LENGTH - len("needle")) // 2) <= 1
    assert "current document" in search_document.description.lower()


def test_search_document_honors_optional_limits():
    document = "hit xx hit yy hit"

    results = json.loads(search_document.func(
        search_text="hit",
        length=6,
        starting_position=4,
        max_hits=1,
        document=document,
    ))

    assert len(results) == 1
    assert results[0]["position"] <= document.index("hit", 4)
    assert "hit" in results[0]["text"]


def test_graph_enables_read_and_search_tools():
    assert read_document in TOOLS
    assert search_document in TOOLS
