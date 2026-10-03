from agentic_editor.text_alignment import match_filtered_words, project_word_marks
from agentic_editor.document import chunk_documents
from agentic_editor.speech import document_cache_dir


def test_filtered_markup_and_repeated_words_map_to_original_spans():
    original = "🌟 **Hello** world. Hello `world`."
    spoken = "Hello world. Hello world."
    matches = match_filtered_words(original, spoken)
    assert [original[m.original_start:m.original_end] for m in matches] == [
        "Hello", "world", "Hello", "world",
    ]
    assert [m.original_start for m in matches] == sorted(m.original_start for m in matches)
    assert [spoken[m.spoken_start:m.spoken_end] for m in matches] == [
        "Hello", "world", "Hello", "world",
    ]


def test_deleted_text_and_empty_filtered_text():
    original = "before markup after"
    matches = match_filtered_words(original, "before after")
    assert [original[m.original_start:m.original_end] for m in matches] == ["before", "after"]
    assert match_filtered_words(original, "") == []


def test_sparse_timing_projection_does_not_invent_coverage():
    original = "**Hello** world again."
    spoken = "Hello world again."
    words = match_filtered_words(original, spoken)
    marks = project_word_marks(words, {0: 0.125, 6: None, 12: 1.250})
    assert [mark.time_ms for mark in marks] == [125, 1250]
    assert [original[mark.chunk_char_start:mark.chunk_char_end] for mark in marks] == ["Hello", "again"]
    assert project_word_marks(words, {0: float("nan"), 6: float("inf")}) == []


def test_expanded_word_keeps_a_best_effort_document_anchor():
    original = "Dr. Jones said hello."
    spoken = "Doctor Jones said hello."
    matches = match_filtered_words(original, spoken)
    assert [original[m.original_start:m.original_end] for m in matches] == ["Dr", "Jones", "said", "hello"]


def test_cache_location_is_stable_and_disambiguates_names(tmp_path):
    base = tmp_path / "cache"
    first = tmp_path / "one" / "Chapter 1.rst"
    second = tmp_path / "two" / "Chapter 1.rst"
    directory = document_cache_dir(base, first)
    assert directory.parent == base
    assert directory.name.startswith("Chapter-1-")
    assert directory == document_cache_dir(base, first.parent / "." / first.name)
    assert directory != document_cache_dir(base, second)
    assert not directory.exists()


def test_long_documents_without_paragraphs_split_and_keep_all_text():
    for text in ("word " * 1500, "x" * 7500):
        spans = chunk_documents(text)
        assert "".join(text[a:b] for a, b in spans) == text
        assert spans[0][0] == 0
        assert spans[-1][1] == len(text)
        assert all(a < b and b - a <= 2400 for a, b in spans)
