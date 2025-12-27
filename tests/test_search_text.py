from agentic_editor.document import DocumentState


class StubTtsCache:
    def __getitem__(self, text: str):
        raise AssertionError("search_text should not access TTS cache")


def _state(document: str) -> DocumentState:
    return DocumentState(
        document=document,
        voice_id="Joanna",
        tts_cache=StubTtsCache(),
    )


def test_search_text_unique_match():
    doc = "alpha before beta target gamma after omega"
    state = _state(doc)
    result = state.search_text("before", "target", "after")
    assert result.index == doc.index("target")
    assert result.candidates == []
    assert result.reason == "unique_match"


def test_search_text_ambiguous_matches():
    doc = "before target after ... before target after"
    state = _state(doc)
    result = state.search_text("before", "target", "after")
    assert result.index is None
    assert result.reason == "ambiguous"
    assert len(result.candidates) == 2
    assert all(candidate.target == "target" for candidate in result.candidates)
    for candidate in result.candidates:
        followup = state.search_text(
            candidate.before, candidate.target, candidate.after
        )
        expected_index = candidate.span_start + len(candidate.before)
        assert followup.index == expected_index


def test_search_text_target_exists_missing_context():
    doc = "intro target outro"
    state = _state(doc)
    result = state.search_text("missing", "target", "absent")
    assert result.index is None
    assert result.reason == "no_match"
    assert result.candidates
    assert result.candidates[0].target == "target"
    candidate = result.candidates[0]
    followup = state.search_text(candidate.before, candidate.target, candidate.after)
    expected_index = candidate.span_start + len(candidate.before)
    assert followup.index == expected_index


def test_search_text_target_missing_before_after_present():
    doc = "before middle after"
    state = _state(doc)
    result = state.search_text("before", "target", "after")
    assert result.index is None
    assert result.reason == "no_match"
    assert result.candidates
    candidate = result.candidates[0]
    assert candidate.before == "before"
    assert candidate.after == "after"
    assert "middle" in candidate.target
    followup = state.search_text(candidate.before, candidate.target, candidate.after)
    expected_index = candidate.span_start + len(candidate.before)
    assert followup.index == expected_index


def test_search_text_unique_match_with_radius():
    doc = (
        "before "
        + "x" * 10
        + "target after "
        + "x" * 500
        + "target after"
    )
    state = _state(doc)
    result = state.search_text("before", "target", "after")
    assert result.index == doc.index("target")
    assert result.candidates == []
    assert result.reason == "unique_match"


def test_search_text_long_before_after_within_radius():
    long_before = "before-" + ("x" * 120)
    long_after = ("y" * 120) + "-after"
    doc = f"{long_before}TARGET{long_after}"
    state = _state(doc)
    result = state.search_text(long_before, "TARGET", long_after)
    assert result.index == doc.index("TARGET")
    assert result.candidates == []
    assert result.reason == "unique_match"
