from agentic_editor.tts import build_document_state


def test_copy_and_update_preserve_document_services():
    state = build_document_state("Hello world")
    state.aws_session = object()
    new_state = state.copy()
    new_state.update("there", 6, 11)
    assert state.document == "Hello world"
    assert new_state.document == "Hello there"
    assert new_state.speech is state.speech
    assert new_state.aws_session is state.aws_session
    assert len(new_state.chunks) == 1
    assert new_state.chunks[0].text == "Hello there"
    assert new_state.chunks[0].plan is None


from agentic_editor.document import DocumentState, TextChunk
import pytest


def split_state(*texts):
    return DocumentState("".join(texts), "", chunks=[TextChunk(t) for t in texts])


def test_copy_on_write_preserves_resources_and_revision_offsets():
    old = split_state("abc", "def", "ghi")
    plan = object()
    old.chunks[2].plan = plan
    draft = old.copy()
    assert all(a is b for a, b in zip(old.chunks, draft.chunks))
    draft.update("longer", 0, 3)
    edited = draft.chunks[0]
    draft.update("X", 0, 1)
    assert draft.chunks[0] is edited
    assert old.document == "abcdefghi"
    assert draft.document == "Xongerdefghi"
    assert draft.chunks[2] is old.chunks[2]
    assert draft.chunks[2].plan is plan
    assert old.chunk_start(2) == 6
    assert draft.chunk_start(2) == 9
    old.update("Y", 3, 4)
    assert draft.chunks[1].text == "def"
    assert old.chunks[1] is not draft.chunks[1]


def test_cross_chunk_update_and_empty_slots():
    old = split_state("abc", "def", "ghi", "jkl")
    draft = old.copy()
    draft.update("X", 1, 8)
    assert [c.text for c in draft.chunks] == ["aX", "", "i", "jkl"]
    assert draft.document == "aXijkl"
    assert draft.find_chunk_id(2) == 2
    draft.update("Q", 2, 2)
    assert [c.text for c in draft.chunks] == ["aX", "", "Qi", "jkl"]
    draft.update("", 0, len(draft.document))
    assert len(draft.chunks) == 4
    assert all(c.text == "" for c in draft.chunks)
    draft.update("new", 0, 0)
    assert draft.document == "new"
    assert draft.chunks[-1].text == "new"
    assert old.document == "abcdefghijkl"


def test_noop_and_empty_document_insertion():
    old = split_state("abc", "def")
    draft = old.copy()
    draft.update("cd", 2, 4)
    assert draft.chunks == old.chunks
    assert all(a is b for a, b in zip(old.chunks, draft.chunks))
    blank = build_document_state("")
    assert len(blank.chunks) == 1
    blank.update("hello", 0, 0)
    assert blank.chunks[0].text == "hello"
    with pytest.raises(ValueError):
        blank.update("", 3, 2)


def test_copy_of_edited_draft_detaches_on_next_write():
    original = split_state("abc", "def")
    draft = original.copy()
    draft.update("X", 0, 1)
    fork = draft.copy()
    draft.update("Y", 1, 2)
    fork.update("Z", 2, 3)
    assert original.document == "abcdef"
    assert draft.document == "XYcdef"
    assert fork.document == "XbZdef"


import asyncio
from agentic_editor.tts import prepare_document_round


def test_successive_edit_rounds_introduce_only_local_resume_boundaries():
    original = split_state("Opening. Middle. Ending.", " Unaffected.")
    async def exercise():
        draft = original.copy()
        draft.update("New", 9, 15)
        first = await prepare_document_round(original, draft, 9)
        assert first.chunk_start(first.find_chunk_id(9)) == 9
        assert [c.text for c in first.chunks] == ["Opening. ", "New. Ending.", " Unaffected."]
        assert first.chunks[-1] is original.chunks[-1]
        second_draft = first.copy()
        pos = second_draft.document.index("Ending")
        second_draft.update("Final", pos, pos + len("Ending"))
        second = await prepare_document_round(first, second_draft, pos)
        assert second.chunk_start(second.find_chunk_id(pos)) == pos
        assert [c.text for c in second.chunks] == ["Opening. ", "New. ", "Final.", " Unaffected."]
        assert second.chunks[0] is first.chunks[0]
        assert second.chunks[-1] is original.chunks[-1]
        assert first.document == "Opening. New. Ending. Unaffected."
        assert original.document == "Opening. Middle. Ending. Unaffected."
        # Ownership indices must move with the inserted chunk boundary.
        second.update("!", len(second.document) - 1, len(second.document))
        assert original.chunks[-1].text == " Unaffected."
    asyncio.run(exercise())


def test_reposition_and_complete_undo_keep_original_state_and_chunks():
    original = split_state("Opening. ", "Ending.")
    async def exercise():
        assert await prepare_document_round(original, original.copy(), 3) is original
        draft = original.copy()
        draft.update("Different.", 0, len(draft.document))
        draft.update(original.document, 0, len(draft.document))
        assert await prepare_document_round(original, draft, 3) is original
    asyncio.run(exercise())


def test_boundary_splits_are_noops_and_splitting_a_shared_chunk_is_isolated():
    original = split_state("abc", "def")
    draft = original.copy()
    assert not draft.split_chunk(0)
    assert not draft.split_chunk(3)
    assert not draft.split_chunk(6)
    assert draft.split_chunk(4)
    assert [c.text for c in draft.chunks] == ["abc", "d", "ef"]
    assert draft.chunks[0] is original.chunks[0]
    assert [c.text for c in original.chunks] == ["abc", "def"]
    fork = draft.copy()
    draft.update("X", 4, 5)
    assert fork.document == "abcdef"
    assert draft.document == "abcdXf"
