from agentic_editor.document import apply_document_update
from agentic_editor.tts import build_document_state


def test_apply_document_update_rebuilds_text_metadata_without_audio():
    state = build_document_state("Hello world")
    state.aws_session = object()
    new_state = apply_document_update("Hello there", state)
    assert state.document == "Hello world"
    assert new_state.document == "Hello there"
    assert new_state.speech is state.speech
    assert new_state.aws_session is state.aws_session
    assert len(new_state.chunks) == 1
    assert new_state.chunks[0].text == "Hello there"
    assert new_state.chunks[0].plan is None
