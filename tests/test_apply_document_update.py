from agentic_editor.document import DocumentUpdate, SpeechMark, apply_document_update
from agentic_editor.tts import build_document_state


class StubTtsCache:
    def __init__(self):
        self.requests = []

    def __getitem__(self, text: str):
        self.requests.append(text)
        return b"audio", [SpeechMark(time_ms=0, chunk_char_start=0, chunk_char_end=0)]


def test_apply_document_update_rebuilds_state():
    cache = StubTtsCache()
    session = object()
    state = build_document_state(
        document_text="Hello world",
        voice_id="Joanna",
        region_name="us-east-1",
        tts_cache=cache,
        session=session,
    )
    update = DocumentUpdate(offset=6, length=5, new_text="there")

    new_state = apply_document_update(update, state)

    assert new_state.document == "Hello there"
    assert new_state.voice_id == "Joanna"
    assert new_state.tts_cache is cache
    assert new_state.aws_session is session
    assert len(new_state.chunks) == 1
    assert new_state.chunks[0].text == "Hello there"
    assert cache.requests[-1] == "Hello there"
