import pytest
from fastapi.testclient import TestClient

import tts_server


@pytest.fixture(scope="session")
def sample_document(tmp_path_factory):
    doc_text = "Hello world.\n\nSecond line."
    path = tmp_path_factory.mktemp("docs") / "doc.txt"
    path.write_text(doc_text, encoding="utf-8")
    return path, doc_text


@pytest.fixture(scope="session")
def document_state(sample_document):
    path, doc_text = sample_document
    return tts_server.build_document_state(
        document_text=doc_text,
        voice_id="Joanna",
        chunking_strategy="paragraph",
    )


def test_chunking_and_mapping(document_state):
    # Expect two chunks split by blank lines.
    assert len(document_state.chunks) == 2
    first = document_state.chunks[0]
    mapping = tts_server.map_elapsed_time_to_document(first, elapsed_ms=0)
    assert mapping["chunk_id"] == 0
    assert mapping["word_text"].lower().startswith("hello")
    assert mapping["doc_char_start"] == first.doc_char_start
    assert mapping["doc_char_end"] > mapping["doc_char_start"]


def test_api_endpoints(document_state):
    app = tts_server.create_app(document_state)
    client = TestClient(app)

    status_resp = client.get("/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["chunk_count"] == 2

    chunks_resp = client.get("/chunks")
    assert chunks_resp.status_code == 200
    chunks = chunks_resp.json()
    assert len(chunks) == 2
    assert chunks[0]["chunk_id"] == 0

    detail_resp = client.get("/chunks/0")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["chunk_id"] == 0
    assert detail["speech_marks"]

    audio_resp = client.get("/chunks/0/audio")
    assert audio_resp.status_code == 200
    assert audio_resp.content  # audio bytes present

    interrupt_resp = client.post("/interruptions", json={"chunk_id": 0, "elapsed_ms": 0})
    assert interrupt_resp.status_code == 200
    interrupt = interrupt_resp.json()
    assert interrupt["chunk_id"] == 0
    assert interrupt["doc_char_start"] >= 0
