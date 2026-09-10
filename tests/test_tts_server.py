import boto3
import asyncio
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

import backend
from agentic_editor.tts import build_document_state, map_elapsed_time_to_document


@pytest.fixture(scope="session")
def sample_document(tmp_path_factory):
    doc_text = "Hello world.\n\nSecond line."
    path = tmp_path_factory.mktemp("docs") / "doc.txt"
    path.write_text(doc_text, encoding="utf-8")
    return path, doc_text


@pytest.fixture(scope="session")
def document_state(sample_document, tmp_path_factory):
    _path, doc_text = sample_document
    cache_dir = tmp_path_factory.mktemp("cache")
    session = boto3.Session()
    return build_document_state(
        document_text=doc_text,
        session=session,
        config={
            "voice_id": "Joanna",
            "engine": "standard",
            "region_name": "us-east-1",
            "cache_dir": str(cache_dir),
        },
    )


def test_chunking_and_mapping(document_state):
    assert len(document_state.chunks) == 1
    first = document_state.chunks[0]
    mapping = map_elapsed_time_to_document(
        document_state.tts_cache,
        first,
        chunk_id=0,
        elapsed_ms=0,
    )
    assert mapping["chunkId"] == 0
    assert mapping["docCharPosition"] == first.doc_char_start


def test_api_endpoints(document_state, sample_document):
    path, _doc_text = sample_document
    app = backend.create_app(document_state, path, config={"region_name": "us-east-1"})
    client = TestClient(app)

    status_resp = client.get("/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    document_id = status_data["documentId"]
    assert status_data["chunkCount"] == 1
    assert status_data["document"]["chunks"][0]["chunk_id"] == 0

    doc_resp = client.get(f"/document/{document_id}")
    assert doc_resp.status_code == 200
    doc = doc_resp.json()
    assert doc["chunks"][0]["chunk_id"] == 0

    audio_resp = client.get(f"/chunks/{document_id}/0/audio")
    assert audio_resp.status_code == 200
    assert audio_resp.content  # audio bytes present

    interrupt_resp = client.post("/interruptions", json={
        "documentId": document_id, "chunk_id": 0, "elapsed_ms": 0,
    })
    assert interrupt_resp.status_code == 200
    interrupt = interrupt_resp.json()
    assert interrupt["chunkId"] == 0
    assert interrupt["docCharPosition"] >= 0
    assert interrupt["documentId"] == document_id


def test_historical_revision_audio_and_interruption(document_state, sample_document):
    path, _ = sample_document
    app = backend.create_app(document_state, path, config={})
    client = TestClient(app)
    original_id = client.get("/status").json()["documentId"]
    # Install another real document snapshot while keeping the original revision.
    updated = build_document_state(
        "A replacement document.", tts_cache=document_state.tts_cache,
    )
    app.state.documents["updated-revision"] = updated
    app.state.document_state = updated
    app.state.document_id = "updated-revision"

    assert client.get("/status").json()["documentId"] == "updated-revision"
    old_document = client.get(f"/document/{original_id}").json()
    assert old_document["document"] == document_state.document
    audio = client.get(f"/chunks/{original_id}/0/audio")
    assert audio.status_code == 200
    assert audio.content == document_state.tts_cache[document_state.chunks[0].text][0]
    point = client.post("/interruptions", json={
        "documentId": original_id, "chunk_id": 0, "elapsed_ms": 1000,
    })
    expected = map_elapsed_time_to_document(
        document_state.tts_cache, document_state.chunks[0], 0, 1000,
    )
    assert point.status_code == 200
    assert point.json() == {**expected, "documentId": original_id}


def test_instructions_recheck_revision_after_waiting(document_state, sample_document):
    path, _ = sample_document
    app = backend.create_app(document_state, path, config={})
    original_id = app.state.document_id

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            # Represent an edit currently holding the lock while awaiting its
            # model. A second request must wait, then check the published revision.
            async with app.state.instructions_lock:
                request = asyncio.create_task(client.post("/instructions", json={
                    "baseDocumentId": original_id,
                    "instructions": "Continue reading",
                    "resumePoint": {
                        "documentId": original_id,
                        "docCharPosition": 0,
                    },
                }))
                done, _ = await asyncio.wait({request}, timeout=0.05)
                assert not done
                app.state.document_id = "newer-revision"
            response = await asyncio.wait_for(request, timeout=5)
            assert response.status_code == 409
            assert app.state.document_id == "newer-revision"

    asyncio.run(exercise())


def test_instructions_reject_resume_from_another_revision(document_state, sample_document):
    path, _ = sample_document
    app = backend.create_app(document_state, path, config={})
    client = TestClient(app)
    response = client.post("/instructions", json={
        "baseDocumentId": app.state.document_id,
        "instructions": "Continue reading",
        "resumePoint": {
            "documentId": "old-revision",
            "docCharPosition": 0,
        },
    })
    assert response.status_code == 409


def test_asr_endpoint_aws(document_state, sample_document):
    path, _doc_text = sample_document
    fake_creds = {
        "AccessKeyId": "AKIAFAKE",
        "SecretAccessKey": "SECRET",
        "SessionToken": "TOKEN",
        "Expiration": "2024-01-01T00:00:00Z",
    }

    class FakeStsClient:
        def get_federation_token(self, **kwargs):
            return {"Credentials": fake_creds}

    class FakeSession:
        def client(self, name, region_name=None):
            assert name == "sts"
            return FakeStsClient()

    document_state.aws_session = FakeSession()
    document_state.region_name = "us-east-1"
    app = backend.create_app(document_state, path, config={"region_name": "us-east-1"})
    client = TestClient(app)

    resp = client.get("/asr")
    assert resp.status_code == 200
    data = resp.json()
    assert data["type"] == "aws"
    assert data["region"] == "us-east-1"
    assert data["stsToken"]["access_key_id"] == fake_creds["AccessKeyId"]
    assert data["stsToken"]["secret_access_key"] == fake_creds["SecretAccessKey"]
    assert data["stsToken"]["session_token"] == fake_creds["SessionToken"]


def test_asr_endpoint_gemma4(document_state, sample_document):
    path, _doc_text = sample_document
    app = backend.create_app(document_state, path, config={})
    client = TestClient(app)

    resp = client.get("/asr")
    assert resp.status_code == 200
    assert resp.json() == {"type": "gemma4"}


def test_asr_websocket_returns_transcription(document_state, sample_document):
    path, _doc_text = sample_document

    class FakeAsrReceiver:
        model_name = "google/gemma-4-E4B-it"

        def create_buffer(self):
            from agentic_editor.gemma4 import AudioWebSocketBuffer

            return AudioWebSocketBuffer()

        def transcribe_audio(self, audio, *, document: str):
            assert isinstance(audio, np.ndarray)
            assert "Hello world." in document
            return "replace hello with hi"

    app = backend.create_app(document_state, path, config={}, asr_receiver=FakeAsrReceiver())
    client = TestClient(app)

    with client.websocket_connect("/ws/asr") as websocket:
        websocket.send_json({"type": "start"})
        started = websocket.receive_json()
        assert started["type"] == "started"
        websocket.send_bytes((np.zeros(4000, dtype=np.int16)).tobytes())
        websocket.send_json({"type": "end"})
        result = websocket.receive_json()
        assert result["type"] == "result"
        assert result["instructions"] == "replace hello with hi"


def test_create_app_stores_gemma4_debug_dir(document_state, sample_document, tmp_path):
    path, _doc_text = sample_document

    app = backend.create_app(
        document_state,
        path,
        config={},
        gemma4_debug_audio_dir=tmp_path / "logs",
    )

    assert app.state.gemma4_debug_audio_dir == tmp_path / "logs"


def test_chunk_marks_are_revision_scoped_document_positions(document_state, sample_document):
    from agentic_editor.document import DocumentState, TextChunk

    path, _ = sample_document
    app = backend.create_app(document_state, path, config={})
    chunk = document_state.chunks[0]
    # Unicode offsets count code points, not UTF-8 bytes or UTF-16 units.
    prefix = "🌟 Préface\n\n"
    offset = len(prefix)
    shifted = DocumentState(
        document=prefix + chunk.text,
        voice_id=document_state.voice_id,
        tts_cache=document_state.tts_cache,
        chunks=[
            TextChunk(prefix, 0, offset),
            TextChunk(chunk.text, offset, offset + len(chunk.text)),
        ],
    )
    app.state.documents["shifted"] = shifted
    original_id = app.state.document_id
    app.state.document_id = "shifted"
    app.state.document_state = shifted
    client = TestClient(app)
    response = client.get("/chunks/shifted/1/marks")
    assert response.status_code == 200
    _, marks = document_state.tts_cache[chunk.text]
    assert response.json() == {
        "documentId": "shifted", "chunkId": 1,
        "marks": [{
            "timeMs": mark.time_ms,
            "docCharStart": offset + mark.chunk_char_start,
            "docCharEnd": offset + mark.chunk_char_end,
        } for mark in marks],
    }
    assert marks
    historical = client.get(f"/chunks/{original_id}/0/marks")
    assert historical.status_code == 200
    assert historical.json() == {
        "documentId": original_id, "chunkId": 0,
        "marks": [{
            "timeMs": mark.time_ms,
            "docCharStart": mark.chunk_char_start,
            "docCharEnd": mark.chunk_char_end,
        } for mark in marks],
    }
    for mark in response.json()["marks"]:
        assert shifted.document[mark["docCharStart"]:mark["docCharEnd"]]



def test_instructions_return_before_resume_audio_finishes(document_state, tmp_path, monkeypatch):
    from threading import Event
    from uuid import uuid4

    path = tmp_path / "document.txt"
    path.write_text(document_state.document)
    app = backend.create_app(document_state, path, config={})
    release = Event()
    started = Event()
    impl = document_state.tts_cache._impl
    synthesize = impl._synthesize_entry

    def delayed_synthesis(*args):
        started.set()
        if not release.wait(timeout=90):
            raise TimeoutError("Test did not release synthesis")
        return synthesize(*args)

    # Delay actual synthesis, then let the real engine run after HTTP returns.
    monkeypatch.setattr(impl, "_synthesize_entry", delayed_synthesis)
    unique_text = f"The replacement document has identifier {uuid4().hex}."

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            response = await asyncio.wait_for(client.post("/instructions", json={
                "baseDocumentId": app.state.document_id,
                "instructions": f"Replace the entire document with this exact text: {unique_text}",
                "resumePoint": {
                    "documentId": app.state.document_id, "docCharPosition": 0,
                },
            }), timeout=60)
            assert response.status_code == 200, response.text
            result = response.json()
            assert unique_text in result["document"]["document"]
            assert result["documentId"] != document_state_id
            assert result["resumePoint"] == {
                "documentId": result["documentId"], "docCharPosition": 0,
            }
            return result

    document_state_id = app.state.document_id
    try:
        result = asyncio.run(exercise())
        assert started.wait(timeout=5)
        assert not release.is_set()
    finally:
        release.set()
    # Finish real synthesis and verify the new endpoint exposes its marks.
    client = TestClient(app)
    assert client.get(f'/chunks/{result["documentId"]}/0/marks').status_code == 200


@pytest.fixture
def make_marks_app(tmp_path):
    """Use a fresh real Polly cache, initially without any submitted synthesis."""
    def create(text):
        state = build_document_state(
            text,
            config={
                "voice_id": "Joanna", "engine": "standard",
                "region_name": "us-east-1", "cache_dir": str(tmp_path / "cache"),
            },
            submit_audio=False,
        )
        path = tmp_path / "document.txt"
        path.write_text(text, encoding="utf-8")
        app = backend.create_app(state, path, config={})
        return app, state

    return create


def test_marks_request_synthesizes_uncached_chunk_and_audio_reuses_it(make_marks_app):
    app, state = make_marks_app("Hello from the chunk marks endpoint.")
    client = TestClient(app)
    document_id = app.state.document_id
    impl = state.tts_cache._impl
    assert impl._cache == {}
    assert not list(impl.cache_dir.iterdir())

    response = client.get(f"/chunks/{document_id}/0/marks")
    assert response.status_code == 200
    marks = response.json()["marks"]
    assert marks
    assert [mark["timeMs"] for mark in marks] == sorted(mark["timeMs"] for mark in marks)
    assert [state.document[m["docCharStart"]:m["docCharEnd"]] for m in marks] == [
        "Hello", "from", "the", "chunk", "marks", "endpoint",
    ]
    files = {path: path.stat().st_mtime_ns for path in impl.cache_dir.iterdir()}
    assert len(files) == 2
    audio = client.get(f"/chunks/{document_id}/0/audio")
    assert audio.status_code == 200
    assert audio.headers["content-type"] == "audio/mpeg"
    assert audio.content == state.tts_cache[state.chunks[0].text][0]
    assert client.get(f"/chunks/{document_id}/0/marks").json() == response.json()
    assert {path: path.stat().st_mtime_ns for path in impl.cache_dir.iterdir()} == files

    # A fresh cache instance must serve the same API from disk as well.
    reloaded, reloaded_state = make_marks_app(state.document)
    assert reloaded_state.tts_cache._impl._cache == {}
    reloaded_response = TestClient(reloaded).get(f"/chunks/{reloaded.state.document_id}/0/marks")
    assert reloaded_response.status_code == 200
    assert reloaded_response.json()["marks"] == marks
    assert {path: path.stat().st_mtime_ns for path in impl.cache_dir.iterdir()} == files


def test_chunk_marks_can_be_empty(make_marks_app):
    # Keep real synthesized audio, but supply a persisted empty alignment. This
    # exercises cache loading without depending on a provider's silence marks.
    _, state = make_marks_app("Audio without word alignment.")
    audio_bytes, _ = state.tts_cache[state.document]
    impl = state.tts_cache._impl
    impl._write_speech_marks(impl._marks_path(impl._key(state.document)), [])
    app, _ = make_marks_app(state.document)
    client = TestClient(app)
    document_id = app.state.document_id
    response = client.get(f"/chunks/{document_id}/0/marks")
    assert response.status_code == 200
    assert response.json() == {"documentId": document_id, "chunkId": 0, "marks": []}
    audio = client.get(f"/chunks/{document_id}/0/audio")
    assert audio.status_code == 200
    assert audio.content == audio_bytes


@pytest.mark.parametrize(("document", "chunk_id", "status", "detail"), [
    ("missing", "0", 404, "Unknown document_id"),
    ("current", "-1", 404, "Unknown chunk_id"),
    ("current", "1", 404, "Unknown chunk_id"),
    ("current", "999999", 404, "Unknown chunk_id"),
    ("current", "not-a-number", 422, None),
])
def test_invalid_marks_request_does_not_submit_audio(make_marks_app, document, chunk_id, status, detail):
    app, state = make_marks_app("An unrendered document.")
    document_id = app.state.document_id if document == "current" else document
    response = TestClient(app).get(f"/chunks/{document_id}/{chunk_id}/marks")
    assert response.status_code == status
    if detail is not None:
        assert response.json() == {"detail": detail}
    assert state.tts_cache._impl._cache == {}
    assert not list(state.tts_cache._impl.cache_dir.iterdir())


def test_empty_document_has_no_chunk_marks(make_marks_app):
    app, state = make_marks_app("")
    response = TestClient(app).get(f"/chunks/{app.state.document_id}/0/marks")
    assert response.status_code == 404
    assert response.json() == {"detail": "Unknown chunk_id"}
    assert state.tts_cache._impl._cache == {}


def test_marks_synthesis_failure_returns_http_error(make_marks_app):
    # Exercise a real service error: this is invalid SSML, not a mocked engine.
    app, _ = make_marks_app("<unclosed>")
    response = TestClient(app).get(f"/chunks/{app.state.document_id}/0/marks")
    assert response.status_code == 500
    assert response.json() == {"detail": "Failed to synthesize audio"}


@pytest.mark.parametrize("point", [{}, {"docCharPosition": -1}, {"docCharPosition": None},
                                   {"docCharPosition": "invalid"}])
def test_instructions_validate_character_resume_position(make_marks_app, point):
    app, state = make_marks_app("An unrendered document.")
    response = TestClient(app).post("/instructions", json={
        "baseDocumentId": app.state.document_id,
        "instructions": "Continue reading",
        "resumePoint": {"documentId": app.state.document_id, **point},
    })
    assert response.status_code == 422
    assert any(error["loc"] == ["body", "resumePoint", "docCharPosition"]
               for error in response.json()["detail"])
    assert state.tts_cache._impl._cache == {}


@pytest.mark.parametrize("position", [14, 1000])
def test_no_edit_instructions_preserve_character_position_and_revision(make_marks_app, position):
    app, state = make_marks_app("Hello world. Second line.")
    original_id = app.state.document_id
    response = TestClient(app).post("/instructions", json={
        "baseDocumentId": original_id,
        "instructions": "Do not edit the document or reposition playback. Just say Ready.",
        "resumePoint": {"documentId": original_id, "docCharPosition": position},
    })
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["documentId"] == original_id
    assert result["document"]["document"] == state.document
    assert result["resumePoint"] == {
        "documentId": original_id,
        "docCharPosition": min(position, len(state.document)),
    }
    # The spoken response is addressable through the same marks API.
    client = TestClient(app)
    for chunk in result["responseDocument"]["chunks"]:
        marks = client.get(f'/chunks/{result["responseDocumentId"]}/{chunk["chunk_id"]}/marks')
        assert marks.status_code == 200
        assert marks.json()["documentId"] == result["responseDocumentId"]
        assert marks.json()["chunkId"] == chunk["chunk_id"]
