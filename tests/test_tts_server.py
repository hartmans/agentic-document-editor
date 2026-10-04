import asyncio
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

import backend
from agentic_editor.tts import build_document_state


@pytest.fixture(scope="session")
def sample_document(tmp_path_factory):
    doc_text = "Hello world.\n\nSecond line."
    path = tmp_path_factory.mktemp("docs") / "doc.txt"
    path.write_text(doc_text, encoding="utf-8")
    return path, doc_text


@pytest.fixture
def document_state(sample_document):
    _, doc_text = sample_document
    return build_document_state(doc_text)


def test_api_endpoints(document_state, sample_document):
    path, _doc_text = sample_document
    app = backend.create_app(document_state, path, config={"region_name": "us-east-1"})
    client = TestClient(app)

    status_resp = client.get("/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    document_id = status_data["documentId"]
    assert status_data["chunkCount"] == 1
    assert "chunk_id" not in status_data["document"]["chunks"][0]

    doc_resp = client.get(f"/document/{document_id}")
    assert doc_resp.status_code == 200
    doc = doc_resp.json()
    assert "chunk_id" not in doc["chunks"][0]

    assert client.post("/interruptions", json={}).status_code == 404

def test_historical_revision_audio(document_state, sample_document):
    path, _ = sample_document
    app = backend.create_app(document_state, path, config={})
    client = TestClient(app)
    original_id = client.get("/status").json()["documentId"]
    # Install another real document snapshot while keeping the original revision.
    updated = build_document_state(
        "A replacement document.",
    )
    app.state.documents["updated-revision"] = updated
    app.state.document_state = updated
    app.state.document_id = "updated-revision"

    assert client.get("/status").json()["documentId"] == "updated-revision"
    old_document = client.get(f"/document/{original_id}").json()
    assert old_document["document"] == document_state.document


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

        def transcribe_audio(self, audio):
            assert isinstance(audio, np.ndarray)
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


@pytest.fixture
def make_marks_app(tmp_path):
    def create(text):
        state = build_document_state(text)
        path = tmp_path / "document.txt"
        path.write_text(text, encoding="utf-8")
        return backend.create_app(state, path, config={}), state
    return create


@pytest.mark.parametrize(("document", "chunk_id", "status", "detail"), [
    ("missing", "0", 404, "Unknown document_id"),
    ("current", "-1", 404, "Unknown chunk_id"),
    ("current", "1", 404, "Unknown chunk_id"),
    ("current", "999999", 404, "Unknown chunk_id"),
    ("current", "not-a-number", 422, None),
])
@pytest.mark.parametrize("endpoint", ["audio", "marks"])
def test_invalid_chunk_request_does_not_submit_audio(make_marks_app, document, chunk_id, status, detail, endpoint):
    app, state = make_marks_app("An unrendered document.")
    document_id = app.state.document_id if document == "current" else document
    response = TestClient(app).get(f"/chunks/{document_id}/{chunk_id}/{endpoint}")
    assert response.status_code == status
    if detail is not None:
        assert response.json() == {"detail": detail}
    assert state.speech is None


def test_empty_document_has_no_chunk_marks(make_marks_app):
    app, state = make_marks_app("")
    response = TestClient(app).get(f"/chunks/{app.state.document_id}/0/marks")
    assert response.status_code == 200
    assert response.json()["marks"] == []
    assert backend._build_document_response(state).chunks[0].has_audio is False
    assert state.speech is None


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
    assert state.speech is None
