import boto3
import hashlib
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


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_api_endpoints(document_state, sample_document):
    path, _doc_text = sample_document
    document_id = _sha256_text(str(path.resolve()))
    app = backend.create_app(document_state, path, config={"region_name": "us-east-1"})
    client = TestClient(app)

    status_resp = client.get("/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["chunkCount"] == 1
    assert status_data["document"]["chunks"][0]["chunk_id"] == 0

    doc_resp = client.get(f"/document/{document_id}")
    assert doc_resp.status_code == 200
    doc = doc_resp.json()
    assert doc["chunks"][0]["chunk_id"] == 0

    audio_resp = client.get(f"/chunks/{document_id}/0/audio")
    assert audio_resp.status_code == 200
    assert audio_resp.content  # audio bytes present

    interrupt_resp = client.post("/interruptions", json={"chunk_id": 0, "elapsed_ms": 0})
    assert interrupt_resp.status_code == 200
    interrupt = interrupt_resp.json()
    assert interrupt["chunkId"] == 0
    assert interrupt["docCharPosition"] >= 0


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
