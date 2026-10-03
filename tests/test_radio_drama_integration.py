"""Editor integration tests using the real radio_drama resources.

Library batching, cache formats, and alignment guarantees belong upstream; see
/docs/radio_drama_test_suggestions.md. No engine or model is mocked here.
"""
import asyncio
import io
import os
from pathlib import Path
import shlex
import sys

import httpx
import pytest
import soundfile as sf

import backend
from agentic_editor.document import TextChunk, DocumentState, apply_document_update
from agentic_editor.speech import SpeechService
from agentic_editor.tts import build_document_state


@pytest.fixture
def speech_config(tmp_path):
    voice = Path(os.environ.get("RADIO_DRAMA_TEST_VOICE", "chandra.wav")).expanduser().resolve()
    if not voice.is_file():
        pytest.fail("Set RADIO_DRAMA_TEST_VOICE to a real narrator reference recording")
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser()
    source = config_home / "radio-drama" / "tts.toml"
    if not source.is_file():
        pytest.fail("Configure the real voxcpm2_longform engine in ~/.config/radio-drama/tts.toml")
    return {
        "voice_sample": str(voice), "cache_dir": str(tmp_path / "cache"),
        "radio_drama": {"device": os.environ.get("RADIO_DRAMA_TEST_DEVICE", "cuda:1")},
        "llm": {"model": "openai:test", "base_url": "http://127.0.0.1:8080/v1", "api_key": "test"},
    }


def test_preparation_is_cached_across_edits_and_planning_is_separate(tmp_path, speech_config):
    script = tmp_path / "filter.py"
    counter = tmp_path / "filter_calls"
    script.write_text(
        "import sys\nfrom pathlib import Path\np=Path(sys.argv[1])\n"
        "p.write_text(p.read_text()+'x' if p.exists() else 'x')\n"
        "print(sys.stdin.read().replace('*',''), end='')\n"
    )
    speech_config["filter"] = shlex.join([sys.executable, str(script), str(counter)])
    path = tmp_path / "document.rst"
    state = build_document_state("**Hello** world.", config=speech_config, document_path=path)
    chunk = state.chunks[0]
    assert state.speech.tts == "voxcpm2_longform"
    assert chunk.filtered_text == "Hello world."
    assert len(chunk.dialogue_lines) == 1
    assert chunk.plan is None
    assert not state.speech.cache_dir.exists()
    edited = apply_document_update(state.document, state)
    assert edited.speech is state.speech
    assert edited.chunks[0].dialogue_lines == chunk.dialogue_lines
    assert counter.read_text() == "x"

    async def exercise():
        try:
            await state.speech.plan_chunks([chunk, edited.chunks[0]])
            assert chunk.plan is not None
            assert edited.chunks[0].plan is chunk.plan
            assert chunk.dialogue_lines[0].mark_offsets
        finally:
            await state.speech.close()
    asyncio.run(exercise())


def test_audio_endpoint_does_not_wait_for_marks_and_marks_use_revision_offsets(tmp_path, speech_config):
    async def exercise():
        entered = asyncio.Event()
        release = asyncio.Event()

        class DelayedMarksService(SpeechService):
            async def _align_marks(self, chunk):
                # Delay only the editor's alignment operation, then perform the
                # real inference. This exposes HTTP readiness without a fake TTS.
                entered.set()
                await release.wait()
                return await super()._align_marks(chunk)

        path = tmp_path / "document.txt"
        path.write_text("Hello world.")
        speech = DelayedMarksService(speech_config, path)
        state = build_document_state("Hello world.", speech=speech)
        app = backend.create_app(state, path, config=speech_config)
        pending = None
        try:
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://editor",
                ) as client:
                    document_id = app.state.document_id
                    pending = asyncio.create_task(client.get(f"/chunks/{document_id}/0/marks"))
                    await asyncio.wait_for(entered.wait(), 60)
                    audio = await asyncio.wait_for(client.get(f"/chunks/{document_id}/0/audio"), 240)
                    assert audio.status_code == 200, audio.text
                    assert audio.headers["content-type"] == "audio/wav"
                    info = sf.info(io.BytesIO(audio.content))
                    assert info.frames > 0
                    assert info.samplerate == speech.production_config.resolved_output_sample_rate
                    assert not pending.done()
                    release.set()
                    marks_response = await asyncio.wait_for(pending, 240)
                    assert marks_response.status_code == 200, marks_response.text
                    marks = marks_response.json()["marks"]
                    assert marks
                    assert all(state.document[m["docCharStart"]:m["docCharEnd"]] in {"Hello", "world"}
                               for m in marks)
                    prefix = "🌟 Préface\n\n"
                    shifted_chunk = TextChunk(state.document, len(prefix), len(prefix) + len(state.document))
                    speech.prepare_chunk(shifted_chunk)
                    shifted = DocumentState(prefix + state.document, speech.voice_id, speech,
                                            chunks=[shifted_chunk])
                    app.state.documents["shifted"] = shifted
                    shifted_response = await client.get("/chunks/shifted/0/marks")
                    assert shifted_response.status_code == 200
                    assert shifted_response.json()["marks"] == [
                        {**mark, "docCharStart": mark["docCharStart"] + len(prefix),
                         "docCharEnd": mark["docCharEnd"] + len(prefix)} for mark in marks
                    ]
                    assert (await client.get(f"/chunks/{document_id}/0/marks")).json()["marks"] == marks
        finally:
            release.set()
            if pending is not None and not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            await speech.close()
    asyncio.run(exercise())
