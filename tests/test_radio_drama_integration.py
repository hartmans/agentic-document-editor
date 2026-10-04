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
from agentic_editor.document import TextChunk, DocumentState
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


def test_shared_chunks_reuse_preparation_and_changed_chunks_prepare_once(tmp_path, speech_config):
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
    edited = state.copy()
    assert edited.speech is state.speech
    assert edited.chunks[0].dialogue_lines == chunk.dialogue_lines
    assert counter.read_text() == "x"

    async def exercise():
        try:
            await state.speech.plan_chunks([chunk, edited.chunks[0]])
            assert chunk.plan is not None
            assert edited.chunks[0].plan is chunk.plan
            assert chunk.dialogue_lines[0].mark_offsets
            edited.update("there", 10, 15)
            changed = edited.chunks[0]
            edited.update("friend", 10, 15)
            assert changed is edited.chunks[0]
            assert counter.read_text() == "x"
            edited.prepare_changed_chunks()
            edited.prepare_changed_chunks()
            assert counter.read_text() == "xx"
            assert changed.filtered_text == "Hello friend."
            await state.speech.plan_chunks([changed])
            assert changed.plan is not chunk.plan
            assert not changed.changed
            assert chunk.filtered_text == "Hello world."
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
                    shifted = DocumentState(prefix + state.document, speech.voice_id, speech,
                                            chunks=[TextChunk(prefix), state.chunks[0]])
                    app.state.documents["shifted"] = shifted
                    shifted_response = await client.get("/chunks/shifted/1/marks")
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



def test_round_planning_splits_unavailable_resume_without_rendering(tmp_path, speech_config):
    from agentic_editor.tts import prepare_document_round, submit_document_audio
    async def exercise():
        speech = SpeechService(speech_config, tmp_path / "doc.txt")
        try:
            original = build_document_state("Hello world.", speech=speech)
            draft = original.copy()
            draft.update("friend", 6, 11)
            result = await prepare_document_round(original, draft, 6)
            assert [c.text for c in result.chunks] == ["Hello ", "friend."]
            assert all(c.plan is not None and not c.changed for c in result.chunks)
            assert not speech.audio_available(result.chunks[1])
            assert not speech._tasks  # Registration must not launch editor render tasks.
            await submit_document_audio(result, eager=False)
            assert not speech._tasks
            assert await prepare_document_round(result, result.copy(), 8) is result

            # Empty speech is genuinely available without invoking an engine.
            blank = build_document_state("  ", speech=speech)
            edited = blank.copy()
            edited.update(" ", 0, 0)
            ready = await prepare_document_round(blank, edited, 1)
            assert len(ready.chunks) == 1
            assert speech.audio_available(ready.chunks[0])
        finally:
            await speech.close()
    asyncio.run(exercise())


def test_streaming_playback_starts_next_chunk_and_background_timing(tmp_path, speech_config):
    """Requires GPU inference; validates editor ordering with real radio-drama."""
    from contextlib import aclosing
    from agentic_editor.tts import stream_document_audio, submit_document_audio, prepare_document_round
    async def exercise():
        prefetched = []
        timed = asyncio.Event()
        class ObservedSpeechService(SpeechService):
            def prefetch(self, chunks):
                prefetched.extend(chunks)
                return super().prefetch(chunks)

            async def _ensure_timing(self, chunk, result):
                await super()._ensure_timing(chunk, result)
                timed.set()

        speech = ObservedSpeechService(speech_config, tmp_path / "stream.txt")
        state = DocumentState("Hello world.Hello world.", speech.voice_id, speech,
                              chunks=[TextChunk("Hello world."), TextChunk("Hello world.")])
        try:
            state.prepare_changed_chunks()
            await submit_document_audio(state, eager=False)
            assert not prefetched
            assert not speech.audio_available(state.chunks[0])
            async with aclosing(stream_document_audio(state, 0)) as stream:
                first = await asyncio.wait_for(anext(stream), 240)
                assert first.frame_count > 0
                assert len(prefetched) == 1 and prefetched[0] is state.chunks[1]
                async for _ in stream:
                    pass
            await asyncio.wait_for(timed.wait(), 240)
            marks = await speech.marks(state.chunks[0])
            assert marks
            # An edit elsewhere must preserve a rendered mid-chunk resume target.
            draft = state.copy()
            draft.update("Goodbye", len("Hello world."), len("Hello world.Hello"))
            result = await prepare_document_round(state, draft, 6)
            assert len(result.chunks) == 2
            assert result.chunks[0] is state.chunks[0]
        finally:
            await speech.close()
    asyncio.run(exercise())


def test_startup_prefetches_entire_document_and_empty_playback_needs_no_engine(tmp_path, speech_config):
    """Exercise real empty plans to test editor lifecycle without GPU inference."""
    from contextlib import aclosing
    from agentic_editor.tts import stream_document_audio, submit_document_audio
    async def exercise():
        rendered = []
        completed = asyncio.Event()
        class ObservedSpeechService(SpeechService):
            async def _render_and_time(self, chunk):
                await super()._render_and_time(chunk)
                rendered.append(chunk)
                if len(rendered) == 3:
                    completed.set()

        speech = ObservedSpeechService(speech_config, tmp_path / "empty.txt")
        state = DocumentState(" \n ", speech.voice_id, speech,
                              chunks=[TextChunk(" "), TextChunk("\n"), TextChunk(" ")])
        try:
            state.prepare_changed_chunks()
            await submit_document_audio(state)
            await asyncio.wait_for(completed.wait(), 10)
            assert {id(c) for c in rendered} == {id(c) for c in state.chunks}
            async with aclosing(stream_document_audio(state, 1)) as stream:
                results = [result async for result in stream]
            assert sum(result.frame_count for result in results) == 0
            assert await speech.marks(state.chunks[1]) == []
            wav = await speech.audio(state.chunks[1])
            assert sf.info(io.BytesIO(wav)).frames == 0
        finally:
            await speech.close()
    asyncio.run(exercise())
