"""Editor-owned text preparation and radio_drama resource integration."""

from __future__ import annotations

import asyncio
from dataclasses import fields
import hashlib
import io
import logging
from pathlib import Path
import re
import shlex
import subprocess
from typing import TYPE_CHECKING, Any

from .speech_marks import SpeechMark
from .text_alignment import match_filtered_words, project_word_marks

if TYPE_CHECKING:
    from .document import TextChunk


def document_cache_dir(base: str | Path, document_path: Path) -> Path:
    path = document_path.expanduser().resolve()
    name = re.sub(r"[^\w.-]+", "-", path.stem).strip(".-")[:80] or "document"
    digest = hashlib.sha256(str(path).encode()).hexdigest()[:16]
    return Path(base).expanduser().resolve() / f"{name}-{digest}"


class SpeechService:
    """One document's speech resources; radio_drama owns audio caching/batching.

    Preparation is synchronous so document edits can rebuild metadata without
    starting synthesis. Planning and all resource calls run on the app loop.
    """

    def __init__(self, config: dict[str, Any], document_path: Path):
        from radio_drama import ProductionConfig, SpeakerVoiceReference

        self.config = dict(config)
        self.document_path = document_path.expanduser().resolve()
        self.tts = str(config.get("tts", "voxcpm2_longform")).strip().lower()
        self.cache_dir = document_cache_dir(config.get("cache_dir", ".tts_cache"), document_path)
        voice = config.get("voice_sample")
        if not voice:
            raise ValueError("voice_sample must name a reference recording for radio_drama TTS")
        voice_path = Path(voice).expanduser().resolve()
        if not voice_path.is_file():
            raise ValueError(f"Voice reference does not exist: {voice_path}")
        self.voice_id = voice_path.stem
        self.speaker = SpeakerVoiceReference(
            authored_name="Narrator", voice_name=voice_path.name, resolved_path=voice_path,
        )
        settings = dict(config.get("radio_drama", {}))
        supported = {field.name for field in fields(ProductionConfig)}
        unknown = settings.keys() - supported
        if unknown:
            raise ValueError(f"Unknown radio_drama settings: {sorted(unknown)}")
        self.production_config = ProductionConfig(**settings)
        self._prepared: dict[str, tuple] = {}
        self._plans: dict[str, Any] = {}
        self._audio: dict[str, asyncio.Task] = {}
        self._marks: dict[str, asyncio.Task] = {}
        self._planning_lock = asyncio.Lock()
        self._injector = None
        self._ainjector = None

    def prepare_chunk(self, chunk: TextChunk) -> None:
        from radio_drama import DialogueLine

        prepared = self._prepared.get(chunk.text)
        if prepared is None:
            text = chunk.text
            command = self.config.get("filter")
            if command:
                text = subprocess.run(
                    shlex.split(str(command)), input=text, text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
                ).stdout
            matches = match_filtered_words(chunk.text, text)
            offsets = tuple(sorted({m.spoken_start for m in matches} | {m.spoken_end for m in matches}))
            # Empty filtered chunks still have a plan, but no synthesis request.
            lines = [DialogueLine(self.speaker, text, mark_offsets=offsets)] if text.strip() else []
            prepared = (text, lines, matches)
            self._prepared[chunk.text] = prepared
        chunk.filtered_text, lines, matches = prepared
        chunk.dialogue_lines = list(lines)
        chunk.word_correspondences = list(matches)

    async def initialize(self) -> None:
        if self._injector is not None:
            return
        from carthage.dependency_injection import AsyncInjector, InjectionKey
        from radio_drama import TtsResource, radio_drama_injector
        from radio_drama.cache import CACHE_DIRECTORY_KEY
        from radio_drama.planning import PRODUCTION_PLANNING_INJECTOR_KEY

        injector = radio_drama_injector(
            config=self.production_config, event_loop=asyncio.get_running_loop(),
            document_path=self.document_path,
        )
        injector.add_provider(CACHE_DIRECTORY_KEY, self.cache_dir)
        injector.add_provider(PRODUCTION_PLANNING_INJECTOR_KEY, injector)
        if injector.injector_containing(InjectionKey(TtsResource, tts=self.tts)) is None:
            injector.close()
            raise ValueError(
                f"TTS {self.tts!r} is not configured; define it in ~/.config/radio-drama/tts.toml"
            )
        self._injector = injector
        self._ainjector = injector(AsyncInjector)

    async def plan_chunks(self, chunks: list[TextChunk]) -> None:
        from radio_drama import ScriptNode, ScriptPlan, SourceLocation

        async with self._planning_lock:
            await self.initialize()
            for chunk in chunks:
                if chunk.plan is not None:
                    continue
                plan = self._plans.get(chunk.text)
                if plan is None:
                    node = ScriptNode(
                        location=SourceLocation(str(self.document_path), None, None),
                        attributes={"tts": self.tts},
                    )
                    plan = await self._ainjector(
                        ScriptPlan, node=node, script_events=chunk.dialogue_lines,
                        tts=self.tts, attrs={},
                    )
                    self._plans[chunk.text] = plan
                chunk.plan = plan

    @staticmethod
    def _task(entries: dict, key: str, operation) -> asyncio.Task:
        task = entries.get(key)
        if task is None or (task.done() and (task.cancelled() or task.exception() is not None)):
            task = asyncio.create_task(operation())
            entries[key] = task
            task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        return task

    async def audio(self, chunk: TextChunk) -> bytes:
        await self.plan_chunks([chunk])
        return await asyncio.shield(self._task(self._audio, chunk.text, lambda: self._encode_audio(chunk)))

    async def _encode_audio(self, chunk: TextChunk) -> bytes:
        result = await chunk.plan.render()
        return await asyncio.to_thread(self._wav_bytes, result.audio)

    def _wav_bytes(self, audio) -> bytes:
        import soundfile as sf

        output = io.BytesIO()
        sf.write(output, audio, self.production_config.resolved_output_sample_rate,
                 format="WAV", subtype="PCM_16")
        return output.getvalue()

    async def marks(self, chunk: TextChunk) -> list[SpeechMark]:
        await self.plan_chunks([chunk])
        return await asyncio.shield(self._task(self._marks, chunk.text, lambda: self._align_marks(chunk)))

    async def _align_marks(self, chunk: TextChunk) -> list[SpeechMark]:
        if not chunk.dialogue_lines:
            return []
        result = await chunk.plan.render()
        timing = await chunk.plan.ensure_timing(chunk.dialogue_lines, result)
        marks = []
        # The concrete plan calls stay here; playback consumes chunk-local
        # audio/marks, leaving room for a future shared plan plus script slices.
        for line, span in zip(chunk.dialogue_lines, timing.dialogue_lines, strict=True):
            starts = {offset: mark.next_start
                      for offset, mark in zip(line.mark_offsets, span.marks, strict=True)}
            marks.extend(project_word_marks(chunk.word_correspondences, starts))
        return sorted(marks, key=lambda mark: mark.time_ms)

    def prefetch(self, chunks: list[TextChunk]) -> None:
        # Every plan must be registered before kicking off rendering. No editor
        # worker limits or priority queues interfere with radio_drama batching.
        for chunk in chunks:
            task = self._task(self._audio, chunk.text, lambda chunk=chunk: self._encode_audio(chunk))
            task.add_done_callback(self._report_failure)

    @staticmethod
    def _report_failure(task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception() is not None:
            logging.error("Background speech rendering failed: %s", task.exception())

    async def close(self) -> None:
        tasks = [*self._audio.values(), *self._marks.values()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._injector is not None:
            self._injector.close()
        self._injector = self._ainjector = None
        self._plans.clear()
        self._audio.clear()
        self._marks.clear()
