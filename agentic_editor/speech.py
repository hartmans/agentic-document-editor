"""Editor-owned text preparation and radio_drama resource integration."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import fields
import hashlib
import logging
from pathlib import Path
import re
import shlex
import subprocess
from typing import TYPE_CHECKING, Any

from .speech_marks import SpeechMark
from .text_alignment import match_filtered_words, project_word_marks

if TYPE_CHECKING:
    from radio_drama import RenderResult
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
        self._tasks: set[asyncio.Task] = set()
        self._planning_lock = asyncio.Lock()
        self._injector = None
        self._ainjector = None

    def prepare_chunk(self, chunk: TextChunk) -> None:
        from radio_drama import DialogueLine

        if chunk.prepared:
            return
        text = chunk.text
        command = self.config.get("filter")
        if command and text.strip():
            text = subprocess.run(
                shlex.split(str(command)), input=text, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
            ).stdout
        matches = match_filtered_words(chunk.text, text)
        offsets = tuple(sorted({m.spoken_start for m in matches} | {m.spoken_end for m in matches}))
        chunk.filtered_text = text
        chunk.dialogue_lines = [DialogueLine(self.speaker, text, mark_offsets=offsets)] if text.strip() else []
        chunk.word_correspondences = list(matches)
        chunk.prepared = True

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
                self.prepare_chunk(chunk)
                node = ScriptNode(
                    location=SourceLocation(str(self.document_path), None, None),
                    attributes={"tts": self.tts},
                )
                chunk.plan = await self._ainjector(
                    ScriptPlan, node=node, script_events=chunk.dialogue_lines,
                    tts=self.tts, attrs={},
                )
                chunk.changed = False

    @staticmethod
    def audio_available(chunk: TextChunk) -> bool:
        if chunk.plan is None:
            return False
        request = chunk.plan.registered_request
        return request is None or request.audio_available()

    def _background(self, operation) -> asyncio.Task:
        task = asyncio.create_task(operation)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        task.add_done_callback(self._report_failure)
        return task

    async def stream_audio(self, chunk: TextChunk) -> AsyncIterator[RenderResult]:
        """Yield production-format PCM; radio-drama owns stream replay/lifetime.

        A consumer leaving early does not cancel radio-drama's producer. Timing
        generation joins that producer after its first nonempty audio arrives.
        """
        await self.plan_chunks([chunk])
        request = chunk.plan.registered_request
        started = False
        if request is None or request.audio_available():
            # A plan's memoized batch render may still be pending even though
            # streaming has already published a complete registration result.
            result = await request.render() if request is not None else await chunk.plan.render()
            self._background(self._ensure_timing(chunk, result))
            yield result
            return
        async with aclosing(request.render_stream()) as stream:
            async for result in stream:
                if result.frame_count and not started:
                    started = True
                    self._background(self._render_and_time(chunk))
                yield result

    async def _playback_result(self, chunk: TextChunk):
        # Starting a marks request must not start a batch ahead of playback's
        # stream for newly edited text. Both consumers join radio-drama's stream.
        async with aclosing(self.stream_audio(chunk)) as stream:
            async for _ in stream:
                pass
        request = chunk.plan.registered_request
        return await request.render() if request is not None else await chunk.plan.render()

    async def marks(self, chunk: TextChunk) -> list[SpeechMark]:
        await self.plan_chunks([chunk])
        return await self._align_marks(chunk)

    async def _align_marks(self, chunk: TextChunk) -> list[SpeechMark]:
        if not chunk.dialogue_lines:
            return []
        result = await self._playback_result(chunk)
        timing = await chunk.plan.ensure_timing(chunk.dialogue_lines, result)
        marks = []
        # The concrete plan calls stay here; playback consumes chunk-local
        # audio/marks, leaving room for a future shared plan plus script slices.
        for line, span in zip(chunk.dialogue_lines, timing.dialogue_lines, strict=True):
            starts = {offset: mark.next_start
                      for offset, mark in zip(line.mark_offsets, span.marks, strict=True)}
            marks.extend(project_word_marks(chunk.word_correspondences, starts))
        return sorted(marks, key=lambda mark: mark.time_ms)

    async def _ensure_timing(self, chunk: TextChunk, result) -> None:
        if chunk.dialogue_lines:
            await chunk.plan.ensure_timing(chunk.dialogue_lines, result)

    async def _render_and_time(self, chunk: TextChunk) -> None:
        result = await chunk.plan.render()
        await self._ensure_timing(chunk, result)

    def prefetch(self, chunks: list[TextChunk]) -> None:
        # Register all plans first. The library owns batching and timing sharing;
        # our task set exists only to observe failures and clean up at shutdown.
        for chunk in chunks:
            self._background(self._render_and_time(chunk))

    @staticmethod
    def _report_failure(task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception() is not None:
            logging.error("Background speech rendering failed: %s", task.exception())

    async def close(self) -> None:
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._injector is not None:
            self._injector.close()
        self._injector = self._ainjector = None
        self._tasks.clear()
