from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
import logging
import shlex
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import threading
from typing import List, Sequence
import typing

import sh

from agentic_editor.speech_marks import SpeechMark



def _speech_marks_to_records(marks: Sequence[SpeechMark]) -> List[dict]:
    """Serialize SpeechMark objects for JSON storage."""
    return [mark.model_dump() for mark in marks]


@dataclass
class _CacheEntry:
    text: str
    urgent: bool = False
    state: str = "pending"
    result: tuple[bytes, List[SpeechMark]] | None = None
    exception: BaseException | None = None


class AbstractTtsCache(ABC):
    """
    Shared on-disk cache for synthesized speech results.

    Subclasses provide engine-specific text preparation, audio synthesis,
    and speech-mark generation while callers continue to use `submit` and
    `__getitem__`.
    """

    driver_name = "base"
    audio_extension = "bin"
    concurrency_level = 6

    def __init__(
        self,
        config: dict[str, typing.Any],
        boto_session: object | None,
        prefix: str = "",
    ) -> None:
        self.config = config
        self.prefix = prefix
        if self.prefix and not self.prefix.endswith("_"):
            self.prefix = f"{self.prefix}_"
        self._tts_executor = ThreadPoolExecutor(max_workers = self.concurrency_level)

        self.filter = self._get_config("filter")
        cache_dir = self._get_config("cache_dir")
        if cache_dir is None:
            cache_dir = ".tts_cache"

        self.cache_dir = Path(str(cache_dir))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.voice_id = ""
        self.region_name: str | None = None
        self.session = boto_session
        self._cache: dict[
            str,
            _CacheEntry | tuple[bytes, List[SpeechMark]],
        ] = {}
        self._lock = threading.RLock()
        self._work_available = threading.Condition(self._lock)
        self._background_items: deque[str] = deque()
        self._background_keys: set[str] = set()
        self._urgent_items: deque[str] = deque()
        self._urgent_keys: set[str] = set()
        self._running_workers = 0
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name=f"{self.__class__.__name__}-scheduler",
            daemon=True,
        )
        self._scheduler_thread.start()

    def _get_config(self, key: str) -> object | None:
        prefixed_key = f"{self.prefix}{key}" if self.prefix else key
        if prefixed_key in self.config:
            return self.config[prefixed_key]
        if key in self.config:
            return self.config[key]
        return None

    def submit(self, text: str) -> None:
        key = self._key(text)
        audio_path = self._audio_path(key)
        marks_path = self._marks_path(key)
        if audio_path.exists() and marks_path.exists():
            return

        with self._work_available:
            entry = self._cache.get(key)
            if entry is not None:
                return
            self._cache[key] = _CacheEntry(text=text)
            self._enqueue_background_locked(key)

    def __getitem__(self, text: str) -> tuple[bytes, List[SpeechMark]]:
        key = self._key(text)
        audio_path = self._audio_path(key)
        marks_path = self._marks_path(key)
        entry: _CacheEntry | None = None

        with self._work_available:
            cached_entry = self._cache.get(key)
            if isinstance(cached_entry, tuple):
                return cached_entry
            if isinstance(cached_entry, _CacheEntry):
                entry = cached_entry
                if entry.state == "pending" and not entry.urgent:
                    entry.urgent = True
                    self._background_keys.discard(key)
                    self._enqueue_urgent_locked(key)
            elif audio_path.exists() and marks_path.exists():
                pass
            else:
                entry = _CacheEntry(text=text, urgent=True)
                self._cache[key] = entry
                self._enqueue_urgent_locked(key)

        if entry is None:
            assert audio_path.exists() and marks_path.exists()
            audio_bytes = audio_path.read_bytes()
            marks = self._read_speech_marks(marks_path)
            return audio_bytes, marks

        return self._resolve_cache_entry(key, entry)

    def _resolve_cache_entry(
        self,
        cache_key: str,
        entry: _CacheEntry,
    ) -> tuple[bytes, List[SpeechMark]]:
        with self._lock:
            while True:
                current = self._cache.get(cache_key)
                if isinstance(current, tuple):
                    return current
                if entry.result is not None:
                    return entry.result
                if entry.exception is not None:
                    raise entry.exception
                self._work_available.wait()

    def _prepare_text(self, text: str) -> str:
        if self.filter:
            command, *filter_rest = shlex.split(str(self.filter))
            text = str(sh.Command(command)(*filter_rest, _in=text))
        return self._transform_text(text)

    def _transform_text(self, text: str) -> str:
        return text

    def _audio_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.{self.audio_extension}"

    def _marks_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _key(self, text: str) -> str:
        h = hashlib.sha256()
        for part in self._key_parts(text):
            h.update(part.encode("utf-8"))
            h.update(b"\0")
        return h.hexdigest()

    def _key_parts(self, text: str) -> Sequence[str]:
        return (
            self.driver_name,
            str(self.voice_id),
            str(self.filter or ""),
            text,
        )

    def _write_speech_marks(
        self, marks_path: Path, speech_marks: Sequence[SpeechMark]
    ) -> None:
        marks_path.write_text(
            json.dumps(
                {"speech_marks": _speech_marks_to_records(speech_marks)},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _read_speech_marks(self, meta_path: Path) -> List[SpeechMark]:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        records = meta.get("speech_marks", [])
        marks: List[SpeechMark] = []
        for rec in records:
            try:
                marks.append(SpeechMark.model_validate(rec))
            except (KeyError, ValueError, TypeError):
                logging.exception(
                    "Error reading marks; invalidating cache for future run"
                )
                meta_path.unlink()
                continue
        return marks

    def _synthesize_cache_entry(
        self,
        text: str,
        audio_path: Path,
        marks_path: Path,
    ) -> tuple[bytes, List[SpeechMark]]:
        synthesis_text = self._prepare_text(text)
        audio_bytes, speech_marks = self._synthesize_entry(text, synthesis_text)
        audio_path.write_bytes(audio_bytes)
        self._write_speech_marks(marks_path, speech_marks)
        return audio_bytes, list(speech_marks)

    def _scheduler_loop(self) -> None:
        while True:
            try:
                self._dispatch_next_work_item()
            except Exception:
                logging.exception("TTS scheduler loop hit an unexpected error")

    def _dispatch_next_work_item(self) -> None:
        with self._work_available:
            while True:
                while (
                    self._running_workers >= self._tts_executor._max_workers
                    or (not self._urgent_items and not self._background_items)
                ):
                    self._work_available.wait()
                scheduled = self._get_next_pending_entry_locked()
                if scheduled is not None:
                    key, entry = scheduled
                    entry.state = "running"
                    self._running_workers += 1
                    break
        future = self._tts_executor.submit(
            self._synthesize_cache_entry,
            entry.text,
            self._audio_path(key),
            self._marks_path(key),
        )
        future.add_done_callback(
            lambda fut, cache_key=key, cache_entry=entry: self._handle_future_done(
                cache_key,
                cache_entry,
                fut,
            )
        )

    def _get_next_pending_entry_locked(self) -> tuple[str, _CacheEntry] | None:
        while self._urgent_items:
            key = self._urgent_items.popleft()
            self._urgent_keys.discard(key)
            entry = self._cache.get(key)
            if isinstance(entry, _CacheEntry) and entry.state == "pending" and entry.urgent:
                return key, entry
        while self._background_items:
            key = self._background_items.popleft()
            if key not in self._background_keys:
                continue
            self._background_keys.discard(key)
            entry = self._cache.get(key)
            if isinstance(entry, _CacheEntry) and entry.state == "pending" and not entry.urgent:
                return key, entry
        return None

    def _enqueue_background_locked(self, key: str) -> None:
        if key in self._background_keys:
            return
        self._background_keys.add(key)
        self._background_items.append(key)
        self._work_available.notify()

    def _enqueue_urgent_locked(self, key: str) -> None:
        if key in self._urgent_keys:
            return
        self._urgent_keys.add(key)
        self._urgent_items.append(key)
        self._work_available.notify()

    def _handle_future_done(
        self,
        key: str,
        entry: _CacheEntry,
        future: Future[tuple[bytes, List[SpeechMark]]],
    ) -> None:
        try:
            result = future.result()
        except Exception as exc:
            logging.error("TTS synthesis failed for key %s: %s", key, exc)
            with self._work_available:
                self._running_workers -= 1
                entry.exception = exc
                entry.state = "failed"
                cached_entry = self._cache.get(key)
                if cached_entry is entry:
                    self._cache.pop(key, None)
                self._work_available.notify_all()
            return

        with self._work_available:
            self._running_workers -= 1
            entry.result = result
            entry.state = "done"
            cached_entry = self._cache.get(key)
            if cached_entry is entry:
                self._cache[key] = result
            self._work_available.notify_all()

    @abstractmethod
    def _synthesize_entry(
        self,
        text: str,
        synthesis_text: str,
    ) -> tuple[bytes, List[SpeechMark]]:
        raise NotImplementedError


__all__ = ["AbstractTtsCache"]
