from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib
import json
import logging
import re
import shlex
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import threading
from typing import List, Literal, Sequence
import typing

import sh

from agentic_editor.speech_mark_alignment import speech_marks_from_records
from agentic_editor.speech_marks import SpeechMark

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _speech_marks_to_records(marks: Sequence[SpeechMark]) -> List[dict]:
    """Serialize SpeechMark objects for JSON storage."""
    return [mark.model_dump() for mark in marks]


@dataclass
class _CacheEntry:
    text: str
    priority: Literal["background", "demand", "response"] = "background"
    state: str = "pending"
    result: tuple[bytes, List[SpeechMark]] | None = None
    exception: BaseException | None = None


@dataclass(frozen=True)
class _TextToken:
    normalized: str
    start: int
    end: int


@dataclass(frozen=True)
class _TimedToken:
    normalized: str
    time_ms: int


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
        self._response_items: deque[str] = deque()
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

    def submit(
        self, text: str, *, priority: Literal["background", "response"] = "background"
    ) -> None:
        if priority not in ("background", "response"):
            raise ValueError(f"Unknown submission priority: {priority}")
        key = self._key(text)
        audio_path = self._audio_path(key)
        marks_path = self._marks_path(key)
        if audio_path.exists() and marks_path.exists():
            return

        with self._work_available:
            entry = self._cache.get(key)
            if entry is None:
                entry = _CacheEntry(text=text)
                self._cache[key] = entry
                self._enqueue_background_locked(key)
            if (
                priority == "response" and isinstance(entry, _CacheEntry)
                and entry.state == "pending" and entry.priority != "response"
            ):
                entry.priority = "response"
                self._background_keys.discard(key)
                self._urgent_keys.discard(key)
                self._response_items.append(key)
                self._work_available.notify()

    def submit_ordered(
        self, texts: Sequence[str], *, response_texts: Sequence[str] = ()
    ) -> None:
        """Queue responses first and move pending document texts ahead of backlog.

        Register the entire group under the scheduler lock. Running synthesis and
        demand reads retain their precedence over document background work.
        """
        with self._work_available:
            for text in response_texts:
                self.submit(text, priority="response")
            preferred: list[str] = []
            seen: set[str] = set()
            for text in texts:
                self.submit(text)
                key = self._key(text)
                if key in self._background_keys and key not in seen:
                    preferred.append(key)
                    seen.add(key)
            self._background_items = deque(preferred + [
                key for key in self._background_items
                if key in self._background_keys and key not in seen
            ])
            self._work_available.notify_all()

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
                if entry.state == "pending" and entry.priority == "background":
                    entry.priority = "demand"
                    self._background_keys.discard(key)
                    self._enqueue_urgent_locked(key)
            elif audio_path.exists() and marks_path.exists():
                pass
            else:
                entry = _CacheEntry(text=text, priority="demand")
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

    def _reconcile_speech_marks(
        self,
        text: str,
        synthesis_text: str,
        word_records: Sequence[dict[str, typing.Any]],
    ) -> List[SpeechMark]:
        if not word_records:
            return []
        if text == synthesis_text:
            return speech_marks_from_records(word_records, text)

        original_tokens = self._tokenize_text(text)
        synthesis_tokens = self._tokenize_text(synthesis_text)
        aligned_tokens = self._tokenize_word_records(word_records)
        if not original_tokens or not synthesis_tokens or not aligned_tokens:
            return speech_marks_from_records(word_records, text)

        synth_to_original = {
            source_idx: target_idx
            for source_idx, target_idx, _ in self._align_token_sequences(
                [token.normalized for token in synthesis_tokens],
                [token.normalized for token in original_tokens],
            )
        }
        aligned_to_synth = self._align_token_sequences(
            [token.normalized for token in aligned_tokens],
            [token.normalized for token in synthesis_tokens],
        )

        speech_marks: list[SpeechMark] = []
        seen_spans: set[tuple[int, int, int]] = set()
        for aligned_idx, synth_idx, cost in aligned_to_synth:
            if cost > 0.75:
                continue
            original_idx = synth_to_original.get(synth_idx)
            if original_idx is None:
                continue
            source_token = aligned_tokens[aligned_idx]
            original_token = original_tokens[original_idx]
            key = (
                source_token.time_ms,
                original_token.start,
                original_token.end,
            )
            if key in seen_spans:
                continue
            seen_spans.add(key)
            speech_marks.append(
                SpeechMark(
                    time_ms=source_token.time_ms,
                    chunk_char_start=original_token.start,
                    chunk_char_end=original_token.end,
                )
            )

        if not speech_marks:
            return speech_marks_from_records(word_records, text)
        speech_marks.sort(key=lambda mark: mark.time_ms)
        return speech_marks

    def _tokenize_text(self, text: str) -> list[_TextToken]:
        tokens: list[_TextToken] = []
        for match in _TOKEN_RE.finditer(text):
            normalized = self._normalize_token(match.group(0))
            if not normalized:
                continue
            tokens.append(
                _TextToken(
                    normalized=normalized,
                    start=match.start(),
                    end=match.end(),
                )
            )
        return tokens

    def _tokenize_word_records(
        self,
        word_records: Sequence[dict[str, typing.Any]],
    ) -> list[_TimedToken]:
        tokens: list[_TimedToken] = []
        for rec in word_records:
            if not isinstance(rec, dict):
                continue
            mark_type = rec.get("type") or rec.get("mark_type")
            if mark_type not in (None, "word"):
                continue
            time_value = rec.get("time_ms", rec.get("time"))
            word_value = rec.get("value") or rec.get("text")
            if time_value is None or word_value is None:
                continue
            try:
                time_ms = int(time_value)
            except (TypeError, ValueError):
                continue
            for match in _TOKEN_RE.finditer(str(word_value)):
                normalized = self._normalize_token(match.group(0))
                if not normalized:
                    continue
                tokens.append(_TimedToken(normalized=normalized, time_ms=time_ms))
        return tokens

    def _normalize_token(self, token: str) -> str:
        return token.lower()

    def _align_token_sequences(
        self,
        source_tokens: Sequence[str],
        target_tokens: Sequence[str],
    ) -> list[tuple[int, int, float]]:
        if not source_tokens or not target_tokens:
            return []

        insertion_cost = 0.9
        deletion_cost = 0.9
        rows = len(source_tokens) + 1
        cols = len(target_tokens) + 1
        costs = [[0.0] * cols for _ in range(rows)]
        steps: list[list[str | None]] = [[None] * cols for _ in range(rows)]

        for row in range(1, rows):
            costs[row][0] = row * deletion_cost
            steps[row][0] = "up"
        for col in range(1, cols):
            costs[0][col] = col * insertion_cost
            steps[0][col] = "left"

        for row in range(1, rows):
            for col in range(1, cols):
                substitution = self._token_substitution_cost(
                    source_tokens[row - 1],
                    target_tokens[col - 1],
                )
                diagonal_cost = costs[row - 1][col - 1] + substitution
                up_cost = costs[row - 1][col] + deletion_cost
                left_cost = costs[row][col - 1] + insertion_cost

                best_cost = diagonal_cost
                best_step = "diag"
                if up_cost < best_cost:
                    best_cost = up_cost
                    best_step = "up"
                if left_cost < best_cost:
                    best_cost = left_cost
                    best_step = "left"

                costs[row][col] = best_cost
                steps[row][col] = best_step

        pairs: list[tuple[int, int, float]] = []
        row = len(source_tokens)
        col = len(target_tokens)
        while row > 0 or col > 0:
            step = steps[row][col]
            if step == "diag":
                source_idx = row - 1
                target_idx = col - 1
                pairs.append(
                    (
                        source_idx,
                        target_idx,
                        self._token_substitution_cost(
                            source_tokens[source_idx],
                            target_tokens[target_idx],
                        ),
                    )
                )
                row -= 1
                col -= 1
                continue
            if step == "up":
                row -= 1
                continue
            if step == "left":
                col -= 1
                continue
            break

        pairs.reverse()
        return pairs

    def _token_substitution_cost(self, source: str, target: str) -> float:
        if source == target:
            return 0.0
        ratio = SequenceMatcher(a=source, b=target).ratio()
        return 1.0 - ratio

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
                    or (not self._response_items and not self._urgent_items and not self._background_items)
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
        while self._response_items:
            key = self._response_items.popleft()
            entry = self._cache.get(key)
            if isinstance(entry, _CacheEntry) and entry.state == "pending" and entry.priority == "response":
                return key, entry
        while self._urgent_items:
            key = self._urgent_items.popleft()
            self._urgent_keys.discard(key)
            entry = self._cache.get(key)
            if isinstance(entry, _CacheEntry) and entry.state == "pending" and entry.priority == "demand":
                return key, entry
        while self._background_items:
            key = self._background_items.popleft()
            if key not in self._background_keys:
                continue
            self._background_keys.discard(key)
            entry = self._cache.get(key)
            if isinstance(entry, _CacheEntry) and entry.state == "pending" and entry.priority == "background":
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
