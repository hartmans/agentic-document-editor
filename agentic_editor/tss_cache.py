from __future__ import annotations

import hashlib
import json
import logging
import shlex
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import List, Sequence
import typing

import sh

from agentic_editor.speech_marks import SpeechMark

_TTS_EXECUTOR = ThreadPoolExecutor(max_workers=6)


def _speech_marks_to_records(marks: Sequence[SpeechMark]) -> List[dict]:
    """Serialize SpeechMark objects for JSON storage."""
    return [mark.model_dump() for mark in marks]


class AbstractTtsCache(ABC):
    """
    Shared on-disk cache for synthesized speech results.

    Subclasses provide engine-specific text preparation, audio synthesis,
    and speech-mark generation while callers continue to use `submit` and
    `__getitem__`.
    """

    driver_name = "base"
    audio_extension = "bin"

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
            Future[tuple[bytes, List[SpeechMark]]] | tuple[bytes, List[SpeechMark]],
        ] = {}

    def _get_config(self, key: str) -> object | None:
        prefixed_key = f"{self.prefix}{key}" if self.prefix else key
        if prefixed_key in self.config:
            return self.config[prefixed_key]
        if key in self.config:
            return self.config[key]
        return None

    def submit(self, text: str) -> None:
        key = self._key(text)
        if key in self._cache:
            return

        audio_path = self._audio_path(key)
        marks_path = self._marks_path(key)
        if audio_path.exists() and marks_path.exists():
            return

        synthesis_text = self._prepare_text(text)
        future = _TTS_EXECUTOR.submit(
            self._synthesize_cache_entry,
            text,
            synthesis_text,
            audio_path,
            marks_path,
        )
        future.add_done_callback(
            lambda fut, cache_key=key: self._handle_future_done(cache_key, fut)
        )

        self._cache[key] = future

    def __getitem__(self, text: str) -> tuple[bytes, List[SpeechMark]]:
        key = self._key(text)
        entry = self._cache.get(key)
        audio_path = self._audio_path(key)
        marks_path = self._marks_path(key)

        if entry is None:
            self.submit(text)
            entry = self._cache.get(key)

        if entry is None:
            assert audio_path.exists() and marks_path.exists()
            audio_bytes = audio_path.read_bytes()
            marks = self._read_speech_marks(marks_path)
            return audio_bytes, marks

        return self._resolve_cache_entry(key, entry)

    def _resolve_cache_entry(
        self,
        cache_key: str,
        entry: Future[tuple[bytes, List[SpeechMark]]] | tuple[bytes, List[SpeechMark]],
    ) -> tuple[bytes, List[SpeechMark]]:
        if isinstance(entry, Future):
            result = entry.result()
            self._cache[cache_key] = result
            return result
        return entry

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
        synthesis_text: str,
        audio_path: Path,
        marks_path: Path,
    ) -> tuple[bytes, List[SpeechMark]]:
        audio_bytes, speech_marks = self._synthesize_entry(text, synthesis_text)
        audio_path.write_bytes(audio_bytes)
        self._write_speech_marks(marks_path, speech_marks)
        return audio_bytes, list(speech_marks)

    def _handle_future_done(
        self,
        key: str,
        future: Future[tuple[bytes, List[SpeechMark]]],
    ) -> None:
        try:
            exc = future.exception()
        except Exception as err:
            logging.error("TTS synthesis crashed for key %s: %s", key, err)
            self._cache.pop(key, None)
            return

        if exc is None:
            return

        logging.error("TTS synthesis failed for key %s: %s", key, exc)
        self._cache.pop(key, None)

    @abstractmethod
    def _synthesize_entry(
        self,
        text: str,
        synthesis_text: str,
    ) -> tuple[bytes, List[SpeechMark]]:
        raise NotImplementedError


__all__ = ["AbstractTtsCache"]
