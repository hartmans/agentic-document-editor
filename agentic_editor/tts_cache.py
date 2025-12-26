from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
import typing

import boto3
from botocore.exceptions import ClientError

from agentic_editor.speech_marks import SpeechMark

_TTS_EXECUTOR = ThreadPoolExecutor(max_workers=6)


def _speech_marks_from_records(records: Sequence[dict]) -> List[SpeechMark]:
    """Convert serialized SpeechMark records to SpeechMark objects."""
    marks: List[SpeechMark] = []
    for rec in records:
        try:
            if isinstance(rec, SpeechMark):
                marks.append(rec)
            else:
                marks.append(SpeechMark.model_validate(rec))
        except (KeyError, ValueError, TypeError):
            continue
    marks.sort(key=lambda m: m.time_ms)
    return marks


def _speech_marks_to_records(marks: Sequence[SpeechMark]) -> List[dict]:
    """Serialize SpeechMark objects for JSON storage."""
    return [mark.model_dump() for mark in marks]


class TtsCache:
    """
    On-disk cache for Polly TTS results keyed by (voice_id, engine, text) sha256.
    Stores audio as MP3 and speech mark metadata as JSON alongside it.
    """

    def __init__(
        self,
        config: dict[str, typing.Any],
        boto_session: boto3.session.Session | None,
        prefix: str = "",
    ) -> None:
        self.config = config
        self.prefix = prefix
        if self.prefix and not self.prefix.endswith("_"):
            self.prefix = f"{self.prefix}_"
        voice_id = self._get_config("voice_id")
        engine = self._get_config("engine")
        cache_dir = self._get_config("cache_dir")
        region_name = self._get_config("region_name")

        self.voice_id = voice_id if voice_id is not None else "Joanna"
        self.engine = engine if engine is not None else "neural"
        cache_dir = cache_dir if cache_dir is not None else ".tts_cache"
        self.region_name = (
            region_name if region_name is not None else "us-east-1"
        )

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = boto_session or boto3.Session()
        self.client = self.session.client("polly", region_name=self.region_name)
        self._cache: dict[str, dict[str, object]] = {}

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

        entry: dict[str, object] = {}
        audio_future = _TTS_EXECUTOR.submit(self._synthesize_audio, text, audio_path)
        entry["audio"] = audio_future
        audio_future.add_done_callback(
            lambda fut, cache_key=key: self._handle_future_done(
                cache_key, "audio", fut
            )
        )

        marks_future = _TTS_EXECUTOR.submit(self._synthesize_speech_marks, text, marks_path)
        entry["marks"] = marks_future
        marks_future.add_done_callback(
            lambda fut, cache_key=key: self._handle_future_done(
                cache_key, "speech marks", fut
            )
        )

        self._cache[key] = entry

    def __getitem__(self, text: str) -> Tuple[bytes, List[SpeechMark]]:
        key = self._key(text)
        entry = self._cache.get(key)
        audio_path = self._audio_path(key)
        marks_path = self._marks_path(key)

        if entry is None:
            self.submit(text)
            entry = self._cache.get(key)

        if entry is None:
            # If both paths exist, submit is a no-op.
            assert audio_path.exists() and marks_path.exists()
            audio_bytes = audio_path.read_bytes()
            marks = self._read_speech_marks(marks_path)
            return audio_bytes, marks

        audio_bytes = self._resolve_cache_entry(entry, "audio")
        marks = self._resolve_cache_entry(entry, "marks")
        return audio_bytes, marks

    def _resolve_cache_entry(self, entry: dict[str, object], key: str):
        value = entry.get(key)
        if isinstance(value, Future):
            result = value.result()
            entry[key] = result
            return result
        return value

    def _audio_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.mp3"

    def _marks_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _key(self, text: str) -> str:
        h = hashlib.sha256()
        h.update(str(self.voice_id).encode("utf-8"))
        h.update(b"\0")
        h.update(str(self.engine).encode("utf-8"))
        h.update(b"\0")
        h.update(text.encode("utf-8"))
        return h.hexdigest()

    def _call_polly(self, **kwargs):
        if self.engine:
            try:
                return self.client.synthesize_speech(**kwargs, Engine=self.engine)
            except ClientError as exc:
                logging.warning(
                    "Polly synthesis failed with engine %s, retrying without engine: %s",
                    self.engine,
                    exc,
                )
                return self.client.synthesize_speech(**kwargs)
        return self.client.synthesize_speech(**kwargs)

    def _synthesize_audio(self, text: str, audio_path: Path | None = None) -> bytes:
        audio_response = self._call_polly(
            Text=text,
            VoiceId=self.voice_id,
            OutputFormat="mp3",
        )
        audio_stream = audio_response.get("AudioStream")
        if audio_stream is None:
            raise RuntimeError("Polly response missing AudioStream for audio synthesis")
        audio_bytes = audio_stream.read()
        if audio_path is not None:
            audio_path.write_bytes(audio_bytes)
        return audio_bytes

    def _synthesize_speech_marks(
        self, text: str, marks_path: Path | None = None
    ) -> List[SpeechMark]:
        marks_response = self._call_polly(
            Text=text,
            VoiceId=self.voice_id,
            OutputFormat="json",
            SpeechMarkTypes=["word", "sentence"],
        )
        marks_stream = marks_response.get("AudioStream")
        if marks_stream is None:
            raise RuntimeError("Polly response missing AudioStream for speech marks")
        raw = marks_stream.read().decode("utf-8")
        lines = [line for line in raw.splitlines() if line.strip()]
        speech_marks = self._parse_speech_mark_lines(lines, text)
        if marks_path is not None:
            marks_path.write_text(
                json.dumps(
                    {"speech_marks": _speech_marks_to_records(speech_marks)},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        return speech_marks

    def _read_speech_marks(self, meta_path: Path) -> List[SpeechMark]:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        records = meta.get("speech_marks", [])
        return _speech_marks_from_records(records)

    def _handle_future_done(self, key: str, label: str, future: Future) -> None:
        try:
            exc = future.exception()
        except Exception as err:
            logging.error("TTS %s synthesis crashed for key %s: %s", label, key, err)
            self._cache.pop(key, None)
            return
        if exc is None:
            return
        logging.error("TTS %s synthesis failed for key %s: %s", label, key, exc)
        self._cache.pop(key, None)

    @staticmethod
    def _byte_to_char_offset(byte_offset: int, cumulative_bytes: List[int]) -> int:
        import bisect

        idx = bisect.bisect_right(cumulative_bytes, byte_offset) - 1
        return max(idx, 0)

    def _parse_speech_mark_lines(
        self, lines: Iterable[str], chunk_text: str
    ) -> List[SpeechMark]:
        cumulative_bytes: List[int] = [0]
        for ch in chunk_text:
            cumulative_bytes.append(cumulative_bytes[-1] + len(ch.encode("utf-8")))

        marks: List[SpeechMark] = []
        for line in lines:
            data = json.loads(line)
            mark_type = data.get("type")
            if mark_type not in {"word", "sentence"}:
                continue
            start_byte = int(data.get("start", 0))
            end_byte = int(data.get("end", start_byte))
            chunk_char_start = self._byte_to_char_offset(start_byte, cumulative_bytes)
            chunk_char_end = self._byte_to_char_offset(end_byte, cumulative_bytes)
            marks.append(
                SpeechMark(
                    time_ms=int(data["time"]),
                    chunk_char_start=chunk_char_start,
                    chunk_char_end=chunk_char_end,
                )
            )
        marks.sort(key=lambda mark: mark.time_ms)
        return marks
