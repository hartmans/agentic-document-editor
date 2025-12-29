from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
import typing
import sh
import shlex

import boto3
from botocore.exceptions import ClientError

from agentic_editor.speech_marks import SpeechMark

_TTS_EXECUTOR = ThreadPoolExecutor(max_workers=6)


def _speech_marks_from_records(
    records: Sequence[dict], chunk_text: str
) -> List[SpeechMark]:
    """Align Polly speech mark records against the chunk text."""
    marks: List[SpeechMark] = []
    if not records:
        return marks

    def _parse_int(value: object) -> int | None:
        return int(value)  # type: ignore[arg-type]

    search_pos = 0
    base_slop = 60
    max_slop = 100
    slop = base_slop
    chunk_len = len(chunk_text)
    for rec in records:
        if not isinstance(rec, dict):
            raise TypeError

        mark_type = rec.get("type") or rec.get("mark_type")
        word = rec.get("value") or rec.get("text")
        time_ms = _parse_int(rec.get("time_ms", rec.get("time")))

        if word and (mark_type is None or mark_type == "word") and time_ms is not None:
            if search_pos >= chunk_len:
                break
            search_start = search_pos
            search_end = min(chunk_len, search_start + slop)
            idx = chunk_text.find(word, search_start, search_end)
            if idx == -1:
                slop = min(max_slop, slop + len(word))
                continue
            marks.append(
                SpeechMark(
                    time_ms=time_ms,
                    chunk_char_start=idx,
                    chunk_char_end=idx + len(word),
                )
            )
            search_pos = idx + len(word)
            slop = base_slop
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
        boto_session: boto3.session.Session | None, # type: ignore
        prefix: str = "",
    ) -> None:
        self.config = config
        self.prefix = prefix
        if self.prefix and not self.prefix.endswith("_"):
            self.prefix = f"{self.prefix}_"
        voice_id = self._get_config("voice_id")
        engine = self._get_config("engine")
        self.filter = self._get_config('filter')
        cache_dir = self._get_config("cache_dir")
        region_name = self._get_config("region_name")

        self.voice_id = voice_id if voice_id is not None else "Ruth"
        self.engine = engine if engine is not None else "long-form"
        cache_dir = cache_dir if cache_dir is not None else ".tts_cache"
        self.region_name = (
            region_name if region_name is not None else "us-east-1"
        )

        self.cache_dir = Path(typing.cast(str, cache_dir))
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
        if self.filter:
            command, *filter_rest = shlex.split(self.filter)
            text = str(sh.Command(command)(
                *filter_rest,
                _in=text))
        text = f'<speak>{text}</speak>'
        
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
            try:
                return self.client.synthesize_speech(**kwargs, Engine=self.engine)
            except ClientError as exc:
                logging.error(
                    "Polly synthesis failed with engine %s: %s",
                    self.engine,
                    exc,
                )
                breakpoint()
                raise

    def _synthesize_audio(self, text: str, audio_path: Path | None = None) -> bytes:
        audio_response = self._call_polly(
            Text=text,
            VoiceId=self.voice_id,
            OutputFormat="mp3",
            TextType="ssml",
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
            TextType="ssml",
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
        marks: List[SpeechMark] = []
        for rec in records:
            try:
                marks.append(SpeechMark.model_validate(rec))
            except (KeyError, ValueError, TypeError):
                logging.exception('Error reading marks; invalidating cache for future run')
                meta_path.unlink()
                continue
        return marks

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

    def _parse_speech_mark_lines(
        self, lines: Iterable[str], chunk_text: str
    ) -> List[SpeechMark]:
        records: List[dict] = []
        for line in lines:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            mark_type = data.get("type")
            if mark_type not in {"word", "sentence"}:
                continue
            records.append(data)
        return _speech_marks_from_records(records, chunk_text)
