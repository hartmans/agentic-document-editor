from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Sequence
import typing

import boto3
from botocore.exceptions import ClientError

from agentic_editor.speech_marks import SpeechMark
from agentic_editor.tss_cache import AbstractTtsCache


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

    marks.sort(key=lambda mark: mark.time_ms)
    return marks


class PollyTtsCache(AbstractTtsCache):
    """
    On-disk cache for Amazon Polly TTS results keyed by configuration and text.
    Stores audio as MP3 and speech mark metadata as JSON alongside it.
    """

    driver_name = "poly"
    audio_extension = "mp3"

    def __init__(
        self,
        config: dict[str, typing.Any],
        boto_session: boto3.session.Session | None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, boto_session, prefix=prefix)

        voice_id = self._get_config("voice_id")
        engine = self._get_config("engine")
        region_name = self._get_config("region_name")

        self.voice_id = str(voice_id) if voice_id is not None else "Ruth"
        self.engine = str(engine) if engine is not None else "long-form"
        self.region_name = (
            str(region_name) if region_name is not None else "us-east-1"
        )

        self.session = boto_session or boto3.Session()
        self.client = self.session.client("polly", region_name=self.region_name)

    def _transform_text(self, text: str) -> str:
        return f"<speak>{text}</speak>"

    def _key_parts(self, text: str) -> Sequence[str]:
        return (
            self.driver_name,
            str(self.voice_id),
            str(self.engine),
            str(self.filter or ""),
            text,
        )

    def _call_polly(self, **kwargs):
        try:
            return self.client.synthesize_speech(**kwargs, Engine=self.engine)
        except ClientError as exc:
            logging.error(
                "Polly synthesis failed with engine %s: %s",
                self.engine,
                exc,
            )
            raise

    def _synthesize_entry(
        self,
        text: str,
        synthesis_text: str,
    ) -> tuple[bytes, List[SpeechMark]]:
        audio_response = self._call_polly(
            Text=synthesis_text,
            VoiceId=self.voice_id,
            OutputFormat="mp3",
            TextType="ssml",
        )
        audio_stream = audio_response.get("AudioStream")
        if audio_stream is None:
            raise RuntimeError("Polly response missing AudioStream for audio synthesis")
        audio_bytes = audio_stream.read()
        marks_response = self._call_polly(
            Text=synthesis_text,
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
        return audio_bytes, speech_marks

    def _parse_speech_mark_lines(
        self, lines: typing.Iterable[str], chunk_text: str
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


__all__ = ["PollyTtsCache"]
