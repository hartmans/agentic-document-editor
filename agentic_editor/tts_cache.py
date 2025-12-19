from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from agentic_editor.speech_marks import SpeechMark


def _speech_marks_from_records(records: Sequence[dict]) -> List[SpeechMark]:
    """Convert raw speech mark records to word-level SpeechMark objects."""
    marks: List[SpeechMark] = []
    for rec in records:
        try:
            marks.append(
                SpeechMark(
                    time_ms=int(rec["time_ms"]),
                    chunk_char_start=int(rec["chunk_char_start"]),
                    chunk_char_end=int(rec["chunk_char_end"]),
                )
            )
        except (KeyError, ValueError, TypeError):
            continue
    marks.sort(key=lambda m: m.time_ms)
    return marks


class TtsCache:
    """
    Simple on-disk cache for Polly TTS results keyed by (voice_id, text) sha256.
    Stores audio as MP3 and speech mark metadata as JSON alongside it.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        *,
        boto_session: boto3.session.Session | None,
        region_name: str,
        voice_id: str,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.voice_id = voice_id
        self.session = boto_session or boto3.Session()
        self.client = self.session.client("polly", region_name=region_name)

    def __getitem__(self, text: str) -> Tuple[bytes, List[SpeechMark]]:
        key = self._key(text)
        audio_path = self.cache_dir / f"{key}.mp3"
        meta_path = self.cache_dir / f"{key}.json"
        if audio_path.exists() and meta_path.exists():
            audio = audio_path.read_bytes()
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            records = meta.get("speech_marks", [])
            return audio, _speech_marks_from_records(records)

        audio_bytes, records = self._synthesize(text)
        audio_path.write_bytes(audio_bytes)
        meta_path.write_text(
            json.dumps({"speech_marks": records}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return audio_bytes, _speech_marks_from_records(records)

    def _key(self, text: str) -> str:
        h = hashlib.sha256()
        h.update(self.voice_id.encode("utf-8"))
        h.update(b"\0")
        h.update(text.encode("utf-8"))
        return h.hexdigest()

    def _call_polly(self, **kwargs):
        try:
            return self.client.synthesize_speech(**kwargs, Engine="neural")
        except ClientError:
            return self.client.synthesize_speech(**kwargs)

    def _synthesize(self, text: str) -> Tuple[bytes, List[dict]]:
        # Audio
        audio_response = self._call_polly(
            Text=text,
            VoiceId=self.voice_id,
            OutputFormat="mp3",
        )
        audio_stream = audio_response.get("AudioStream")
        if audio_stream is None:
            raise RuntimeError("Polly response missing AudioStream for audio synthesis")
        audio_bytes = audio_stream.read()

        # Speech marks
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
        speech_marks = list(self._parse_speech_mark_lines(lines, text))
        return audio_bytes, speech_marks

    @staticmethod
    def _byte_to_char_offset(byte_offset: int, cumulative_bytes: List[int]) -> int:
        import bisect

        idx = bisect.bisect_right(cumulative_bytes, byte_offset) - 1
        return max(idx, 0)

    def _parse_speech_mark_lines(
        self, lines: Iterable[str], chunk_text: str
    ) -> Iterable[dict]:
        cumulative_bytes: List[int] = [0]
        for ch in chunk_text:
            cumulative_bytes.append(cumulative_bytes[-1] + len(ch.encode("utf-8")))

        for line in lines:
            data = json.loads(line)
            mark_type = data.get("type")
            if mark_type not in {"word", "sentence"}:
                continue
            start_byte = int(data.get("start", 0))
            end_byte = int(data.get("end", start_byte))
            chunk_char_start = self._byte_to_char_offset(start_byte, cumulative_bytes)
            chunk_char_end = self._byte_to_char_offset(end_byte, cumulative_bytes)
            yield {
                "time_ms": int(data["time"]),
                "mark_type": mark_type,
                "text": data.get("value", ""),
                "chunk_char_start": chunk_char_start,
                "chunk_char_end": chunk_char_end,
            }
