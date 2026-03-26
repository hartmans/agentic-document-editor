from __future__ import annotations

from pathlib import Path

import pytest

import agentic_editor.tts_cache as tts_cache_module
from agentic_editor.poly_tts_cache import PollyTtsCache
from agentic_editor.speech_marks import SpeechMark
from agentic_editor.tss_cache import AbstractTtsCache
from agentic_editor.tts_cache import TtsCache, resolve_tts_cache_class


class DummyTtsCache(AbstractTtsCache):
    driver_name = "dummy"
    audio_extension = "wav"

    def __init__(self, config: dict[str, object], boto_session: object, prefix: str = ""):
        super().__init__(config, boto_session, prefix=prefix)
        self.voice_id = "Dummy"
        self.region_name = None

    def _transform_text(self, text: str) -> str:
        return f"prepared:{text}"

    def _synthesize_audio(
        self,
        text: str,
        synthesis_text: str,
        audio_path: Path | None = None,
    ) -> bytes:
        audio_bytes = synthesis_text.encode("utf-8")
        if audio_path is not None:
            audio_path.write_bytes(audio_bytes)
        return audio_bytes

    def _synthesize_speech_marks(
        self,
        text: str,
        synthesis_text: str,
        marks_path: Path | None = None,
    ) -> list[SpeechMark]:
        marks = [
            SpeechMark(
                time_ms=0,
                chunk_char_start=0,
                chunk_char_end=len(text),
            )
        ]
        if marks_path is not None:
            self._write_speech_marks(marks_path, marks)
        return marks


def test_resolve_tts_cache_class_defaults_to_poly():
    assert resolve_tts_cache_class({}) is PollyTtsCache


def test_resolve_tts_cache_class_rejects_unknown_driver():
    with pytest.raises(ValueError, match="Unsupported speech_driver"):
        resolve_tts_cache_class({"speech_driver": "missing"})


def test_tts_cache_uses_prefixed_driver_override(monkeypatch, tmp_path):
    monkeypatch.setitem(tts_cache_module._DRIVER_FACTORIES, "dummy", DummyTtsCache)

    cache = TtsCache(
        {
            "speech_driver": "poly",
            "demo_speech_driver": "dummy",
            "demo_cache_dir": str(tmp_path),
        },
        boto_session=None,
        prefix="demo",
    )

    audio_bytes, speech_marks = cache["hello"]

    assert cache.voice_id == "Dummy"
    assert audio_bytes == b"prepared:hello"
    assert speech_marks == [
        SpeechMark(time_ms=0, chunk_char_start=0, chunk_char_end=5)
    ]
