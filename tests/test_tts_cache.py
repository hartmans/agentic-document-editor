from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock, Thread

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

import agentic_editor.tts_cache as tts_cache_module
from agentic_editor.poly_tts_cache import PollyTtsCache
from agentic_editor.qwen_tts_cache import (
    QWEN_TTS_BATCH_SIZE,
    QwenTtsCache,
    _register_omegaconf_safe_globals,
)
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

    def _synthesize_entry(
        self,
        text: str,
        synthesis_text: str,
    ) -> tuple[bytes, list[SpeechMark]]:
        audio_bytes = synthesis_text.encode("utf-8")
        marks = [
            SpeechMark(
                time_ms=0,
                chunk_char_start=0,
                chunk_char_end=len(text),
            )
        ]
        return audio_bytes, marks


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


def test_qwen_splits_text_into_paragraph_segments():
    cache = object.__new__(QwenTtsCache)

    segments = cache._split_text_into_qwen_segments(
        "First paragraph.\nStill first.\n\nSecond paragraph.\n\n  \nThird paragraph.  "
    )

    assert segments == [
        "First paragraph.\nStill first.",
        "Second paragraph.",
        "Third paragraph.",
    ]


def test_qwen_batches_paragraph_segments(monkeypatch):
    cache = object.__new__(QwenTtsCache)
    cache.temperature = None
    cache.language = "English"
    cache._sample_rate = None

    texts = [f"Paragraph {i}" for i in range(QWEN_TTS_BATCH_SIZE + 2)]
    combined_text = "\n\n".join(texts)

    class FakeModel:
        def __init__(self):
            self.calls: list[list[str]] = []

        def generate_voice_clone(
            self,
            *,
            text,
            language,
            voice_clone_prompt,
            non_streaming_mode,
            **kwargs,
        ):
            self.calls.append(list(text))
            assert language == ["English"] * len(text)
            assert non_streaming_mode is True
            assert len(voice_clone_prompt) == 1
            wavs = [
                np.array([float(item.rsplit(" ", 1)[-1])], dtype="float32")
                for item in text
            ]
            return wavs, 24000

    fake_model = FakeModel()
    monkeypatch.setattr(cache, "_ensure_loaded", lambda: fake_model)
    monkeypatch.setattr(
        cache,
        "_normalize_audio_array",
        lambda audio: np.asarray(audio, dtype="float32"),
    )

    audio, sample_rate = cache._generate_audio(combined_text, prompt_items=[object()])

    assert sample_rate == 24000
    assert cache._sample_rate == 24000
    assert fake_model.calls == [
        texts[:QWEN_TTS_BATCH_SIZE],
        texts[QWEN_TTS_BATCH_SIZE:],
    ]
    assert audio.tolist() == [float(i) for i in range(QWEN_TTS_BATCH_SIZE + 2)]


def test_qwen_prompt_cache_rebuilds_when_load_fails(monkeypatch, tmp_path):
    cache = object.__new__(QwenTtsCache)
    cache._prompt_cache_lock = Lock()

    prompt_path = tmp_path / "prompt.pt"
    prompt_path.write_bytes(b"legacy")
    rebuilt_items = [object()]
    serialized_payload = [{"ref_text": "hello"}]
    saved_payloads: list[tuple[object, Path]] = []
    build_calls: list[bool] = []

    monkeypatch.setattr(cache, "_prompt_cache_path", lambda: prompt_path)
    monkeypatch.setattr(cache, "_serialize_prompt_items", lambda items: serialized_payload)

    def fake_build_prompt_items():
        build_calls.append(True)
        return rebuilt_items

    monkeypatch.setattr(cache, "_build_prompt_items_for_voice", fake_build_prompt_items)

    def fake_torch_load(path, map_location=None, *, weights_only=None, **kwargs):
        raise RuntimeError("broken cache")

    def fake_torch_save(payload, path):
        saved_payloads.append((payload, path))

    monkeypatch.setattr("agentic_editor.qwen_tts_cache.torch.load", fake_torch_load)
    monkeypatch.setattr("agentic_editor.qwen_tts_cache.torch.save", fake_torch_save)

    prompt_items = cache._prompt_items_for_voice()

    assert prompt_items == rebuilt_items
    assert build_calls == [True]
    assert not prompt_path.exists()
    assert saved_payloads == [(serialized_payload, prompt_path)]


def test_qwen_prompt_cache_uses_serialized_cache_in_steady_state(monkeypatch, tmp_path):
    cache = object.__new__(QwenTtsCache)
    cache._prompt_cache_lock = Lock()

    prompt_path = tmp_path / "prompt.pt"

    class FakePromptItem:
        def __init__(self, *, ref_code, ref_spk_embedding, x_vector_only_mode, icl_mode, ref_text):
            self.ref_code = ref_code
            self.ref_spk_embedding = ref_spk_embedding
            self.x_vector_only_mode = x_vector_only_mode
            self.icl_mode = icl_mode
            self.ref_text = ref_text

    build_calls: list[bool] = []

    def fake_build_prompt_items():
        build_calls.append(True)
        return [
            FakePromptItem(
                ref_code=torch.tensor([1, 2, 3]),
                ref_spk_embedding=torch.tensor([0.25, 0.5]),
                x_vector_only_mode=False,
                icl_mode=True,
                ref_text="hello",
            )
        ]

    monkeypatch.setattr(cache, "_prompt_cache_path", lambda: prompt_path)
    monkeypatch.setattr(cache, "_voice_clone_prompt_item_cls", lambda: FakePromptItem)
    monkeypatch.setattr(cache, "_build_prompt_items_for_voice", fake_build_prompt_items)

    first_prompt_items = cache._prompt_items_for_voice()
    second_prompt_items = cache._prompt_items_for_voice()

    assert build_calls == [True]
    assert len(first_prompt_items) == 1
    assert len(second_prompt_items) == 1
    assert isinstance(second_prompt_items[0], FakePromptItem)
    assert torch.equal(second_prompt_items[0].ref_code, torch.tensor([1, 2, 3]))
    assert torch.equal(
        second_prompt_items[0].ref_spk_embedding,
        torch.tensor([0.25, 0.5]),
    )
    assert second_prompt_items[0].x_vector_only_mode is False
    assert second_prompt_items[0].icl_mode is True
    assert second_prompt_items[0].ref_text == "hello"


def test_qwen_serialized_prompt_payload_normalizes_omegaconf_ref_text(tmp_path):
    cache = object.__new__(QwenTtsCache)

    class FakePromptItem:
        def __init__(self, *, ref_code, ref_spk_embedding, x_vector_only_mode, icl_mode, ref_text):
            self.ref_code = ref_code
            self.ref_spk_embedding = ref_spk_embedding
            self.x_vector_only_mode = x_vector_only_mode
            self.icl_mode = icl_mode
            self.ref_text = ref_text

    payload = cache._serialize_prompt_items(
        [
            FakePromptItem(
                ref_code=torch.tensor([1, 2, 3]),
                ref_spk_embedding=torch.tensor([0.5]),
                x_vector_only_mode=False,
                icl_mode=True,
                ref_text=OmegaConf.create(["hello", "world"]),
            )
        ]
    )
    prompt_path = tmp_path / "prompt.pt"
    torch.save(payload, prompt_path)

    loaded = torch.load(prompt_path, map_location="cpu")

    assert loaded[0]["ref_text"] == "['hello', 'world']"


class ControlledTtsCache(AbstractTtsCache):
    driver_name = "controlled"
    audio_extension = "wav"

    def __init__(self, config: dict[str, object], boto_session: object, prefix: str = ""):
        self.started: list[str] = []
        self._release_events: dict[str, Event] = {}
        self._started_events: dict[str, Event] = {}
        self.fail_once: set[str] = set()
        self.calls: dict[str, int] = {}
        super().__init__(config, boto_session, prefix=prefix)
        self.voice_id = "Controlled"

    def add_gate(self, text: str) -> tuple[Event, Event]:
        started = Event()
        release = Event()
        self._started_events[text] = started
        self._release_events[text] = release
        return started, release

    def _synthesize_entry(
        self,
        text: str,
        synthesis_text: str,
    ) -> tuple[bytes, list[SpeechMark]]:
        self.started.append(text)
        self.calls[text] = self.calls.get(text, 0) + 1
        started = self._started_events.get(text)
        if started is not None:
            started.set()
        release = self._release_events.get(text)
        if release is not None:
            release.wait(timeout=5)
        if text in self.fail_once and self.calls[text] == 1:
            raise RuntimeError(f"boom:{text}")
        return (
            synthesis_text.encode("utf-8"),
            [SpeechMark(time_ms=0, chunk_char_start=0, chunk_char_end=len(text))],
        )


def test_getitem_promotes_background_item_to_urgent(monkeypatch, tmp_path):
    executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr("agentic_editor.tss_cache._TTS_EXECUTOR", executor)

    try:
        cache = ControlledTtsCache({"cache_dir": str(tmp_path)}, boto_session=None)
        first_started, first_release = cache.add_gate("first")
        cache.add_gate("second")
        urgent_started, urgent_release = cache.add_gate("urgent")

        cache.submit("first")
        assert first_started.wait(timeout=5)
        cache.submit("second")
        cache.submit("urgent")

        result_holder: dict[str, tuple[bytes, list[SpeechMark]]] = {}

        def load_urgent() -> None:
            result_holder["urgent"] = cache["urgent"]

        thread = Thread(target=load_urgent)
        thread.start()
        first_release.set()
        assert urgent_started.wait(timeout=5)
        urgent_release.set()
        thread.join(timeout=5)

        assert result_holder["urgent"][0] == b"urgent"
        assert cache.started[:2] == ["first", "urgent"]
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def test_failed_item_is_dropped_and_retried(monkeypatch, tmp_path):
    executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr("agentic_editor.tss_cache._TTS_EXECUTOR", executor)

    try:
        cache = ControlledTtsCache({"cache_dir": str(tmp_path)}, boto_session=None)
        cache.fail_once.add("retry")

        with pytest.raises(RuntimeError, match="boom:retry"):
            cache["retry"]

        audio_bytes, speech_marks = cache["retry"]

        assert audio_bytes == b"retry"
        assert speech_marks == [
            SpeechMark(time_ms=0, chunk_char_start=0, chunk_char_end=5)
        ]
        assert cache.calls["retry"] == 2
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def test_register_omegaconf_safe_globals_allows_torch_load(tmp_path):
    checkpoint_path = tmp_path / "omegaconf.pt"
    torch.save(
        {
            "value": OmegaConf.create(
                {
                    "items": ["hello", "world"],
                    "flag": True,
                }
            )
        },
        checkpoint_path,
    )

    _register_omegaconf_safe_globals()
    loaded = torch.load(checkpoint_path, map_location="cpu")

    assert loaded["value"]["items"] == ["hello", "world"]
    assert loaded["value"]["flag"] is True
