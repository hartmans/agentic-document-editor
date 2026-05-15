from __future__ import annotations

import logging
from threading import Lock
from typing import Any

import numpy as np
import torch

_DEFAULT_MODEL_NAME = "google/gemma-4-E4B-it"
_SAMPLE_RATE = 16000
_MAX_AUDIO_SECONDS = 30.0
_MAX_REFERENCE_CHARS = 65536
_MAX_NEW_TOKENS = 2048

_SYSTEM_PROMPT = """<|think|>
You are a literal transcription engine for spoken document-editing instructions.

Your job is to transcribe exactly what the speaker said.

Rules:
- Output only the transcription text.
- Do not summarize, interpret, or explain.
- Do not describe the audio.
- Do not add speaker labels, markdown, bullets, or quotes.
- Preserve likely punctuation and capitalization when they are strongly implied.
- Use the reference document to resolve spelling, punctuation, possessives, plurals, names, homonyms, and domain-specific wording.
- Pay careful attention to apostrophes, possessives, contractions, and homonyms.
- Prefer the most literal wording supported by the audio, not a cleaned-up paraphrase.
"""

_USER_PROMPT_TEMPLATE = """Transcribe the attached audio into literal text instructions for editing the reference document.

Only output the transcription itself.

Reference document:
{reference_document}
"""


class AudioWebSocketBuffer:
    """Accumulates raw PCM16LE mono websocket audio."""

    def __init__(self) -> None:
        self._chunks = bytearray()

    def append_bytes(self, chunk: bytes) -> None:
        if chunk:
            self._chunks.extend(chunk)

    def to_float32(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        if len(self._chunks) % 2 != 0:
            raise ValueError("PCM16 audio payload length must be even")
        audio = np.frombuffer(self._chunks, dtype="<i2").astype(np.float32) / 32768.0
        max_samples = int(_SAMPLE_RATE * _MAX_AUDIO_SECONDS)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        return audio


class Gemma4AsrReceiver:
    """Lazy wrapper around the Gemma 4 any-to-any pipeline for literal ASR."""

    model_name = _DEFAULT_MODEL_NAME
    sample_rate = _SAMPLE_RATE

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.device = self._normalize_device(str(config.get("device") or "cuda"))
        self.enable_thinking = True
        self.max_new_tokens = _MAX_NEW_TOKENS
        self.max_reference_chars = _MAX_REFERENCE_CHARS
        self._pipeline_lock = Lock()
        self._pipeline = None

    def create_buffer(self) -> AudioWebSocketBuffer:
        return AudioWebSocketBuffer()

    def _normalize_device(self, device: str) -> str:
        normalized = (device or "cpu").lower()
        if normalized == "cuda" and not torch.cuda.is_available():
            return "cpu"
        if normalized == "mps":
            return "cpu"
        if normalized not in {"cuda", "cpu"}:
            raise ValueError(f"Unsupported device: {device}")
        return normalized

    def _dtype_for_device(self) -> torch.dtype:
        if self.device == "cuda":
            return torch.bfloat16
        return torch.float32

    def _device_map_for_device(self) -> str:
        if self.device == "cuda":
            return "cuda:0"
        return "cpu"

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        with self._pipeline_lock:
            if self._pipeline is not None:
                return self._pipeline
            from transformers import pipeline

            logging.info("Loading Gemma 4 ASR pipeline for %s", self.model_name)
            self._pipeline = pipeline(
                "any-to-any",
                model=self.model_name,
                device_map=self._device_map_for_device(),
                dtype=self._dtype_for_device(),
            )
        return self._pipeline

    def _reference_document(self, document: str) -> str:
        cleaned = document.strip()
        if len(cleaned) <= self.max_reference_chars:
            return cleaned
        head_chars = self.max_reference_chars // 2
        tail_chars = self.max_reference_chars - head_chars
        return f"{cleaned[:head_chars].rstrip()}\n...\n{cleaned[-tail_chars:].lstrip()}"

    def transcribe_audio(self, audio: np.ndarray, *, document: str) -> str:
        pipe = self._ensure_pipeline()
        reference_document = self._reference_document(document)
        outputs = pipe(
            text=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": _SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio},
                        {
                            "type": "text",
                            "text": _USER_PROMPT_TEMPLATE.format(
                                reference_document=reference_document
                            ),
                        },
                    ],
                },
            ],
            max_new_tokens=self.max_new_tokens,
            return_full_text=True,
            enable_thinking=self.enable_thinking,
            processor_kwargs={"sampling_rate": self.sample_rate},
        )
        messages = outputs[0]["generated_text"]
        if not isinstance(messages, list) or not messages:
            raise RuntimeError("Gemma4 any-to-any pipeline returned an unexpected response shape")
        assistant_message = messages[-1]
        if not isinstance(assistant_message, dict):
            raise RuntimeError("Gemma4 assistant message is not a dictionary")
        content = assistant_message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("Gemma4 assistant content is missing or not text")
        return content.strip()


__all__ = ["AudioWebSocketBuffer", "Gemma4AsrReceiver"]
