from __future__ import annotations

import base64
import hashlib
import io
import logging
import wave
from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy as np
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

_DEFAULT_MODEL_NAME = "openai:google/gemma-4-E4B-it"
_DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
_DEFAULT_API_KEY = "EMPTY"
_DEFAULT_TEMPERATURE = 0.0
_SAMPLE_RATE = 16000
_MAX_AUDIO_SECONDS = 30.0
_MAX_REFERENCE_CHARS = 65536
_MAX_NEW_TOKENS = 2048

_SYSTEM_PROMPT = """You normalize spoken document-editing requests into precise instructions for a future editor model.

When you receive the reference document by itself, prepare guidance that will help disambiguate later audio.
When you receive audio, output only the final normalized instruction text that the future editor model should follow.

Rules for the final instruction output:
- Do not produce a literal transcript unless the literal wording is already the clearest instruction.
- Resolve names, jargon, spelling, capitalization, homophones, possessives, and plurals from document context.
- Apply speaker self-corrections so the output reflects the user's final intent.
- Prefer imperative editing instructions that are unambiguous for a later model.
- Do not describe the audio, reasoning, or your intermediate analysis.
- Do not add markdown, bullets, or quotation marks unless they are part of the intended instruction.
"""

_ANALYSIS_USER_PROMPT_TEMPLATE = """Reference document:
{reference_document}

Before hearing audio, analyze this document and produce compact guidance for a future model that will interpret spoken editing instructions against it.

Focus on:
- Hot words, names, titles, and uncommon spellings that are likely to appear in audio.
- Domain-specific terms, jargon, and capitalization conventions.
- Likely confusions involving homophones, possessives, plurals, contractions, and punctuation.
- How to turn spoken corrections into normalized editing instructions for a future editor model instead of literal transcription.

Return concise guidance only. Keep it compact but information-dense.
"""

_AUDIO_USER_PROMPT = """Use the reference document and your prior guidance to interpret the attached audio.

Output only the final normalized instruction text for a future editor model.

The output should be a clean editing instruction, not a literal transcript. Fold in self-corrections and resolve names or spelling from context.

Example:
If the speaker says "Change Linda said drop the balls to Linda said drop the ball. No wait, that was Laura not Linda,"
output:
Change Laura said drop the balls to Laura said drop the ball.
"""


@dataclass(frozen=True)
class _AnalysisCacheEntry:
    prompt_text: str
    response_text: str


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


def _coerce_float32_mono(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim == 1:
        return np.clip(array, -1.0, 1.0)
    if array.ndim == 2:
        return np.clip(array.mean(axis=1), -1.0, 1.0).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported audio shape: {array.shape!r}")


def _encode_wav_base64(audio: np.ndarray, sample_rate: int) -> str:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2", copy=False)
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return base64.b64encode(output.getvalue()).decode("ascii")


def _extract_text(message: BaseMessage | str) -> str:
    if isinstance(message, str):
        return message.strip()
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                text_parts.append(block["text"])
        if text_parts:
            return "\n".join(part.strip() for part in text_parts if part.strip()).strip()
    raise RuntimeError("Model response did not contain text content")


def _parse_model_ref(model_ref: str) -> tuple[str, str | None]:
    provider, sep, model_name = model_ref.partition(":")
    if sep and provider:
        return model_name, provider
    return model_ref, None


class Gemma4AsrReceiver:
    """Use an OpenAI-compatible vLLM chat endpoint for document-aware ASR normalization."""

    sample_rate = _SAMPLE_RATE

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        gemma4_config = config.get("gemma4") or {}
        if not isinstance(gemma4_config, dict):
            raise ValueError("gemma4 config must be a mapping if provided")

        self.model_name = str(gemma4_config.get("model") or _DEFAULT_MODEL_NAME)
        self.max_audio_seconds = float(
            gemma4_config.get("max_audio_seconds", _MAX_AUDIO_SECONDS)
        )
        self.max_reference_chars = int(
            gemma4_config.get("max_reference_chars", _MAX_REFERENCE_CHARS)
        )
        self.max_new_tokens = int(
            gemma4_config.get("max_new_tokens", _MAX_NEW_TOKENS)
        )

        self._llm_lock = Lock()
        self._llm = None
        self._analysis_lock = Lock()
        self._analysis_cache: dict[str, _AnalysisCacheEntry] = {}
        self._model_init_kwargs = self._build_model_init_kwargs(gemma4_config)

    def create_buffer(self) -> AudioWebSocketBuffer:
        return AudioWebSocketBuffer()

    def _build_model_init_kwargs(self, gemma4_config: dict[str, Any]) -> dict[str, Any]:
        model_name, model_provider = _parse_model_ref(
            str(gemma4_config.get("model") or _DEFAULT_MODEL_NAME)
        )
        init_kwargs: dict[str, Any] = {
            "model": model_name,
            "base_url": str(gemma4_config.get("base_url") or _DEFAULT_BASE_URL),
            "api_key": str(gemma4_config.get("api_key") or _DEFAULT_API_KEY),
            "temperature": float(
                gemma4_config.get("temperature", _DEFAULT_TEMPERATURE)
            ),
        }
        if model_provider:
            init_kwargs["model_provider"] = model_provider

        local_only_keys = {
            "model",
            "base_url",
            "api_key",
            "temperature",
            "max_audio_seconds",
            "max_reference_chars",
            "max_new_tokens",
        }
        for key, value in gemma4_config.items():
            if key not in local_only_keys:
                init_kwargs[key] = value
        return init_kwargs

    def _ensure_llm(self):
        if self._llm is not None:
            return self._llm
        with self._llm_lock:
            if self._llm is not None:
                return self._llm
            logging.info(
                "Initializing Gemma4 ASR chat model via OpenAI-compatible endpoint: %s",
                self.model_name,
            )
            self._llm = init_chat_model(**self._model_init_kwargs)
        return self._llm

    def _reference_document(self, document: str) -> str:
        cleaned = document.strip()
        if len(cleaned) <= self.max_reference_chars:
            return cleaned
        head_chars = self.max_reference_chars // 2
        tail_chars = self.max_reference_chars - head_chars
        return f"{cleaned[:head_chars].rstrip()}\n...\n{cleaned[-tail_chars:].lstrip()}"

    def _analysis_prompt_text(self, reference_document: str) -> str:
        return _ANALYSIS_USER_PROMPT_TEMPLATE.format(reference_document=reference_document)

    def _analysis_cache_key(self, prompt_text: str) -> str:
        return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()

    def _invoke_text(self, messages: list[BaseMessage]) -> str:
        llm = self._ensure_llm()
        response = llm.invoke(messages, max_tokens=self.max_new_tokens)
        if not isinstance(response, BaseMessage):
            raise RuntimeError("Chat model returned an unexpected response type")
        return _extract_text(response)

    def _get_document_analysis(self, document: str) -> _AnalysisCacheEntry:
        reference_document = self._reference_document(document)
        prompt_text = self._analysis_prompt_text(reference_document)
        cache_key = self._analysis_cache_key(prompt_text)
        cached = self._analysis_cache.get(cache_key)
        if cached is not None:
            return cached

        with self._analysis_lock:
            cached = self._analysis_cache.get(cache_key)
            if cached is not None:
                return cached
            response_text = self._invoke_text(
                [
                    SystemMessage(content=_SYSTEM_PROMPT),
                    HumanMessage(content=prompt_text),
                ]
            )
            entry = _AnalysisCacheEntry(
                prompt_text=prompt_text,
                response_text=response_text,
            )
            self._analysis_cache[cache_key] = entry
            return entry

    def _audio_message(self, audio: np.ndarray) -> HumanMessage:
        wav_base64 = _encode_wav_base64(audio, self.sample_rate)
        return HumanMessage(
            content=[
                {"type": "text", "text": _AUDIO_USER_PROMPT},
                {
                    "type": "audio",
                    "base64": wav_base64,
                    "mime_type": "audio/wav",
                },
            ]
        )

    def transcribe_audio(self, audio: np.ndarray, *, document: str) -> str:
        waveform = _coerce_float32_mono(audio)
        max_samples = int(round(self.max_audio_seconds * self.sample_rate))
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]

        analysis = self._get_document_analysis(document)
        return self._invoke_text(
            [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=analysis.prompt_text),
                AIMessage(content=analysis.response_text),
                self._audio_message(waveform),
            ]
        )


__all__ = ["AudioWebSocketBuffer", "Gemma4AsrReceiver"]
