from __future__ import annotations

import base64
import io
import json
import logging
import wave
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    message_to_dict,
    messages_to_dict,
)

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
When you receive audio, confirm it starts with the wake up phrase. If so, output only the final normalized instruction text that the future editor model should follow. If not, output an empty message.



The wakeup phrase is "Editor,". Examples of wakeup phrase handling:
You hear "Editor, change Linda to Laura." You output "Change Linda to Laura."
You hear "The cat is cute." You output an empty message.
You hear "Editor, is the speaker in the last paragraph clear?" You output "Is the speaker in the last paragraph clear?"


Rules for the final instruction output:
- Do not produce a literal transcript unless the literal wording is already the clearest instruction.
- Resolve names, jargon, spelling, capitalization, homophones, possessives, and plurals from document context.
- Apply speaker self-corrections so the output reflects the user's final intent.
- Make it clear whether the audio you receive is an imperative editing instruction, a repositioning request, or a question for the editing model to answer. Behavior of questions vs editing instructions is different, so be clear about that distinction.
- The instruction may talk about jumping to a position or playing from a given point; the editing model will understand that jumping to a position means continuing reading the document from that position.
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

Output only the final normalized prompt for a future editor model. If the audio contains no instructions, output an empty string. If the wakeup phrase is not present, output an empty string. 

The output should be a clean prompt (editing instruction, question, or answer to a question that you did not see), not a literal transcript. Fold in self-corrections and resolve names or spelling from context.

Example:
If the speaker says "Change Linda said drop the balls to Linda said drop the ball. No wait, that was Laura not Linda,"
output:
Change Laura said drop the balls to Laura said drop the ball.
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

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        document: str,
        debug_audio_dir: str | Path | None = None,
    ) -> None:
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
        self._model_init_kwargs = self._build_model_init_kwargs(gemma4_config)
        self._debug_audio_dir = (
            Path(debug_audio_dir).expanduser()
            if debug_audio_dir is not None
            else None
        )
        self._debug_audio_lock = Lock()
        self._debug_audio_counter = 0
        self._analysis_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="gemma4-analysis",
        )
        self._analysis_future: Future[list[BaseMessage]] = (
            self._analysis_executor.submit(self._analyze_document, document)
        )

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

    def _invoke(self, messages: list[BaseMessage]) -> BaseMessage:
        llm = self._ensure_llm()
        response = llm.invoke(messages, max_tokens=self.max_new_tokens)
        if not isinstance(response, BaseMessage):
            raise RuntimeError("Chat model returned an unexpected response type")
        return response

    def _analyze_document(self, document: str) -> list[BaseMessage]:
        reference_document = self._reference_document(document)
        prompt_text = self._analysis_prompt_text(reference_document)
        messages: list[BaseMessage] = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=prompt_text),
        ]
        response = self._invoke(messages)
        return [*messages, response]

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

    def _write_debug_audio(self, audio: np.ndarray) -> Path | None:
        if self._debug_audio_dir is None:
            return None
        audio_dir = self._debug_audio_dir
        audio_dir.mkdir(parents=True, exist_ok=True)
        with self._debug_audio_lock:
            file_index = self._debug_audio_counter
            self._debug_audio_counter += 1
        wav_bytes = base64.b64decode(_encode_wav_base64(audio, self.sample_rate))
        output_path = audio_dir / f"gemma4_fragment_{file_index:04d}.wav"
        output_path.write_bytes(wav_bytes)
        return output_path

    def _write_debug_conversation(
        self,
        audio_path: Path | None,
        messages: list[BaseMessage],
        response: BaseMessage,
    ) -> None:
        if audio_path is None:
            return
        payload = {
            "model": self.model_name,
            "max_tokens": self.max_new_tokens,
            "messages": messages_to_dict(messages),
            "response": message_to_dict(response),
        }
        audio_path.with_suffix(".json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    def transcribe_audio(self, audio: np.ndarray) -> str:
        waveform = _coerce_float32_mono(audio)
        max_samples = int(round(self.max_audio_seconds * self.sample_rate))
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]
        audio_path = self._write_debug_audio(waveform)

        analysis_conversation = self._analysis_future.result()
        messages = [*analysis_conversation, self._audio_message(waveform)]
        response = self._invoke(messages)
        self._write_debug_conversation(audio_path, messages, response)
        return _extract_text(response)


__all__ = ["AudioWebSocketBuffer", "Gemma4AsrReceiver"]
