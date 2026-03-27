from __future__ import annotations

import hashlib
import io
import os
import subprocess
from pathlib import Path
from threading import Lock, RLock
from typing import Any, List, Sequence

import boto3
import numpy as np
import soundfile as sf
import torch
import whisperx

from agentic_editor.poly_tts_cache import _speech_marks_from_records
from agentic_editor.speech_marks import SpeechMark
from agentic_editor.tss_cache import AbstractTtsCache

_QWEN_MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
_QWEN_LANGUAGE = "English"
_WHISPERX_LANGUAGE = "en"
_WHISPERX_MODEL = "large-v3"
_WHISPERX_SAMPLE_RATE = 16000


class QwenTtsCache(AbstractTtsCache):
    """
    Driver-backed cache for local Qwen TTS synthesis with WhisperX speech marks.

    Externally this still behaves like the Polly cache: `__getitem__` returns
    `(audio_bytes, speech_marks)` where audio_bytes are MP3 payloads suitable
    for the existing playback path.
    """

    driver_name = "qwen"
    audio_extension = "mp3"

    def __init__(
        self,
        config: dict[str, Any],
        boto_session: boto3.session.Session | None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, boto_session, prefix=prefix)

        voice_sample = self._get_config("voice_sample")
        if voice_sample is None:
            raise ValueError("Qwen TTS requires a configured voice_sample path")

        self.voice_sample = Path(str(voice_sample)).expanduser().resolve()
        voice_id = self._get_config("voice_id")
        region_name = self._get_config("region_name")
        model_name = self._get_config("model_name")
        language = self._get_config("language")
        device = self._get_config("device")
        whisperx_device = self._get_config("whisperx_device")
        whisperx_model = self._get_config("whisperx_model")
        whisperx_language = self._get_config("whisperx_language")
        prompt_cache_dir = self._get_config("prompt_cache_dir")
        ref_text = self._get_config("ref_text")
        x_vector_only_mode = self._get_config("x_vector_only_mode")

        self.voice_id = str(voice_id) if voice_id is not None else self.voice_sample.stem
        self.region_name = str(region_name) if region_name is not None else "us-east-1"
        self.session = boto_session or boto3.Session()

        self.model_name = str(model_name) if model_name is not None else _QWEN_MODEL_NAME
        self.language = str(language) if language is not None else _QWEN_LANGUAGE
        self.device = self._normalize_device(str(device) if device is not None else "cuda")
        self.whisperx_device = self._normalize_device(
            str(whisperx_device) if whisperx_device is not None else self.device
        )
        self.whisperx_model_name = (
            str(whisperx_model) if whisperx_model is not None else _WHISPERX_MODEL
        )
        self.whisperx_language = (
            str(whisperx_language) if whisperx_language is not None else _WHISPERX_LANGUAGE
        )
        self.ref_text = str(ref_text) if ref_text is not None else None
        self.x_vector_only_mode = (
            bool(x_vector_only_mode) if x_vector_only_mode is not None else False
        )

        if prompt_cache_dir is None:
            prompt_cache_dir = self.cache_dir / "qwen_prompt_cache"
        self.prompt_cache_dir = Path(str(prompt_cache_dir)).expanduser()
        self.prompt_cache_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessed_sample_dir = self.prompt_cache_dir / "samples"
        self.preprocessed_sample_dir.mkdir(parents=True, exist_ok=True)

        self._load_lock = RLock()
        self._prompt_cache_lock = Lock()
        self._model = None
        self._voice_clone_prompt_item_type = None
        self._sample_rate: int | None = None
        self._asr_model = None
        self._align_model = None
        self._align_metadata = None

    def _key_parts(self, text: str) -> Sequence[str]:
        return (
            self.driver_name,
            self.model_name,
            str(self.voice_id),
            self._voice_sample_cache_key(),
            str(self.language),
            str(self.ref_text or ""),
            str(self.x_vector_only_mode),
            str(self.filter or ""),
            text,
        )

    def _synthesize_entry(
        self,
        text: str,
        synthesis_text: str,
    ) -> tuple[bytes, List[SpeechMark]]:
        prompt_items = self._prompt_items_for_voice()
        audio, sample_rate = self._generate_audio(synthesis_text, prompt_items)
        speech_marks = self._align_speech_marks(synthesis_text, audio, sample_rate)
        audio_bytes = self._encode_mp3(audio, sample_rate)
        return audio_bytes, speech_marks

    def _prompt_items_for_voice(self) -> list[object]:
        prompt_path = self._prompt_cache_path()
        with self._prompt_cache_lock:
            if prompt_path.is_file():
                return self._deserialize_prompt_items(
                    torch.load(prompt_path, map_location="cpu")
                )

        prompt_items = self._build_prompt_items_for_voice()
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._serialize_prompt_items(prompt_items), prompt_path)
        return prompt_items

    def _build_prompt_items_for_voice(self) -> list[object]:
        model = self._ensure_loaded()
        sample_path = self._preprocessed_voice_sample_path()
        ref_text = self.ref_text
        if not self.x_vector_only_mode and not ref_text:
            ref_text = self._transcribe_voice_sample(sample_path)
        return list(
            model.create_voice_clone_prompt(
                ref_audio=str(sample_path),
                ref_text=ref_text,
                x_vector_only_mode=self.x_vector_only_mode,
            )
        )

    def _generate_audio(
        self,
        text: str,
        prompt_items: Sequence[object],
    ) -> tuple[np.ndarray, int]:
        model = self._ensure_loaded()
        wavs, sample_rate = model.generate_voice_clone(
            text=[text],
            language=[self.language],
            voice_clone_prompt=list(prompt_items),
            non_streaming_mode=True,
        )
        if not wavs:
            raise RuntimeError("Qwen returned no audio")
        audio = self._normalize_audio_array(wavs[0])
        if self._sample_rate is None:
            self._sample_rate = int(sample_rate)
        return audio, int(sample_rate)

    def _transcribe_voice_sample(self, sample_path: Path) -> str:
        transcript_path = self._transcript_cache_path()
        if transcript_path.is_file():
            return transcript_path.read_text(encoding="utf-8")

        audio = self._load_audio_path(sample_path)
        asr_model = self._ensure_asr_model()
        transcription = asr_model.transcribe(
            audio,
            batch_size=10,
            language=self.whisperx_language,
        )
        transcript = " ".join(
            segment["text"].strip()
            for segment in transcription.get("segments", [])
            if segment.get("text")
        ).strip()
        if not transcript:
            raise RuntimeError("WhisperX returned an empty transcript for the voice sample")
        transcript_path.write_text(transcript, encoding="utf-8")
        return transcript

    def _align_speech_marks(
        self,
        text: str,
        audio: np.ndarray,
        sample_rate: int,
    ) -> List[SpeechMark]:
        aligned_audio = self._resample_for_whisperx(audio, sample_rate)
        duration_seconds = (
            len(aligned_audio) / _WHISPERX_SAMPLE_RATE if len(aligned_audio) else 0.0
        )
        transcript = [{"start": 0.0, "end": duration_seconds, "text": text}]
        align_model, metadata = self._ensure_align_model()
        aligned = whisperx.align(
            transcript,
            align_model,
            metadata,
            aligned_audio,
            self.whisperx_device,
            return_char_alignments=False,
        )
        records: list[dict[str, object]] = []
        for word in aligned.get("word_segments", []):
            word_text = str(word.get("word", "")).strip()
            start = word.get("start")
            if not word_text or start is None:
                continue
            records.append(
                {
                    "type": "word",
                    "text": word_text,
                    "time_ms": int(round(float(start) * 1000)),
                }
            )
        return _speech_marks_from_records(records, text)

    def _serialize_prompt_items(self, prompt_items: Sequence[object]) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for item in prompt_items:
            payload.append(
                {
                    "ref_code": (
                        item.ref_code.detach().cpu()  # type: ignore[attr-defined]
                        if torch.is_tensor(item.ref_code)  # type: ignore[attr-defined]
                        else None
                    ),
                    "ref_spk_embedding": item.ref_spk_embedding.detach().cpu(),  # type: ignore[attr-defined]
                    "x_vector_only_mode": bool(item.x_vector_only_mode),  # type: ignore[attr-defined]
                    "icl_mode": bool(item.icl_mode),  # type: ignore[attr-defined]
                    "ref_text": item.ref_text,  # type: ignore[attr-defined]
                }
            )
        return payload

    def _deserialize_prompt_items(self, payload: Sequence[dict[str, Any]]) -> list[object]:
        prompt_type = self._voice_clone_prompt_item_cls()
        return [
            prompt_type(
                ref_code=entry["ref_code"],
                ref_spk_embedding=entry["ref_spk_embedding"],
                x_vector_only_mode=bool(entry["x_vector_only_mode"]),
                icl_mode=bool(entry["icl_mode"]),
                ref_text=entry.get("ref_text"),
            )
            for entry in payload
        ]

    def _voice_clone_prompt_item_cls(self):
        with self._load_lock:
            if self._voice_clone_prompt_item_type is None:
                os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
                from qwen_tts import VoiceClonePromptItem

                self._voice_clone_prompt_item_type = VoiceClonePromptItem
            return self._voice_clone_prompt_item_type

    def _ensure_loaded(self):
        with self._load_lock:
            if self._model is not None:
                return self._model

            os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
            from qwen_tts import Qwen3TTSModel

            dtype, attn_implementation = self._load_settings_for_device(self.device)
            try:
                model = Qwen3TTSModel.from_pretrained(
                    self.model_name,
                    device_map=self._device_map_for_device(self.device),
                    dtype=dtype,
                    attn_implementation=attn_implementation,
                )
            except Exception:
                if attn_implementation != "flash_attention_2":
                    raise
                model = Qwen3TTSModel.from_pretrained(
                    self.model_name,
                    device_map=self._device_map_for_device(self.device),
                    dtype=dtype,
                    attn_implementation=None,
                )
            self._sample_rate = int(getattr(model.model.config, "sample_rate", 24000))
            self._model = model
            return model

    def _ensure_asr_model(self):
        with self._load_lock:
            if self._asr_model is not None:
                return self._asr_model
            self._asr_model = whisperx.load_model(
                self.whisperx_model_name,
                self.whisperx_device,
                compute_type="default",
                language=self.whisperx_language,
            )
            return self._asr_model

    def _ensure_align_model(self):
        with self._load_lock:
            if self._align_model is not None and self._align_metadata is not None:
                return self._align_model, self._align_metadata
            self._align_model, self._align_metadata = whisperx.load_align_model(
                language_code=self.whisperx_language,
                device=self.whisperx_device,
            )
            return self._align_model, self._align_metadata

    def _prompt_cache_path(self) -> Path:
        return self.prompt_cache_dir / f"{self._voice_sample_cache_key()}.pt"

    def _transcript_cache_path(self) -> Path:
        return self.prompt_cache_dir / f"{self._voice_sample_cache_key()}.txt"

    def _preprocessed_voice_sample_path(self) -> Path:
        target = self.preprocessed_sample_dir / f"{self._voice_sample_cache_key()}.wav"
        if target.is_file():
            return target
        model = self._ensure_loaded()
        target_sample_rate = int(
            getattr(model.model, "speaker_encoder_sample_rate", 24000)
        )
        self._ffmpeg_convert_path(
            self.voice_sample,
            target,
            sample_rate=target_sample_rate,
            codec="pcm_s16le",
            fmt="wav",
        )
        return target

    def _voice_sample_cache_key(self) -> str:
        stat = self.voice_sample.stat()
        h = hashlib.sha256()
        h.update(str(self.voice_sample).encode("utf-8"))
        h.update(b"\0")
        h.update(str(stat.st_mtime_ns).encode("utf-8"))
        h.update(b"\0")
        h.update(str(stat.st_size).encode("utf-8"))
        return h.hexdigest()

    def _encode_mp3(self, audio: np.ndarray, sample_rate: int) -> bytes:
        return self._ffmpeg_convert_array(
            audio,
            sample_rate=sample_rate,
            output_args=["-codec:a", "libmp3lame", "-b:a", "192k", "-f", "mp3", "pipe:1"],
        )

    def _resample_for_whisperx(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        wav_bytes = self._ffmpeg_convert_array(
            audio,
            sample_rate=sample_rate,
            output_args=[
                "-ar",
                str(_WHISPERX_SAMPLE_RATE),
                "-ac",
                "1",
                "-codec:a",
                "pcm_s16le",
                "-f",
                "wav",
                "pipe:1",
            ],
        )
        return self._load_audio_bytes(wav_bytes)

    def _ffmpeg_convert_array(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
        output_args: Sequence[str],
    ) -> bytes:
        audio = self._normalize_audio_array(audio)
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "f32le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            *output_args,
        ]
        proc = subprocess.run(
            cmd,
            input=np.ascontiguousarray(audio, dtype=np.float32).tobytes(),
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode('utf-8', 'replace')}")
        return proc.stdout

    def _ffmpeg_convert_path(
        self,
        source_path: Path,
        target_path: Path,
        *,
        sample_rate: int,
        codec: str,
        fmt: str,
    ) -> None:
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(source_path),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-codec:a",
            codec,
            "-f",
            fmt,
            str(target_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode('utf-8', 'replace')}")

    def _load_audio_path(self, path: Path) -> np.ndarray:
        audio, _ = sf.read(str(path), dtype="float32", always_2d=False)
        return self._normalize_audio_array(audio)

    def _load_audio_bytes(self, wav_bytes: bytes) -> np.ndarray:
        audio, _ = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
        return self._normalize_audio_array(audio)

    def _normalize_audio_array(self, audio: torch.Tensor | np.ndarray) -> np.ndarray:
        if torch.is_tensor(audio):
            array = audio.detach().float().cpu().numpy()
        else:
            array = np.asarray(audio, dtype=np.float32)
        array = np.squeeze(array)
        if array.ndim != 1:
            raise ValueError(f"Expected mono audio, got {array.shape!r}")
        return np.ascontiguousarray(array, dtype=np.float32)

    def _normalize_device(self, device: str) -> str:
        normalized = (device or "cpu").lower()
        if normalized == "cuda" and not torch.cuda.is_available():
            return "cpu"
        if normalized == "mps":
            return "cpu"
        if normalized not in {"cuda", "cpu"}:
            raise ValueError(f"Unsupported device: {device}")
        return normalized

    def _load_settings_for_device(self, device: str) -> tuple[torch.dtype, str | None]:
        if device == "cuda":
            return torch.bfloat16, "flash_attention_2"
        return torch.float32, None

    def _device_map_for_device(self, device: str) -> str:
        if device == "cuda":
            return "cuda:0"
        return "cpu"


__all__ = ["QwenTtsCache"]
