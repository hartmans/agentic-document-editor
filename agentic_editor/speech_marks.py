from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class SpeechMark(BaseModel):
    """Maps time within an audio stream to a character offset within the chunk."""

    model_config = ConfigDict(extra="ignore")
    time_ms: int
    chunk_char_start: int
    chunk_char_end: int


__all__ = ["SpeechMark"]
