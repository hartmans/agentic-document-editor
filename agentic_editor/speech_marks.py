from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpeechMark:
    """Maps time within an audio stream to a character offset within the chunk."""

    time_ms: int
    chunk_char_start: int
    chunk_char_end: int


__all__ = ["SpeechMark"]
