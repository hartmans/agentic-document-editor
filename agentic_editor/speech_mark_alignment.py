from __future__ import annotations

from typing import List, Sequence

from agentic_editor.speech_marks import SpeechMark


def speech_marks_from_records(
    records: Sequence[dict], chunk_text: str
) -> List[SpeechMark]:
    """Align provider word records against chunk text."""
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


__all__ = ["speech_marks_from_records"]
