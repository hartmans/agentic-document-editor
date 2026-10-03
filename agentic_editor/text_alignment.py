"""Best-effort correspondence between document text and filtered speech.

Extracted from the old TTS cache's token reconciliation. Audio alignment is
owned by radio_drama; this module only matches the two versions of the text.
"""

from dataclasses import dataclass
from difflib import SequenceMatcher
import math
import re

from .speech_marks import SpeechMark


_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


@dataclass(frozen=True)
class WordCorrespondence:
    spoken_start: int
    spoken_end: int
    original_start: int
    original_end: int


def project_word_marks(
    words: list[WordCorrespondence], starts: dict[int, float | None],
) -> list[SpeechMark]:
    """Project known spoken-boundary seconds to chunk-local document spans."""
    marks = []
    for word in words:
        time = starts.get(word.spoken_start)
        if time is not None and math.isfinite(time):
            marks.append(SpeechMark(
                time_ms=max(0, round(time * 1000)),
                chunk_char_start=word.original_start, chunk_char_end=word.original_end,
            ))
    return sorted(marks, key=lambda mark: mark.time_ms)


def match_filtered_words(original: str, spoken: str) -> list[WordCorrespondence]:
    source = list(_TOKEN_RE.finditer(spoken))
    target = list(_TOKEN_RE.finditer(original))
    if not source or not target:
        return []
    if original == spoken:
        return [WordCorrespondence(m.start(), m.end(), m.start(), m.end()) for m in source]

    # Preserve the old global token alignment and insertion/deletion costs.
    previous = [col * 0.9 for col in range(len(target) + 1)]
    steps = [bytearray(len(target) + 1) for _ in range(len(source) + 1)]
    for col in range(1, len(target) + 1):
        steps[0][col] = 2
    for row, source_word in enumerate(source, 1):
        current = [row * 0.9] + [0.0] * len(target)
        steps[row][0] = 1
        for col, target_word in enumerate(target, 1):
            cost = 1 - SequenceMatcher(
                a=source_word.group().lower(), b=target_word.group().lower()
            ).ratio()
            choices = (previous[col - 1] + cost, previous[col] + 0.9, current[col - 1] + 0.9)
            step = min(range(3), key=choices.__getitem__)
            current[col] = choices[step]
            steps[row][col] = step
        previous = current

    pairs = []
    row, col = len(source), len(target)
    while row or col:
        step = steps[row][col]
        if step == 0:
            s, t = source[row - 1], target[col - 1]
            pairs.append(WordCorrespondence(s.start(), s.end(), t.start(), t.end()))
            row -= 1
            col -= 1
        elif step == 1:
            row -= 1
        else:
            col -= 1
    return list(reversed(pairs))
