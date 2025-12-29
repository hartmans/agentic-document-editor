from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from pydantic import BaseModel

from .speech_marks import SpeechMark
from .tts_cache import TtsCache

@dataclass
class DocumentUpdate:
    offset: int
    length: int
    new_text: str


@dataclass(frozen=True)
class SearchTextConfig:
    radius: int = 400
    context_chars: int = 60
    max_candidates: int = 5


class SearchTextCandidate(BaseModel):
    before: str
    target: str
    after: str
    span_text: str
    span_start: int 
    span_end: int


@dataclass(frozen=True)
class SearchTextResult:
    index: int | None
    candidates: list[SearchTextCandidate]
    reason: str

    def render_error(self)-> str:
        candidates = [c.model_dump() for c in self.candidates]
        return f'Unable to find the text you pointed to: {self.reason}. Call the tool again using the before/after text from one of these candidates: {candidates}'

@dataclass
class TextChunk:
    """Single contiguous span of the original document."""

    text: str
    doc_char_start: int
    doc_char_end: int
    audio_mime_type: str = "audio/mpeg"


class ResumePoint(BaseModel):
    chunkId: int
    offsetMs: int
    docCharPosition: int


@dataclass
class DocumentState:
    """In-memory representation of the preprocessed document."""

    document:str #: Original text of the document
    voice_id: str
    tts_cache: TtsCache
    chunks: List[TextChunk] = field(default_factory=list)
    aws_session: object | None = None  # boto3.Session | None, kept loose to avoid hard dep
    region_name: str | None = None

    def find_chunk_id(self, position: int) -> int:
        if not self.chunks:
            return 0
        if position <= self.chunks[0].doc_char_start:
            return 0
        for idx, chunk in enumerate(self.chunks):
            if chunk.doc_char_start <= position < chunk.doc_char_end:
                return idx
        return len(self.chunks) - 1

    def search_text(
        self,
        before: str,
        target: str,
        after: str,
        *,
        config: SearchTextConfig | None = None,
    ) -> SearchTextResult:
        """
        Locate a unique target occurrence constrained by nearby before/after context.
        """
        cfg = config or SearchTextConfig()
        document = self.document

        if not document:
            return SearchTextResult(index=None, candidates=[], reason="empty_document")

        target_indices = _find_all(document, target) if target else []

        if target_indices:
            matches = _matching_targets(
                document, target, before, after, cfg.radius, target_indices
            )
            if len(matches) == 1:
                return SearchTextResult(
                    index=matches[0], candidates=[], reason="unique_match"
                )
            candidates = _unique_candidates_for_targets(
                document,
                target,
                matches if matches else target_indices,
                before,
                after,
                cfg,
                target_indices,
            )
            reason = "ambiguous" if matches else "no_match"
            return SearchTextResult(index=None, candidates=candidates, reason=reason)

        candidates = _unique_candidates_without_target(document, before, after, cfg)
        return SearchTextResult(index=None, candidates=candidates, reason="no_match")


def _find_all(haystack: str, needle: str) -> list[int]:
    if not needle:
        return []
    indices: list[int] = []
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            break
        indices.append(idx)
        start = idx + 1
    return indices


def _matching_targets(
    document: str,
    target: str,
    before: str,
    after: str,
    radius: int,
    target_indices: list[int] | None = None,
) -> list[int]:
    if not target:
        return []
    indices = target_indices if target_indices is not None else _find_all(document, target)
    matches: list[int] = []
    for idx in indices:
        if before and _has_before_within_radius(document, before, idx, radius) is False:
            continue
        if after and _has_after_within_radius(
            document, after, idx + len(target), radius
        ) is False:
            continue
        matches.append(idx)
    return matches


def _has_before_within_radius(
    document: str, needle: str, target_start: int, radius: int
) -> bool:
    if not needle:
        return True
    if target_start <= 0:
        return False
    window_start = max(0, target_start - radius)
    latest_start = target_start - len(needle)
    if latest_start < 0:
        return False
    search_end = latest_start + len(needle)
    search_start = max(0, window_start - len(needle))
    idx = document.rfind(needle, search_start, search_end)
    if idx == -1:
        return False
    return idx + len(needle) >= window_start


def _has_after_within_radius(
    document: str, needle: str, target_end: int, radius: int
) -> bool:
    if not needle:
        return True
    if target_end >= len(document):
        return False
    limit = target_end + radius
    idx = document.find(needle, target_end)
    return idx != -1 and idx < limit


def _unique_candidates_for_targets(
    document: str,
    target: str,
    indices: list[int],
    before_hint: str,
    after_hint: str,
    cfg: SearchTextConfig,
    target_indices: list[int],
) -> list[SearchTextCandidate]:
    candidates: list[SearchTextCandidate] = []
    seen: set[tuple[int, int, str]] = set()
    for idx in indices:
        candidate = _build_unique_candidate_for_target(
            document,
            target,
            idx,
            before_hint,
            after_hint,
            cfg,
            target_indices,
        )
        if candidate:
            _add_candidate(candidates, seen, candidate, cfg.max_candidates)
        if len(candidates) >= cfg.max_candidates:
            break
    return candidates


def _unique_candidates_without_target(
    document: str,
    before: str,
    after: str,
    cfg: SearchTextConfig,
) -> list[SearchTextCandidate]:
    candidates: list[SearchTextCandidate] = []
    seen: set[tuple[int, int, str]] = set()
    before_indices = _find_all(document, before) if before else []
    after_indices = _find_all(document, after) if after else []

    if before_indices and after_indices:
        _candidates_from_before_after(
            document,
            before,
            after,
            before_indices,
            after_indices,
            cfg,
            candidates,
            seen,
        )
    elif before_indices:
        for before_idx in before_indices:
            candidate = _candidate_from_before_only(
                document,
                before,
                after,
                before_idx,
                cfg,
            )
            if candidate:
                _add_candidate(candidates, seen, candidate, cfg.max_candidates)
            if len(candidates) >= cfg.max_candidates:
                break
    elif after_indices:
        for after_idx in after_indices:
            candidate = _candidate_from_after_only(
                document,
                before,
                after,
                after_idx,
                cfg,
            )
            if candidate:
                _add_candidate(candidates, seen, candidate, cfg.max_candidates)
            if len(candidates) >= cfg.max_candidates:
                break

    return candidates


def _build_unique_candidate_for_target(
    document: str,
    target: str,
    target_start: int,
    before_hint: str,
    after_hint: str,
    cfg: SearchTextConfig,
    target_indices: list[int],
) -> SearchTextCandidate | None:
    target_end = target_start + len(target)
    left_full = document[:target_start]
    right_full = document[target_end:]

    left_len = _initial_before_len(left_full, before_hint, cfg)
    right_len = _initial_after_len(right_full, after_hint, cfg)

    left_len = min(left_len, len(left_full))
    right_len = min(right_len, len(right_full))

    left_max = len(left_full)
    right_max = len(right_full)

    while True:
        before_text = left_full[-left_len:] if left_len else ""
        after_text = right_full[:right_len] if right_len else ""
        matches = _matching_targets(
            document, target, before_text, after_text, cfg.radius, target_indices
        )
        if len(matches) == 1 and matches[0] == target_start:
            return _make_candidate(
                document,
                before_text,
                target,
                after_text,
                target_start,
                target_end,
            )
        if left_len >= left_max and right_len >= right_max:
            return None
        if left_len < left_max:
            left_len += 1
        if right_len < right_max:
            right_len += 1


def _initial_before_len(left_full: str, before_hint: str, cfg: SearchTextConfig) -> int:
    if before_hint:
        pos = left_full.rfind(before_hint)
        if pos != -1:
            return len(left_full) - pos
    return min(cfg.context_chars, len(left_full))


def _initial_after_len(right_full: str, after_hint: str, cfg: SearchTextConfig) -> int:
    if after_hint:
        pos = right_full.find(after_hint)
        if pos != -1:
            return pos + len(after_hint)
    return min(cfg.context_chars, len(right_full))


def _make_candidate(
    document: str,
    before_text: str,
    target_text: str,
    after_text: str,
    target_start: int,
    target_end: int,
) -> SearchTextCandidate:
    span_start = max(0, target_start - len(before_text))
    span_end = min(len(document), target_end + len(after_text))
    return SearchTextCandidate(
        before=before_text,
        target=target_text,
        after=after_text,
        span_text=document[span_start:span_end],
        span_start=span_start,
        span_end=span_end,
    )


def _candidates_from_before_after(
    document: str,
    before: str,
    after: str,
    before_indices: list[int],
    after_indices: list[int],
    cfg: SearchTextConfig,
    candidates: list[SearchTextCandidate],
    seen: set[tuple[int, int, str]],
) -> None:
    after_pos = 0
    for before_start in before_indices:
        before_end = before_start + len(before)
        while after_pos < len(after_indices) and after_indices[after_pos] < before_end:
            after_pos += 1
        scan_pos = after_pos
        while scan_pos < len(after_indices):
            after_start = after_indices[scan_pos]
            if after_start > before_end + cfg.radius:
                break
            candidate = _candidate_from_before_after(
                document,
                before,
                after,
                before_start,
                after_start,
                cfg,
            )
            if candidate:
                _add_candidate(candidates, seen, candidate, cfg.max_candidates)
            if len(candidates) >= cfg.max_candidates:
                return
            scan_pos += 1


def _candidate_from_before_after(
    document: str,
    before: str,
    after: str,
    before_start: int,
    after_start: int,
    cfg: SearchTextConfig,
) -> SearchTextCandidate | None:
    before_text = document[before_start : before_start + len(before)] if before else ""
    before_end = before_start + len(before_text)
    after_text = document[after_start : after_start + len(after)] if after else ""
    after_end = after_start + len(after_text)

    if before_end < after_start:
        target_start = before_end
        target_end = min(after_start, target_start + cfg.context_chars)
    else:
        target_end = after_start
        target_start = max(0, target_end - cfg.context_chars)

    target_start, target_end = _ensure_non_empty_target(
        document,
        target_start,
        target_end,
        fallback_start=before_end,
    )

    target_text = document[target_start:target_end]

    unique_candidate = _build_unique_candidate_for_target(
        document,
        target_text,
        target_start,
        before,
        after,
        cfg,
        _find_all(document, target_text),
    )
    if unique_candidate:
        return unique_candidate

    return None


def _candidate_from_before_only(
    document: str,
    before: str,
    after: str,
    before_start: int,
    cfg: SearchTextConfig,
) -> SearchTextCandidate | None:
    before_text = document[before_start : before_start + len(before)] if before else ""
    before_end = before_start + len(before_text)
    target_start = before_end
    target_end = min(len(document), target_start + cfg.context_chars)
    target_start, target_end = _ensure_non_empty_target(
        document,
        target_start,
        target_end,
        fallback_start=before_end,
    )
    target_text = document[target_start:target_end]

    unique_candidate = _build_unique_candidate_for_target(
        document,
        target_text,
        target_start,
        before,
        after,
        cfg,
        _find_all(document, target_text),
    )
    if unique_candidate:
        return unique_candidate

    return None


def _candidate_from_after_only(
    document: str,
    before: str,
    after: str,
    after_start: int,
    cfg: SearchTextConfig,
) -> SearchTextCandidate | None:
    after_text = document[after_start : after_start + len(after)] if after else ""
    target_end = after_start
    target_start = max(0, target_end - cfg.context_chars)
    target_start, target_end = _ensure_non_empty_target(
        document,
        target_start,
        target_end,
        fallback_end=after_start,
    )
    target_text = document[target_start:target_end]

    unique_candidate = _build_unique_candidate_for_target(
        document,
        target_text,
        target_start,
        before,
        after,
        cfg,
        _find_all(document, target_text),
    )
    if unique_candidate:
        return unique_candidate

    return None


def _ensure_non_empty_target(
    document: str,
    target_start: int,
    target_end: int,
    *,
    fallback_start: int | None = None,
    fallback_end: int | None = None,
) -> tuple[int, int]:
    if target_end > target_start:
        return target_start, target_end
    doc_len = len(document)
    if fallback_start is not None:
        target_start = fallback_start
    if fallback_end is not None:
        target_end = fallback_end
    if target_end <= target_start:
        if target_start < doc_len:
            target_end = min(doc_len, target_start + 1)
        elif target_start > 0:
            target_start = max(0, target_start - 1)
            target_end = target_start + 1
    return target_start, target_end


def _add_candidate(
    candidates: list[SearchTextCandidate],
    seen: set[tuple[int, int, str]],
    candidate: SearchTextCandidate,
    max_candidates: int,
) -> None:
    if len(candidates) >= max_candidates:
        return
    key = (candidate.span_start, candidate.span_end, candidate.target)
    if key in seen:
        return
    seen.add(key)
    candidates.append(candidate)


def chunk_documents(document: str) -> List[tuple[int, int]]:
    """
    Split the document into contiguous spans, preferring paragraph breaks.

    Returns a list of (start, end) offsets into the original document. 
    """
    import re

    if document.strip() == "":
        return []

    min_chunk_size = 1000
    max_chunk_size = 2000
    separator = re.compile(r"(?:\r?\n\s*){2,}")
    break_ends = [match.end() for match in separator.finditer(document)]
    spans: List[tuple[int, int]] = []
    cursor = 0
    break_idx = 0
    doc_len = len(document)

    while cursor < doc_len:
        remaining = doc_len - cursor
        if remaining <= max_chunk_size:
            spans.append((cursor, doc_len))
            break

        window_start = cursor + min_chunk_size
        window_end = min(cursor + max_chunk_size, doc_len)

        while break_idx < len(break_ends) and break_ends[break_idx] < window_start:
            break_idx += 1

        candidate_end = None
        scan_idx = break_idx
        while scan_idx < len(break_ends) and break_ends[scan_idx] <= window_end:
            candidate_end = break_ends[scan_idx]
            scan_idx += 1
        break_idx = scan_idx

        if candidate_end is not None:
            end = candidate_end
        else:
            search_start = cursor + max_chunk_size
            if search_start >= doc_len:
                end = doc_len
            else:
                whitespace_match = re.search(r"\s", document[search_start:])
                if whitespace_match:
                    end = search_start + whitespace_match.start()
                    while end < doc_len and document[end].isspace():
                        end += 1
                else:
                    end = doc_len

        if doc_len - end < min_chunk_size:
            end = doc_len

        if end <= cursor:
            end = min(doc_len, cursor + max_chunk_size)

        spans.append((cursor, end))
        cursor = end

    return spans


def apply_document_update(update: str, state: DocumentState) -> DocumentState:
    """Return a new DocumentState after applying a document update."""
    updated_document = update
    from agentic_editor.tts import build_document_state

    return build_document_state(
        document_text=updated_document,
        tts_cache=state.tts_cache,
        session=state.aws_session,  # type: ignore
    )


def load_document(path: Path) -> str:
    """Read a UTF-8 text document from disk."""
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Document path not found or unreadable: {path}")
    return path.read_text(encoding="utf-8")


__all__ = [
    "SpeechMark",
    "TextChunk",
    "ResumePoint",
    "DocumentState",
    "SearchTextConfig",
    "SearchTextCandidate",
    "SearchTextResult",
    "chunk_documents",
    'apply_document_update',
    "load_document",
]
