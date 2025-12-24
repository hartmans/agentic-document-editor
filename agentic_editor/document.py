from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from .speech_marks import SpeechMark
from .tts_cache import TtsCache

@dataclass
class DocumentUpdate:
    offset: int
    length: int
    new_text: str


@dataclass
class TextChunk:
    """Single contiguous span of the original document."""

    text: str
    doc_char_start: int
    doc_char_end: int
    audio_mime_type: str = "audio/mpeg"


@dataclass
class DocumentState:
    """In-memory representation of the preprocessed document."""

    document:str #: Original text of the document
    voice_id: str
    tts_cache: TtsCache
    chunks: List[TextChunk] = field(default_factory=list)
    aws_session: object | None = None  # boto3.Session | None, kept loose to avoid hard dep
    region_name: str | None = None


def chunk_documents(document: str) -> List[tuple[int, int]]:
    """
    Split the document into contiguous spans separated by one or more blank lines.

    Returns a list of (start, end) offsets into the original document.
    """
    import re

    separator = re.compile(r"(?:\r?\n\s*){2,}")
    spans: List[tuple[int, int]] = []
    cursor = 0

    for match in separator.finditer(document):
        start = cursor
        end = match.start()
        cursor = match.end()
        if start == end:
            continue
        text = document[start:end]
        if text.strip() == "":
            continue
        spans.append((start, end))

    if cursor < len(document):
        trailing = document[cursor:]
        if trailing.strip() != "":
            spans.append((cursor, len(document)))

    # If the document was entirely whitespace or separators, fall back to one chunk.
    if not spans and document.strip() != "":
        spans.append((0, len(document)))
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
    "DocumentState",
    "chunk_documents",
    'apply_document_update',
    "load_document",
]
