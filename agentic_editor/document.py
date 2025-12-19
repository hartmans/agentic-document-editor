from __future__ import annotations

from pathlib import Path
from typing import List, Tuple


def chunk_document_paragraphs(document: str) -> List[tuple[int, int]]:
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


def load_document(path: Path) -> str:
    """Read a UTF-8 text document from disk."""
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Document path not found or unreadable: {path}")
    return path.read_text(encoding="utf-8")
