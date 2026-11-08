"""Two chunking strategies + document loading (stdlib only)."""
from __future__ import annotations

import re
from pathlib import Path

from .config import FIXED_OVERLAP, FIXED_SIZE, KB_DIR

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def load_documents(kb_dir: Path = KB_DIR) -> list[dict]:
    docs = []
    for path in sorted(Path(kb_dir).glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        title = split_sentences(text)[0].rstrip(".")
        docs.append({"doc_id": path.stem, "title": title, "text": text})
    return docs


def chunk_fixed(text: str, size: int = FIXED_SIZE, overlap: int = FIXED_OVERLAP) -> list[str]:
    """Fixed-size character windows with overlap; the cut snaps back to a space."""
    assert 0 <= overlap < size // 2, "overlap must be < size/2"
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            if space != -1:
                end = space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
        nxt = text.find(" ", start)          # never begin a chunk mid-word
        if nxt != -1 and nxt < end:
            start = nxt + 1
    return chunks


def chunk_sentences(text: str) -> list[str]:
    """Sentence-based chunks: one chunk per sentence."""
    return split_sentences(text)


def chunk_document(doc: dict, strategy: str) -> list[dict]:
    pieces = chunk_fixed(doc["text"]) if strategy == "fixed" else chunk_sentences(doc["text"])
    return [
        {
            "chunk_id": f"{doc['doc_id']}::{strategy}::{i}",
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "strategy": strategy,
            "chunk_index": i,
            "text": piece,
        }
        for i, piece in enumerate(pieces)
    ]
