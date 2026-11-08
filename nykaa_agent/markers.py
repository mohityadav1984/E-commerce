"""Tool results carry a one-line marker so downstream agents / code can parse them
without guessing at formatting (stdlib only)."""
from __future__ import annotations

import json

RAG = "[[RAG_RESULT]]"
ORDER = "[[ORDER_RESULT]]"


def encode(marker: str, payload: dict) -> str:
    return f"{marker} {json.dumps(payload, separators=(',', ':'))}"


def decode(text: str, marker: str) -> dict | None:
    """Find the LAST occurrence of `marker` in text and parse the JSON after it."""
    idx = text.rfind(marker)
    if idx == -1:
        return None
    rest = text[idx + len(marker):]
    start = rest.find("{")
    if start == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(rest[start:])
        return obj
    except ValueError:
        return None
