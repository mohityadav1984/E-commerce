"""JSON-Lines request log. Everything is passed through the SAME PII masker the
model sees before touching disk (Task 12)."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone

from .config import LOG_DIR, REQUEST_LOG
from .guardrails import mask_pii

_LOCK = threading.Lock()


def _sanitize(obj):
    if isinstance(obj, str):
        return mask_pii(obj)[0]
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def log_request(entry: dict) -> None:
    entry = {"ts": datetime.now(timezone.utc).isoformat(), **entry}
    line = json.dumps(_sanitize(entry), ensure_ascii=False)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with _LOCK, open(REQUEST_LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
