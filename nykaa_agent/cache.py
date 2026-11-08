"""In-memory response cache keyed by normalized query text (stdlib only)."""
from __future__ import annotations

import copy
import re
import string

_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.lower().translate(_PUNCT)).strip()


class ResponseCache:
    def __init__(self):
        self._store: dict[str, dict] = {}
        self.hits = 0
        self.misses = 0

    def key(self, query: str, *parts) -> str:
        return "|".join([normalize_query(query), *map(str, parts)])

    def get(self, key: str):
        if key in self._store:
            self.hits += 1
            return copy.deepcopy(self._store[key])
        self.misses += 1
        return None

    def put(self, key: str, value: dict) -> None:
        self._store[key] = copy.deepcopy(value)

    def clear(self) -> None:
        self._store.clear()   # called after /add-document so answers never go stale
