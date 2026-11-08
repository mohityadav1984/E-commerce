"""Deterministic MOCK_LLM text generation for the grounded-generation step.

The 'model' is extractive: it can ONLY emit sentences that already exist in the
retrieved context, so answers are grounded by construction. Sentences are ranked
by stemmed query-term overlap, sentences about a different product category than
the question are dropped, and the answer is kept short to avoid padding.
A real LLM could be wired in behind MOCK_LLM=false; that path is intentionally
left unimplemented (every acceptance criterion is satisfied under MOCK_LLM)."""
from __future__ import annotations

import re

from .chunking import split_sentences
from .config import MOCK_LLM

FALLBACK = ("I don't know based on the Nykaa support knowledge base. "
            "I can connect you with a human support agent if you'd like.")

CATEGORY_NAMES = {"beauty", "electronics", "apparel", "footwear", "home"}
CATEGORY_TRIGGERS = {
    "beauty": {"serum", "cream", "lipstick", "face", "makeup", "skin", "perfume", "beauty"},
    "electronics": {"laptop", "phone", "electronics", "headphones", "tv"},
    "apparel": {"kurta", "shirt", "dress", "jeans", "apparel", "clothes", "clothing"},
    "footwear": {"sneaker", "sneakers", "shoe", "shoes", "footwear", "sandal", "sandals", "boots"},
    "home": {"kitchen", "appliance", "appliances", "decor", "furniture"},
}
_STOP = set("""a an the is are was were be been of to in on for and or if it its this that
with as at by from do does did can could i my me you your we our they their them it what when
how many much which who will would should not no yes about than then so such any all there here
up want item home still come get also just have has had need order orders regarding nyk""".split())
MIN_SHARED_TERMS = 2   # one shared word (e.g. 'card', 'return') is not evidence
_TIME_INTENT = re.compile(r"\bhow long\b|\bhow many days\b|\bhow soon\b|\bwhen\b", re.I)


def _stem(w: str) -> str:
    """Crude suffix stripper so 'returned'/'returns'/'return' and 'damaged'/'damage' match."""
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"                              # cities -> city
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)]
            break
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "aeiou":
        w = w[:-1]                                       # cancelled -> cancel
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]                                       # damage -> damag
    return w


def _normalize(text: str) -> str:
    t = text.lower()
    return t.replace("cash on delivery", "cod").replace("pick up", "pickup").replace("pin code", "pincode")


def _stems(text: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[a-z0-9]+", _normalize(text))
            if w not in _STOP and len(w) > 1}


def _query_categories(query: str) -> set[str]:
    words = set(re.findall(r"[a-z]+", query.lower()))
    return {c for c, trig in CATEGORY_TRIGGERS.items() if words & trig}


def _sentence_categories(sentence: str) -> set[str]:
    stems = {_stem(w) for w in re.findall(r"[a-z]+", sentence.lower())}
    return {c for c in CATEGORY_NAMES if _stem(c) in stems}


def generate_grounded(query: str, context_chunks: list[str], max_sentences: int = 2) -> str:
    if not MOCK_LLM:
        raise NotImplementedError("Real-LLM mode is not wired in; use MOCK_LLM=true.")
    q_stems = _stems(query)
    qcats = _query_categories(query)
    q_stems |= {_stem(c) for c in qcats}
    if _TIME_INTENT.search(query):
        q_stems.add("day")

    candidates, seen = [], set()
    for chunk in context_chunks:
        for sent in split_sentences(chunk):
            if len(sent.split()) < 6 or sent in seen:    # skip titles and fragments
                continue
            seen.add(sent)
            scats = _sentence_categories(sent)
            if qcats and scats and not (qcats & scats):  # wrong product category
                continue
            score = len(q_stems & _stems(sent))
            if score:
                candidates.append((score, len(candidates), sent))
    if not candidates:
        return FALLBACK

    candidates.sort(key=lambda c: (-c[0], c[1]))
    best = candidates[0][0]
    if best < MIN_SHARED_TERMS:
        return FALLBACK
    chosen = []
    for score, _, sent in candidates:
        if len(chosen) >= max_sentences:
            break
        # top-scoring sentences always; a second-tier sentence only for multi-part questions
        if score >= best or (best >= 3 and score >= max(2, best - 1)):
            chosen.append(sent)
    return " ".join(chosen)
