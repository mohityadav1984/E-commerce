"""Input-side guardrails (PII masking, prompt-injection) + output-side groundedness.
Stdlib only."""
from __future__ import annotations

import re

# ---------------------------------------------------------------- PII masking
# Fixed-format fields only: phone numbers and card last-4 (name/address are out
# of scope for a keyless masker, per the brief).
_CARD_FULL = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_CARD_LAST4_CONTEXT = re.compile(
    r"(?i)\b((?:card|visa|mastercard|rupay|debit|credit)\b[^.\d]{0,30}?"
    r"(?:ending(?:\s+in|\s+with)?|ends(?:\s+in|\s+with)?|last\s*(?:4|four)(?:\s*digits)?(?:\s*(?:is|are|:))?)\s*)(\d{4})(?!\d)"
)
_CARD_LAST4_BARE = re.compile(r"(?i)\b(ending(?:\s+in|\s+with)?|last\s*(?:4|four)\s*digits?(?:\s*(?:is|are|:))?)\s*(\d{4})(?!\d)")
_CARD_MASKED_STYLE = re.compile(r"(?i)(?:x{4}|\*{4})[\s-]?(\d{4})(?!\d)")
_PHONE = re.compile(r"(?<!\d)(?:\+?91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")


def mask_pii(text: str) -> tuple[str, list[str]]:
    """Return (masked_text, list_of_pii_kinds_found)."""
    fired: list[str] = []

    def sub(pattern, repl, s, kind):
        new, n = pattern.subn(repl, s)
        if n and kind not in fired:
            fired.append(kind)
        return new

    out = text
    out = sub(_CARD_FULL, "[CARD]", out, "card_number")
    out = sub(_CARD_LAST4_CONTEXT, r"\1[CARD_LAST4]", out, "card_last4")
    out = sub(_CARD_LAST4_BARE, r"\1 [CARD_LAST4]", out, "card_last4")
    out = sub(_CARD_MASKED_STYLE, "[CARD_LAST4]", out, "card_last4")
    out = sub(_PHONE, "[PHONE]", out, "phone")
    return out, fired


def contains_raw_pii(text: str) -> bool:
    return bool(mask_pii(text)[1])


# ------------------------------------------------------- prompt-injection
_INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier)\s+(?:instructions|rules|prompts?)",
    r"disregard\s+(?:all\s+|any\s+|the\s+|your\s+)?(?:previous|prior|above|earlier)?\s*(?:instructions|rules|guidelines)",
    r"forget\s+(?:everything|all|your)\s+(?:above|previous|instructions|rules)?",
    r"(?:reveal|show|print|repeat|leak)\s+(?:me\s+)?(?:your|the)\s+(?:system\s+)?(?:prompt|instructions)",
    r"you\s+are\s+now\s+(?:a|an|in)\b",
    r"\bdeveloper\s+mode\b",
    r"\bjailbreak\b",
    r"\bDAN\s+mode\b",
    r"override\s+(?:your|the|all)\s+(?:rules|instructions|safety|guardrails)",
    r"pretend\s+(?:you\s+are|to\s+be)\b",
]
_INJECTION_RE = [re.compile(p, re.I) for p in _INJECTION_PATTERNS]


def detect_injection(text: str) -> list[str]:
    return [p.pattern for p in _INJECTION_RE if p.search(text)]


# ------------------------------------------------- output-side groundedness
_STOP = set("""a an the is are was were be been of to in on for and or if it its this that
with as at by from do does did can could i my me you your we our they their what when how
which who will would should not no yes about than then so such any all""".split())


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9][a-z0-9\-]*", text.lower()) if w not in _STOP}


def support_ratio(sentence: str, context: str) -> float:
    """Fraction of a sentence's content words that appear in the context."""
    words = content_words(sentence)
    if not words:
        return 1.0
    return len(words & content_words(context)) / len(words)


def check_output_grounded(top_score: float, threshold: float) -> tuple[bool, str]:
    """Refuse to answer when retrieval similarity says the KB doesn't cover the question."""
    if top_score < threshold:
        return False, f"top retrieval score {top_score:.3f} < threshold {threshold:.3f}"
    return True, "ok"
