"""Four-layer governance pieces: least autonomy (Application layer), risk
classification, per-request budget cap (Runtime layer). Stdlib only."""
from __future__ import annotations

import math

from .config import MAX_REQUEST_TOKENS

# ----------------------------------------------------------- Application layer
# Least autonomy: each sensitive tool lists the ONLY agent roles allowed to hold it.
TOOL_ALLOWLIST: dict[str, set[str]] = {
    "check_order_status": {"Lookup Agent"},
}


class LeastAutonomyViolation(PermissionError):
    pass


def authorize_tool(agent_role: str, tool_name: str) -> None:
    allowed = TOOL_ALLOWLIST.get(tool_name)
    if allowed is not None and agent_role not in allowed:
        raise LeastAutonomyViolation(
            f"'{agent_role}' is not allowed to use '{tool_name}' (allowed: {sorted(allowed)})"
        )


def audit_crew_wiring(agents) -> dict[str, list[str]]:
    """Re-check every (agent, tool) pair on an already-built crew."""
    report = {}
    for a in agents:
        names = [t.name for t in (a.tools or [])]
        for n in names:
            authorize_tool(a.role, n)
        report[a.role] = names
    return report


RISK_CLASSIFICATION = "Medium"
RISK_JUSTIFICATION = (
    "The agent is a customer-support assistant, which the Low/Medium/High scheme "
    "places in Medium (customer support tickets). It reads order status and order "
    "value, which touches transactional but not medical, hiring or financial-account "
    "data, so it is not High. It is above Low because a wrong answer can mislead a "
    "customer about refunds or returns, which is why it uses grounded retrieval, a "
    "refusal threshold, PII masking, an independent review stage, and has no write "
    "access to any order."
)

# ------------------------------------------------------------- Runtime layer
CHARS_PER_TOKEN = 4          # rough heuristic for budgeting
PROMPT_OVERHEAD_TOKENS = 600  # crew prompts + tool descriptions (estimate)
EXPECTED_OUTPUT_TOKENS = 250
CONTEXT_TOKENS = 400          # top-k retrieved context (estimate)


class BudgetExceeded(Exception):
    pass


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def estimate_request_tokens(question: str) -> int:
    return (estimate_tokens(question) + PROMPT_OVERHEAD_TOKENS
            + CONTEXT_TOKENS + EXPECTED_OUTPUT_TOKENS)


def enforce_budget(question: str, cap: int = MAX_REQUEST_TOKENS) -> int:
    est = estimate_request_tokens(question)
    if est > cap:
        raise BudgetExceeded(f"estimated {est} tokens exceeds per-request cap of {cap}")
    return est
