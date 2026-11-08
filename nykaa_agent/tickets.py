"""Support ticket creation for questions the policy knowledge base cannot answer.

No real ticketing system is connected, so tickets are appended to logs/tickets.jsonl.
Replace `create_ticket` with a call to your helpdesk API when one is available.
Stdlib only."""
from __future__ import annotations

import datetime
import json
import threading
import uuid

from .config import LOG_DIR
from .guardrails import mask_pii

TICKETS_FILE = LOG_DIR / "tickets.jsonl"
ESCALATION_LEVEL = "Level 2 (team lead)"   # per the escalation matrix in kb/12_escalation_matrix.txt
_LOCK = threading.Lock()


def create_ticket(record_id: str, question: str, reason: str) -> dict:
    entry = {
        "ticket_id": "TKT-" + uuid.uuid4().hex[:8].upper(),
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "record_id": record_id,
        "question": mask_pii(question)[0],      # PII is masked before anything is written
        "reason": reason,
        "escalated_to": ESCALATION_LEVEL,
        "status": "open",
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with _LOCK, open(TICKETS_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def policy_gap_text(ticket_id: str) -> str:
    return ("I don't have information on that in our policy knowledge base, so I've created "
            f"ticket {ticket_id} and escalated it to a {ESCALATION_LEVEL} support agent.")
