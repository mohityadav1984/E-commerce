"""check_order_status + designed escalation score (stdlib only)."""
from __future__ import annotations

import math
import re

from dataset import ORDERS

_BY_ID = {o["record_id"]: o for o in ORDERS}
ORDER_ID_RE = re.compile(r"\bNYK-\d{4}\b", re.I)

# ---- escalation score ------------------------------------------------------
# escalation_score = W_DELAY * delayed_shipment + W_RECENCY * (days_since_created / 30)
#   delayed_shipment  in {0,1}; days/30 in [0,1]  =>  score in [0,1]
# W_DELAY (0.6) > W_RECENCY (0.4): a delay alone is a stronger complaint signal
# than age alone, and a non-delayed order can never reach the escalation
# threshold (max 0.4) no matter how old it is.
W_DELAY, W_RECENCY, MAX_DAYS = 0.6, 0.4, 30


def _percentile(sorted_vals: list[int], q: float) -> float:
    idx = (len(sorted_vals) - 1) * q
    lo, hi = math.floor(idx), math.ceil(idx)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


P80_DAYS = _percentile(sorted(o["days_since_created"] for o in ORDERS), 0.80)
# Threshold = score of a delayed order exactly as old as the 80th-percentile
# order in OUR generated data: only delayed orders in the oldest ~20% escalate.
ESCALATION_THRESHOLD = round(W_DELAY + W_RECENCY * P80_DAYS / MAX_DAYS, 2)


def escalation_score(delayed: bool, days: int) -> float:
    return round(W_DELAY * float(delayed) + W_RECENCY * min(days, MAX_DAYS) / MAX_DAYS, 3)


def check_order_status(record_id: str) -> dict:
    """Look up an order. Returns status, order_value_inr and escalation_score."""
    rec = _BY_ID.get(record_id.strip().upper())
    if rec is None:
        return {"found": False, "record_id": record_id, "error": "order not found"}
    score = escalation_score(rec["delayed_shipment"], rec["days_since_created"])
    return {
        "found": True,
        "record_id": rec["record_id"],
        "status": rec["status"],
        "order_value_inr": rec["order_value_inr"],
        "escalation_score": score,
        "recommend_escalation": score >= ESCALATION_THRESHOLD,
        "escalation_threshold": ESCALATION_THRESHOLD,
    }


def extract_order_id(text: str) -> str | None:
    m = ORDER_ID_RE.search(text or "")
    return m.group(0).upper() if m else None


def order_sentences(order: dict) -> list[str]:
    """Plain-English sentences about an order. Used by BOTH the composer and the
    review stage's context, so order facts are always 'supported'."""
    if not order or not order.get("found"):
        rid = (order or {}).get("record_id", "that ID")
        return [f"I could not find an order with ID {rid}."]
    out = [f"Order {order['record_id']} is currently {order['status']} "
           f"with an order value of INR {order['order_value_inr']}."]
    if order.get("recommend_escalation"):
        out.append("This order looks delayed and is being flagged for escalation "
                   "to a Level 2 team lead.")
    return out
