"""Seeded, deterministic order dataset generator (Part 1, Task 1).

Design choices (reproduce exactly with `python dataset.py`):
  SEED = 7, N_ORDERS = 60
  Category weights  : Apparel .25, Electronics .20, Home .15, Footwear .20, Beauty .20
  Status weights    : Placed .15, Shipped .20, Delivered .45, Returned .12, Refunded .08
  Amount range      : INR 199 - 14,999, log-uniform, rounded to a "x9" price point.
      Reasoning: Nykaa sells low-ticket beauty items (~INR 200-1,500) up to
      electronics/appliances (~INR 15k), so log-uniform keeps most orders cheap
      while still producing a tail of expensive ones.
  days_since_created: uniform integer 0-30
  delayed_shipment  : Bernoulli(0.20) for orders that were ever shipped
                      (Shipped/Delivered/Returned/Refunded); a Placed order has
                      not shipped yet, so it cannot be a delayed *shipment*.
Nothing is hand-edited after generation; if the delay rate leaves 10-30%,
change SEED and regenerate.
"""
from __future__ import annotations

import math
import random
from collections import Counter

SEED = 7
N_ORDERS = 60
CATEGORIES = ["Apparel", "Electronics", "Home", "Footwear", "Beauty"]
CATEGORY_WEIGHTS = [0.25, 0.20, 0.15, 0.20, 0.20]
STATUSES = ["Placed", "Shipped", "Delivered", "Returned", "Refunded"]
STATUS_WEIGHTS = [0.15, 0.20, 0.45, 0.12, 0.08]
AMOUNT_MIN, AMOUNT_MAX = 199, 14_999
DELAY_PROB = 0.20


def _amount(rng: random.Random) -> int:
    raw = math.exp(rng.uniform(math.log(AMOUNT_MIN), math.log(AMOUNT_MAX)))
    return min(AMOUNT_MAX, int(raw // 10) * 10 + 9)  # price point ending in 9


def generate_orders(seed: int = SEED, n: int = N_ORDERS) -> list[dict]:
    rng = random.Random(seed)
    orders = []
    for i in range(1, n + 1):
        status = rng.choices(STATUSES, STATUS_WEIGHTS)[0]
        shipped = status != "Placed"
        orders.append(
            {
                "record_id": f"NYK-{i:04d}",
                "category": rng.choices(CATEGORIES, CATEGORY_WEIGHTS)[0],
                "status": status,
                "order_value_inr": _amount(rng),
                "days_since_created": rng.randint(0, 30),
                "delayed_shipment": bool(shipped and rng.random() < DELAY_PROB),
            }
        )
    return orders


def validate(orders: list[dict]) -> dict:
    cat = Counter(o["category"] for o in orders)
    sta = Counter(o["status"] for o in orders)
    delayed_pct = 100 * sum(o["delayed_shipment"] for o in orders) / len(orders)
    problems = []
    if len(orders) < 40:
        problems.append("fewer than 40 orders")
    for c in CATEGORIES:
        if cat[c] < 3:
            problems.append(f"category {c} has {cat[c]} (<3)")
    for s in STATUSES:
        if sta[s] < 1:
            problems.append(f"status {s} missing")
    if not 10 <= delayed_pct <= 30:
        problems.append(f"delayed_shipment {delayed_pct:.1f}% outside 10-30%")
    for o in orders:
        if not 0 <= o["days_since_created"] <= 30:
            problems.append(f"{o['record_id']} days out of range")
    return {"by_category": dict(cat), "by_status": dict(sta),
            "delayed_pct": round(delayed_pct, 1), "problems": problems}


ORDERS: list[dict] = generate_orders()

if __name__ == "__main__":
    report = validate(ORDERS)
    print(f"seed={SEED}  orders={len(ORDERS)}")
    print("count per category:", report["by_category"])
    print("count per status  :", report["by_status"])
    print(f"delayed_shipment  : {report['delayed_pct']}%")
    print("problems:", report["problems"] or "none")
    raise SystemExit(1 if report["problems"] else 0)
