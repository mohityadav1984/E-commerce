"""Run all 15 test queries through the full pipeline and score them (Task 13)."""
from __future__ import annotations

from .config import REQUEST_LOG
from .judge import build_judge_prompt, mock_judge
from .pipeline import get_agent
from .queries import TEST15
from .tools import order_sentences, check_order_status, extract_order_id

PROPS = ("accuracy", "grounding", "completeness", "safety")


def run_eval() -> dict:
    agent = get_agent()
    rows = []
    for qid, question, topic, expected, kind in TEST15:
        res = agent.ask(question, session_id=f"eval-{qid}", endpoint="eval")
        ctx = res["_internal"]["context"]
        oid = extract_order_id(question)
        if oid:
            ctx += "\n" + "\n".join(order_sentences(check_order_status(oid)))
        _ = build_judge_prompt(question, ctx, res["answer"], expected)   # what a real judge would see
        scores = mock_judge(question=question, answer=res["answer"], context=ctx, expected=expected,
                            kind=kind, refused=res["refused"], raw_question=question)
        rows.append({"id": qid, "topic": topic, "question": question, "answer": res["answer"],
                     "refused": res["refused"], "fired": res["guardrails_fired"], **scores})
    avg = {p: round(sum(r[p] for r in rows) / len(rows), 2) for p in PROPS}
    return {"rows": rows, "averages": avg}
