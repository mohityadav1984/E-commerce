"""End-to-end orchestration: guardrails -> budget -> memory -> crew -> output guard
-> Autogen review -> structured log. One place so HTTP, WebSocket and scripts agree."""
from __future__ import annotations

from . import config  # noqa: F401
import threading
import time
import uuid
from datetime import datetime, timezone

from . import memory
from .crew import needs_policy, run_crew
from .governance import BudgetExceeded, enforce_budget
from .guardrails import check_output_grounded, detect_injection, mask_pii
from .llm import FALLBACK
from .logging_setup import log_request
from .rag import get_kb
from .review import review_answer
from .tickets import create_ticket, policy_gap_text
from .schemas import CrewAnswer
from .tools import order_sentences

INJECTION_REFUSAL = "I can't help with that request. I can only answer Nykaa support questions."


def enforce_groundedness(answer: CrewAnswer, rag: dict | None, order: dict | None):
    """Output-side guardrail: refuse unless the answer rests on retrieved policy
    text that cleared the calibrated threshold, or on a real order record."""
    if order and order.get("found"):
        return answer, []
    if rag is not None:
        ok, why = check_output_grounded(float(rag["top_score"]), float(rag["threshold"]))
        if not ok:
            refused = answer.model_copy(update={"answer": FALLBACK, "grounded": False,
                                                "refused": True, "sources": [], "escalate": False})
            return refused, [f"output_groundedness ({why})"]
    return answer, []


class SupportAgent:
    def __init__(self):
        self.kb = get_kb()
        self._results: dict[str, dict] = {}
        self._lock = threading.Lock()
        self.chat = memory.build_chat_runnable(self._turn)

    # one conversational turn, executed inside RunnableWithMessageHistory
    def _turn(self, inputs: dict, config: dict) -> str:
        sid = config["configurable"]["session_id"]
        question, carried = memory.resolve_with_history(inputs["input"], inputs.get("history", []))
        out = run_crew(question)
        answer, fired = enforce_groundedness(out["answer"], out["rag"], out["order"])
        if not out["schema_ok"]:
            fired.append("schema_validation_failed")

        review, transcript, cache_hit = None, [], bool(out["rag"] and out["rag"].get("cache_hit"))
        if not answer.refused:
            ctx = "\n".join((out["rag"] or {}).get("context", []) +
                            (order_sentences(out["order"]) if out["order"] else []))
            review, transcript = review_answer(answer.answer, ctx)
            if not review.approved:
                fired.append("review_revised_answer")
        final = review.final_answer if review else answer.answer
        # Order record found, but the policy part of the question has no supporting sentence:
        # say so, open a ticket and escalate. Done after review so the reviewer can't strip it.
        ticket_id = None
        if (out["order"] and out["order"].get("found") and out["rag"] is not None
                and not out["rag"].get("grounded") and needs_policy(question)):
            ticket = create_ticket(out["order"]["record_id"], question,
                                   reason="no policy sentence answers the question")
            ticket_id = ticket["ticket_id"]
            final = f"{final} {policy_gap_text(ticket_id)}"
            answer = answer.model_copy(update={"escalate": True})
            fired.append("policy_gap_ticket_created")
        with self._lock:
            self._results[sid] = {"answer": final, "crew": answer, "review": review,
                                  "fired": fired, "tool_calls": out["tool_calls"],
                                  "cache_hit": cache_hit, "resolved": question if carried else None,
                                  "review_transcript": transcript, "rag": out["rag"], "order": out["order"],
                                  "ticket_id": ticket_id}
        return final

    def ask(self, question: str, session_id: str = "default", endpoint: str = "ask") -> dict:
        trace_id, t0 = uuid.uuid4().hex, time.perf_counter()
        masked, pii_kinds = mask_pii(question)
        fired = [f"pii_masked:{k}" for k in pii_kinds]
        entry = {"trace_id": trace_id, "endpoint": endpoint, "session_id": session_id,
                 "question": masked, "guardrails_fired": fired}
        try:
            if detect_injection(masked):
                fired.append("prompt_injection")
                body = {"answer": INJECTION_REFUSAL, "crew": CrewAnswer(answer=INJECTION_REFUSAL, grounded=False, refused=True),
                        "review": None, "fired": [], "tool_calls": {}, "cache_hit": False, "resolved": None}
                est, status = 0, "blocked_injection"
            else:
                est = enforce_budget(masked)                       # raises BudgetExceeded
                self.chat.invoke({"input": masked}, config={"configurable": {"session_id": session_id}})
                with self._lock:
                    body = self._results.pop(session_id)
                fired.extend(body["fired"])
                status = "refused" if body["crew"].refused else "ok"
        except BudgetExceeded as exc:
            entry.update(status="rejected_budget", detail=str(exc),
                         duration_ms=round((time.perf_counter() - t0) * 1000, 2))
            log_request(entry)
            raise
        except Exception as exc:                                   # log, then re-raise
            entry.update(status="error", detail=f"{type(exc).__name__}: {exc}",
                         duration_ms=round((time.perf_counter() - t0) * 1000, 2))
            log_request(entry)
            raise

        crew: CrewAnswer = body["crew"]
        ms = round((time.perf_counter() - t0) * 1000, 2)
        entry.update(status=status, duration_ms=ms, cache_hit=body["cache_hit"],
                     tool_calls=body["tool_calls"], estimated_tokens=est,
                     review_approved=body["review"].approved if body["review"] else None)
        log_request(entry)
        return {
            "trace_id": trace_id, "session_id": session_id, "answer": body["answer"],
            "sources": crew.sources, "grounded": crew.grounded, "refused": crew.refused,
            "escalate": crew.escalate, "order": crew.order.model_dump() if crew.order else None,
            "review": body["review"].model_dump() if body["review"] else None,
            "guardrails_fired": fired, "cache_hit": body["cache_hit"], "ticket_id": body.get("ticket_id"),
            "tool_calls": body["tool_calls"], "resolved_question": body["resolved"],
            "estimated_tokens": est, "duration_ms": ms,
            "_internal": {"context": "\n".join((body.get("rag") or {}).get("context", [])),
                          "review_transcript": body.get("review_transcript", [])},
        }


_AGENT: SupportAgent | None = None
_INIT = threading.Lock()


def get_agent() -> SupportAgent:
    global _AGENT
    with _INIT:
        if _AGENT is None:
            _AGENT = SupportAgent()
        return _AGENT
