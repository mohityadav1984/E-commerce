"""Deterministic, keyless MockLLM for CrewAI (subclass of crewai.llms.base_llm.BaseLLM).

Behaviour (CrewAI's text ReAct protocol, since function calling is declared unsupported):
  * agent WITH a tool  : 1st call -> emit `Action:` / `Action Input:`;
                         2nd call -> emit `Final Answer:` = the tool's observation.
  * agent WITHOUT tools: the Response Composer; emits a JSON CrewAnswer.

The two silent pitfalls called out in the brief are handled explicitly:
  (1) The observation is read ONLY from non-system messages and ONLY from text that
      follows an `Action Input:` that WE generated - never from CrewAI's system-prompt
      template (which literally contains "Observation: the result of the action").
  (2) The tool is dispatched from its declared ARGUMENT SCHEMA (record_id vs query),
      never by testing substrings of the tool's name.
"""
from __future__ import annotations

from . import config  # noqa: F401
import json
import re
from typing import Any

from crewai.llms.base_llm import BaseLLM

from . import markers
from .llm import FALLBACK
from .tools import extract_order_id, order_sentences

_TOOL_BLOCK = re.compile(r"Tool Name:\s*(?P<name>\S+)\s*\n\s*Tool Arguments:\s*(?P<args>\{.*?\})\s*\n", re.S)
_ARG_KEYS = re.compile(r"""['"](\w+)['"]\s*:\s*\{""")
_QUESTION = re.compile(r"QUESTION:\s*(.+)")


def _normalize(messages: Any) -> list[dict]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    out = []
    for m in messages:
        role = m.get("role", "") if isinstance(m, dict) else getattr(m, "role", "")
        content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        out.append({"role": role, "content": content or ""})
    return out


def parse_tools(system_text: str) -> list[dict]:
    tools = []
    for m in _TOOL_BLOCK.finditer(system_text):
        tools.append({"name": m.group("name"), "arg_keys": _ARG_KEYS.findall(m.group("args"))})
    return tools


def build_tool_args(tool: dict, question: str) -> dict:
    """Dispatch off the tool's declared argument schema, NOT its name."""
    keys = tool["arg_keys"]
    if "record_id" in keys:
        return {"record_id": extract_order_id(question) or ""}
    if "query" in keys:
        return {"query": question}
    return {keys[0]: question} if keys else {}


def latest_observation(convo: list[dict]) -> str | None:
    """Tool result produced AFTER an Action Input that the model itself emitted."""
    acted, obs = False, None
    for m in convo:
        c = m["content"]
        if m["role"] == "assistant" and "Action Input:" in c:
            acted = True
            tail = c[c.rfind("Action Input:"):]
            if "Observation:" in tail:
                obs = tail.split("Observation:", 1)[1].strip()
        elif acted and m["role"] != "assistant" and c.lstrip().startswith("Observation:"):
            obs = c.lstrip()[len("Observation:"):].strip()
    return obs


class MockLLM(BaseLLM):
    """Zero-key, zero-network deterministic LLM."""

    def __init__(self, label: str = "mock"):
        super().__init__(model=f"mock-llm-{label}", temperature=0.0)
        self.calls = 0

    # -- BaseLLM interface -------------------------------------------------
    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, **kwargs) -> str:
        self.calls += 1
        msgs = _normalize(messages)
        system = "\n".join(m["content"] for m in msgs if m["role"] == "system")
        convo = [m for m in msgs if m["role"] != "system"]
        convo_text = "\n".join(m["content"] for m in convo)
        q = _QUESTION.search(convo_text)
        question = q.group(1).strip() if q else convo_text[-300:]

        declared = parse_tools(system)
        if declared:                                    # tool-using agent
            obs = latest_observation(convo)
            if obs is None:
                tool = declared[0]
                args = build_tool_args(tool, question)
                return (f"Thought: I should use the {tool['name']} tool.\n"
                        f"Action: {tool['name']}\n"
                        f"Action Input: {json.dumps(args)}")
            return f"Thought: I now know the final answer\nFinal Answer: {obs}"
        return f"Thought: I now know the final answer\nFinal Answer: {self._compose(convo_text, question)}"

    def supports_function_calling(self) -> bool:
        return False            # forces CrewAI's plain-text ReAct loop

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 8192

    # -- Response Composer -------------------------------------------------
    @staticmethod
    def _compose(convo_text: str, question: str) -> str:
        rag = markers.decode(convo_text, markers.RAG)
        order = markers.decode(convo_text, markers.ORDER)
        parts, sources, grounded, escalate, conf = [], [], False, False, 0.0
        if rag:
            grounded = bool(rag["grounded"])
            sources = rag.get("sources", [])
            conf = float(rag.get("top_score", 0.0))
            if grounded or order is None:          # never echo a fallback next to an order record
                parts.append(rag["answer"])
        order_model = None
        if order is not None:
            parts.extend(order_sentences(order))
            if order.get("found"):
                grounded = True
                escalate = bool(order.get("recommend_escalation"))
                conf = max(conf, 0.99)
                order_model = {k: order.get(k) for k in
                               ("record_id", "status", "order_value_inr",
                                "escalation_score", "recommend_escalation")}
                order_model["found"] = True
            else:
                order_model = {"record_id": order.get("record_id", ""), "found": False}
        answer = " ".join(parts) if parts else FALLBACK
        draft = {"answer": answer, "sources": sources, "order": order_model,
                 "grounded": grounded, "refused": not grounded, "escalate": escalate,
                 "confidence": round(max(0.0, min(conf, 1.0)), 3)}
        return json.dumps(draft, separators=(",", ":"))
