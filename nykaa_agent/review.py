"""Autogen review stage (Task 14): 2-agent RoundRobinGroupChat, max_turns=2.

  Policy_Compliance_Reviewer -> flags draft sentences not supported by the context
  Final_Editor               -> returns a structured ReviewVerdict (approve / revise)

Both agents run on a deterministic keyless ChatCompletionClient (MOCK_LLM)."""
from __future__ import annotations

from . import config  # noqa: F401
import asyncio
import json
import re

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.models import ChatCompletionClient, CreateResult, RequestUsage

from .chunking import split_sentences
from .guardrails import support_ratio
from .llm import FALLBACK
from .schemas import ReviewVerdict

SUPPORT_MIN = 0.6     # a sentence needs >=60% of its content words in the context

_DRAFT = re.compile(r"DRAFT:\n(.*?)\nEND_DRAFT", re.S)
_CONTEXT = re.compile(r"CONTEXT:\n(.*?)\nEND_CONTEXT", re.S)
_FINDINGS = re.compile(r"unsupported_sentences=(\[.*?\])\s*(?:\n|$)", re.S)


def _flatten(messages) -> str:
    out = []
    for m in messages:
        c = getattr(m, "content", "")
        out.append(c if isinstance(c, str) else str(c))
    return "\n".join(out)


def reviewer_reply(text: str) -> str:
    draft, ctx = _DRAFT.search(text), _CONTEXT.search(text)
    if not draft or not ctx:
        return 'REVIEW: could not parse input. unsupported_sentences=[]'
    bad = [s for s in split_sentences(draft.group(1)) if support_ratio(s, ctx.group(1)) < SUPPORT_MIN]
    verdict = "All sentences are supported by the context." if not bad else \
        f"{len(bad)} sentence(s) are NOT supported by the retrieved context."
    return f"REVIEW: {verdict}\nunsupported_sentences={json.dumps(bad)}\n"


def editor_reply(text: str) -> str:
    draft = _DRAFT.search(text)
    findings = _FINDINGS.findall(text)
    original = draft.group(1).strip() if draft else ""
    bad = json.loads(findings[-1]) if findings else []
    if not bad:
        v = {"approved": True, "final_answer": original,
             "reason": "Every sentence is supported by the retrieved context; draft approved unchanged."}
    else:
        kept = [s for s in split_sentences(original) if s not in bad]
        v = {"approved": False, "final_answer": " ".join(kept) if kept else FALLBACK,
             "reason": f"Removed {len(bad)} ungrounded sentence(s): {json.dumps(bad)}"}
    return json.dumps(v)


class MockReviewClient(ChatCompletionClient):
    """Deterministic keyless model client for the Autogen agents."""

    def __init__(self, role: str):
        self.role = role
        self._total = RequestUsage(prompt_tokens=0, completion_tokens=0)

    @property
    def model_info(self):
        return {"vision": False, "function_calling": False, "json_output": True,
                "family": "unknown", "structured_output": True}

    @property
    def capabilities(self):
        return self.model_info

    async def create(self, messages, *, tools=[], tool_choice="auto", json_output=None,
                     extra_create_args={}, cancellation_token=None) -> CreateResult:
        text = _flatten(messages)
        content = reviewer_reply(text) if self.role == "reviewer" else editor_reply(text)
        usage = RequestUsage(prompt_tokens=len(text) // 4, completion_tokens=len(content) // 4)
        self._total = RequestUsage(prompt_tokens=self._total.prompt_tokens + usage.prompt_tokens,
                                   completion_tokens=self._total.completion_tokens + usage.completion_tokens)
        return CreateResult(finish_reason="stop", content=content, usage=usage, cached=False)

    async def create_stream(self, messages, *, tools=[], tool_choice="auto", json_output=None,
                            extra_create_args={}, cancellation_token=None):
        yield await self.create(messages, tools=tools, json_output=json_output)

    async def close(self) -> None:
        return None

    def actual_usage(self) -> RequestUsage:
        return self._total

    def total_usage(self) -> RequestUsage:
        return self._total

    def count_tokens(self, messages, *, tools=[]) -> int:
        return len(_flatten(messages)) // 4

    def remaining_tokens(self, messages, *, tools=[]) -> int:
        return 8192 - self.count_tokens(messages)


def build_review_team() -> RoundRobinGroupChat:
    reviewer = AssistantAgent(
        "Policy_Compliance_Reviewer", model_client=MockReviewClient("reviewer"),
        system_message="Check each sentence of the DRAFT against the CONTEXT and list unsupported ones.")
    editor = AssistantAgent(
        "Final_Editor", model_client=MockReviewClient("editor"),
        output_content_type=ReviewVerdict,
        system_message="Approve the draft unchanged or revise it, returning a ReviewVerdict.")
    return RoundRobinGroupChat(
        [reviewer, editor],
        max_turns=2,    # real parameter name (not max_iterations): each agent speaks once
        custom_message_types=[StructuredMessage[ReviewVerdict]],  # else ValueError at runtime
    )


async def review_answer_async(draft: str, context: str) -> tuple[ReviewVerdict, list[str]]:
    task = f"Review the DRAFT customer-support answer against the retrieved CONTEXT.\nDRAFT:\n{draft}\nEND_DRAFT\nCONTEXT:\n{context}\nEND_CONTEXT"
    result = await build_review_team().run(task=task)
    transcript = [f"{m.source}: {m.content if isinstance(m.content, str) else m.content.model_dump_json()}"
                  for m in result.messages]
    for m in reversed(result.messages):
        if isinstance(m, StructuredMessage) and isinstance(m.content, ReviewVerdict):
            return m.content, transcript
    raise RuntimeError("Final_Editor returned no structured verdict")


def review_answer(draft: str, context: str) -> tuple[ReviewVerdict, list[str]]:
    """Sync wrapper (call from a non-async thread)."""
    return asyncio.run(review_answer_async(draft, context))
