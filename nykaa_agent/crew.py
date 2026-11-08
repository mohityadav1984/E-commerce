"""CrewAI crew: Retrieval Agent, Lookup Agent, Response Composer (Tasks 7, 9, 15)."""
from __future__ import annotations

from . import config  # noqa: F401  (sets CREWAI_DISABLE_TELEMETRY before crewai loads)
import json
import re

from crewai import Agent, Crew, Process, Task
from pydantic import ValidationError

from . import markers
from .crew_tools import TOOL_CALLS, OrderStatusTool, RagLookupTool
from .governance import audit_crew_wiring, authorize_tool
from .guardrails import content_words
from .llm import FALLBACK
from .mock_llm import MockLLM
from .schemas import CrewAnswer
from .tools import ORDER_ID_RE, extract_order_id

ROLE_RETRIEVAL, ROLE_LOOKUP, ROLE_COMPOSER = "Retrieval Agent", "Lookup Agent", "Response Composer"
_ORDER_WORDS = set("where status track tracking check order orders my the is of what whats "
                   "please show me about".split())


def _wire(role: str, tools: list) -> list:
    """Least autonomy: refuse to hand a tool to a role that is not allow-listed."""
    for t in tools:
        authorize_tool(role, t.name)
    return tools


def build_agents(extra_wiring: dict[str, list] | None = None) -> dict[str, Agent]:
    """`extra_wiring` exists only so the governance demo can TRY to over-wire a tool."""
    extra = extra_wiring or {}
    retrieval = Agent(
        role=ROLE_RETRIEVAL, goal="Find grounded answers to policy questions in the Nykaa knowledge base.",
        backstory="You only answer from retrieved policy text and never guess.",
        tools=_wire(ROLE_RETRIEVAL, [RagLookupTool()] + extra.get(ROLE_RETRIEVAL, [])),
        llm=MockLLM("retrieval"), allow_delegation=False, verbose=False)
    lookup = Agent(
        role=ROLE_LOOKUP, goal="Look up the status of a specific order.",
        backstory="You are the only agent allowed to read order records.",
        tools=_wire(ROLE_LOOKUP, [OrderStatusTool()] + extra.get(ROLE_LOOKUP, [])),
        llm=MockLLM("lookup"), allow_delegation=False, verbose=False)
    composer = Agent(
        role=ROLE_COMPOSER, goal="Combine retrieval and lookup outputs into one final draft answer.",
        backstory="You write short, accurate customer-support answers as structured JSON.",
        tools=_wire(ROLE_COMPOSER, extra.get(ROLE_COMPOSER, [])),
        llm=MockLLM("composer"), allow_delegation=False, verbose=False)
    agents = {ROLE_RETRIEVAL: retrieval, ROLE_LOOKUP: lookup, ROLE_COMPOSER: composer}
    audit_crew_wiring(agents.values())
    return agents


_AGENTS: dict[str, Agent] | None = None


def get_agents() -> dict[str, Agent]:
    global _AGENTS
    if _AGENTS is None:
        _AGENTS = build_agents()
    return _AGENTS


def needs_policy(question: str) -> bool:
    """Skip the Retrieval Agent for pure order-status questions ('status of NYK-0007')."""
    if not extract_order_id(question):
        return True
    rest = ORDER_ID_RE.sub(" ", question)
    leftover = content_words(rest) - _ORDER_WORDS
    return len(leftover) >= 2


def parse_crew_answer(raw: str) -> CrewAnswer:
    """Validate the Composer's output against the declared Pydantic schema."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in crew output")
    return CrewAnswer.model_validate_json(raw[start:end + 1])


def run_crew(question: str) -> dict:
    """Run the crew via kickoff(). Returns the validated answer plus tool evidence."""
    agents = get_agents()
    before = dict(TOOL_CALLS)
    tasks, composer_ctx = [], []
    if needs_policy(question):
        t = Task(description=f"Answer the customer's policy question with the rag_lookup tool.\nQUESTION: {question}",
                 expected_output="The tool's grounded result.", agent=agents[ROLE_RETRIEVAL])
        tasks.append(t); composer_ctx.append(t)
    if extract_order_id(question):
        t = Task(description=f"Look up the order mentioned with the check_order_status tool.\nQUESTION: {question}",
                 expected_output="The tool's order record.", agent=agents[ROLE_LOOKUP])
        tasks.append(t); composer_ctx.append(t)
    compose = Task(description=f"Combine the context into one final draft answer as a JSON object.\nQUESTION: {question}",
                   expected_output="A JSON object matching the CrewAnswer schema.",
                   agent=agents[ROLE_COMPOSER], context=composer_ctx, output_pydantic=CrewAnswer)
    tasks.append(compose)

    crew = Crew(agents=list(agents.values()), tasks=tasks, process=Process.sequential, verbose=False)
    result = crew.kickoff()

    outputs = [getattr(o, "raw", str(o)) for o in result.tasks_output]
    blob = "\n".join(outputs)
    rag, order = markers.decode(blob, markers.RAG), markers.decode(blob, markers.ORDER)
    try:
        answer = parse_crew_answer(result.raw)
        schema_ok = True
    except (ValueError, ValidationError):
        answer = CrewAnswer(answer=FALLBACK, grounded=False, refused=True)
        schema_ok = False
    delta = {k: TOOL_CALLS[k] - before.get(k, 0) for k in TOOL_CALLS if TOOL_CALLS[k] - before.get(k, 0)}
    return {"answer": answer, "rag": rag, "order": order, "tool_calls": delta, "schema_ok": schema_ok}
