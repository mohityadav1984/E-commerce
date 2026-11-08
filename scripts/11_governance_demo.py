import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from nykaa_agent import config  # noqa: F401  (sets CREWAI_DISABLE_TELEMETRY etc. first)

from nykaa_agent.crew import ROLE_COMPOSER, ROLE_RETRIEVAL, build_agents, get_agents
from nykaa_agent.crew_tools import OrderStatusTool
from nykaa_agent.governance import (RISK_CLASSIFICATION, RISK_JUSTIFICATION, BudgetExceeded,
                                    LeastAutonomyViolation, audit_crew_wiring, enforce_budget)
from nykaa_agent.pipeline import get_agent

print("=== Application layer: least autonomy ===")
print("wiring of the real crew:", audit_crew_wiring(get_agents().values()))
for role in (ROLE_RETRIEVAL, ROLE_COMPOSER):
    try:
        build_agents(extra_wiring={role: [OrderStatusTool()]})
        print(f"!! {role} was allowed to hold check_order_status")
    except LeastAutonomyViolation as exc:
        print(f"BLOCKED  -> {exc}")
print("""
Guard: governance.TOOL_ALLOWLIST maps check_order_status to {'Lookup Agent'}. Every
tool handed to an agent passes through crew._wire() -> authorize_tool(), which raises
LeastAutonomyViolation for any other role, and audit_crew_wiring() re-checks the built
crew. The restriction is therefore enforced when the crew is constructed, not by prompt
instructions an agent could ignore.""")

print("\n=== Application layer: risk classification ===")
print("RISK LEVEL:", RISK_CLASSIFICATION, "\n" + RISK_JUSTIFICATION)

print("\n=== Runtime layer: per-request token budget ===")
print("normal request  ->", enforce_budget("What is the return window for beauty products?"), "tokens (OK)")
big = "Tell me about returns. " * 400
try:
    enforce_budget(big)
except BudgetExceeded as exc:
    print("oversized (direct) -> REJECTED:", exc)
try:
    get_agent().ask(big, session_id="budget-demo", endpoint="demo")
except BudgetExceeded as exc:
    print("oversized (full pipeline) -> REJECTED before any crew/LLM call:", exc)
