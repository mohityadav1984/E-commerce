import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from nykaa_agent import config  # noqa: F401  (sets CREWAI_DISABLE_TELEMETRY etc. first)

from nykaa_agent.crew import get_agents, run_crew
from nykaa_agent.crew_tools import TOOL_CALLS
from nykaa_agent.tools import ESCALATION_THRESHOLD, P80_DAYS

agents = get_agents()
print("agents:", [a.role for a in agents.values()])
print("tools per agent:", {a.role: [t.name for t in a.tools] for a in agents.values()})
print(f"escalation threshold={ESCALATION_THRESHOLD} (p80 of days_since_created = {P80_DAYS:.1f})")
samples = [
    ("POLICY ONLY  -> expect rag_lookup", "How many days do I have to return a face serum?"),
    ("ORDER ONLY   -> expect check_order_status", "What is the status of order NYK-0054?"),
    ("BOTH         -> expect both tools", "Order NYK-0003 is placed; can I cancel it for free?"),
]
for label, q in samples:
    out = run_crew(q)
    print(f"\n--- {label}\nQ: {q}")
    print("tool calls this run:", out["tool_calls"])
    print("schema valid (CrewAnswer):", out["schema_ok"])
    print("draft answer:", out["answer"].model_dump_json(indent=2))
print("\ncumulative TOOL_CALLS:", dict(TOOL_CALLS))
