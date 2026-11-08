import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from nykaa_agent import config  # noqa: F401  (sets CREWAI_DISABLE_TELEMETRY etc. first)

from nykaa_agent.memory import get_session_history, reset_session
from nykaa_agent.pipeline import get_agent

agent = get_agent()
reset_session("demo-A"); reset_session("fresh-B")
print("=== TRANSCRIPT 1: multi-turn, state CARRIED across turns (session demo-A) ===")
for q in ["What is the status of order NYK-0054?",
          "Is it eligible for a return?",
          "And what is the refund timeline for COD?"]:
    r = agent.ask(q, session_id="demo-A", endpoint="demo")
    print(f"\nUSER: {q}\n  resolved_question: {r['resolved_question']}\n  order: {r['order']}\n  AGENT: {r['answer']}")
print("\nstored messages in demo-A:", len(get_session_history("demo-A").messages))

print("\n=== TRANSCRIPT 2: FRESH conversation, same follow-up, state ABSENT (session fresh-B) ===")
r = agent.ask("Is it eligible for a return?", session_id="fresh-B", endpoint="demo")
print(f"USER: Is it eligible for a return?\n  resolved_question: {r['resolved_question']} (None = nothing carried over)\n  order: {r['order']}\n  AGENT: {r['answer']}")
print("stored messages in fresh-B:", len(get_session_history("fresh-B").messages))
