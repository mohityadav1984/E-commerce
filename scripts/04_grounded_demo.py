import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from nykaa_agent.config import active_strategy
from nykaa_agent.queries import CHUNK_EVAL, OUT_OF_SCOPE_DEMO
from nykaa_agent.rag import get_kb

kb = get_kb()
print("active strategy:", active_strategy())
for q, _ in CHUNK_EVAL:
    r = kb.grounded_answer(q, use_cache=False)
    print(f"\nQ: {q}\n   grounded={r['grounded']} top_score={r['top_score']} threshold={r['threshold']} sources={r['sources']}\n   A: {r['answer']}")
r = kb.grounded_answer(OUT_OF_SCOPE_DEMO, use_cache=False)
print(f"\n[OUT-OF-SCOPE] Q: {OUT_OF_SCOPE_DEMO}\n   grounded={r['grounded']} top_score={r['top_score']} threshold={r['threshold']}\n   A: {r['answer']}")
assert not r["grounded"], "out-of-scope query must trigger the fallback"
