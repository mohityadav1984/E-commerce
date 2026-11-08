import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from nykaa_agent import config  # noqa: F401  (sets CREWAI_DISABLE_TELEMETRY etc. first)

import time
from nykaa_agent.crew_tools import TOOL_CALLS
from nykaa_agent.pipeline import get_agent
from nykaa_agent.rag import get_kb

kb = get_kb(); kb.cache.clear()
q = "How many days do I have to return a face serum?"
print("=== grounded-generation cache (direct) ===")
for label, query in [("1st call (miss)", q), ("2nd call, identical (hit)", q),
                     ("3rd call, different case/spacing/punctuation (hit)", "  HOW many days do I have to RETURN a face serum ??  ")]:
    before = kb.counters.snapshot(); t0 = time.perf_counter()
    r = kb.grounded_answer(query)
    ms = (time.perf_counter() - t0) * 1000
    after = kb.counters.snapshot()
    print(f"{label:<55} cache_hit={r['cache_hit']!s:<5} {ms:8.2f} ms  retrieval {before['retrieval_calls']}->{after['retrieval_calls']}  generation {before['generation_calls']}->{after['generation_calls']}")
print("cache hits/misses:", kb.cache.hits, "/", kb.cache.misses)

print("\n=== through the full crew ===")
kb.cache.clear(); agent = get_agent()
for i in (1, 2):
    g0, r0 = kb.counters.generation_calls, kb.counters.retrieval_calls
    res = agent.ask(q, session_id=f"cache-{i}", endpoint="demo")
    print(f"request {i}: cache_hit={res['cache_hit']} duration={res['duration_ms']} ms  "
          f"retrieval +{kb.counters.retrieval_calls - r0}, generation +{kb.counters.generation_calls - g0}")
