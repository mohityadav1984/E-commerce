import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from nykaa_agent import config  # noqa: F401  (sets CREWAI_DISABLE_TELEMETRY etc. first)

import json
from nykaa_agent.config import TRANSCRIPT_DIR
from nykaa_agent.eval_suite import PROPS, run_eval

res = run_eval()
print(f"{'#':>2} {'topic':<22} {'acc':>4} {'grnd':>5} {'comp':>5} {'safe':>5}  refused")
for r in res["rows"]:
    print(f"{r['id']:>2} {r['topic']:<22} {r['accuracy']:>4} {r['grounding']:>5} {r['completeness']:>5} {r['safety']:>5}  {r['refused']}")
print("\nAVERAGES over 15 queries:", res["averages"])
print("\nanswers:")
for r in res["rows"]:
    print(f"  [{r['id']}] {r['question']}\n       -> {r['answer']}")
(TRANSCRIPT_DIR / "eval_results.json").write_text(json.dumps(res, indent=2))
