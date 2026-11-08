import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from nykaa_agent import config  # noqa: F401  (sets CREWAI_DISABLE_TELEMETRY etc. first)

from nykaa_agent.rag import get_kb
from nykaa_agent.review import review_answer

kb = get_kb()
q = "How many days do I have to return a face serum?"
res = kb.grounded_answer(q, use_cache=False)
ctx = "\n".join(c["text"] for c in res["chunks"])
good = res["answer"]
bad = good + " Nykaa also gives a free gold coin with every return."    # deliberately ungrounded
for label, draft in [("CASE 1: faithful draft -> expect APPROVED unchanged", good),
                     ("CASE 2: draft with injected ungrounded claim -> expect REVISED", bad)]:
    verdict, transcript = review_answer(draft, ctx)
    print(f"\n=== {label}\nDRAFT : {draft}")
    for line in transcript[1:]:
        print("  ", line)
    print("VERDICT:", verdict.model_dump())
    print("  unchanged?", verdict.final_answer == draft)
