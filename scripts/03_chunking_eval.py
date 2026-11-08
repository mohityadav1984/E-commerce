import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
from nykaa_agent.config import CHUNKING_RESULT_FILE, COLLECTIONS, TOP_K
from nykaa_agent.queries import CHUNK_EVAL
from nykaa_agent.rag import get_kb

kb = get_kb()
summary = {}
for strategy in COLLECTIONS:
    print(f"\n=== {strategy} collection (top-{TOP_K} chunks -> dedup to parent docs) ===")
    ps, rs = [], []
    for q, gold in CHUNK_EVAL:
        hits = kb.retrieve(q, strategy, TOP_K)
        docs = list(dict.fromkeys(h["doc_id"] for h in hits))
        tp = len(set(docs) & gold)
        p, r = tp / len(docs), tp / len(gold)
        ps.append(p); rs.append(r)
        print(f"Q: {q}")
        print(f"   retrieved docs={docs}  gold={sorted(gold)}  TP={tp}")
        print(f"   precision={tp}/{len(docs)}={p:.3f}  recall={tp}/{len(gold)}={r:.3f}")
    mp, mr = sum(ps) / len(ps), sum(rs) / len(rs)
    f1 = 2 * mp * mr / (mp + mr) if mp + mr else 0.0
    print(f"MEAN precision={sum(ps):.3f}/{len(ps)}={mp:.3f}  recall={sum(rs):.3f}/{len(rs)}={mr:.3f}  F1={f1:.3f}")
    summary[strategy] = {"precision": round(mp, 4), "recall": round(mr, 4), "f1": round(f1, 4)}

order = sorted(summary, key=lambda s: (summary[s]["f1"], summary[s]["precision"], s == "sentence"), reverse=True)
best, other = order[0], order[1]
summary["recommended"] = best
CHUNKING_RESULT_FILE.write_text(json.dumps(summary, indent=2))
b, o = summary[best], summary[other]
print(f"\nRECOMMENDATION (auto-drafted from my numbers; rewrite in your own words in README):")
print(f"I would deploy the {best} collection: mean precision {b['precision']} / recall {b['recall']} "
      f"(F1 {b['f1']}) versus {o['precision']} / {o['recall']} (F1 {o['f1']}) for {other} chunks.")
