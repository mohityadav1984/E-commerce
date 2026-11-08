import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
from nykaa_agent.config import CALIBRATION_FILE, COLLECTIONS
from nykaa_agent.queries import CALIB_IN, CALIB_OUT
from nykaa_agent.rag import get_kb

kb = get_kb()
print("indexed chunks:", {s: c.count() for s, c in kb.collections.items()})
out = {}
for strategy in COLLECTIONS:
    ins = [(q, kb.retrieve(q, strategy, 1)[0]["score"]) for q in CALIB_IN]
    outs = [(q, kb.retrieve(q, strategy, 1)[0]["score"]) for q in CALIB_OUT]
    print(f"\n=== {strategy} collection: top-1 cosine similarity ===")
    for q, s in ins:
        print(f"  IN   {s:.4f}  {q}")
    for q, s in outs:
        print(f"  OUT  {s:.4f}  {q}")
    min_in, max_out = min(s for _, s in ins), max(s for _, s in outs)
    if min_in <= max_out:
        raise SystemExit(f"[{strategy}] clusters overlap (min_in={min_in}, max_out={max_out}); "
                         "inspect the scores above and add/replace calibration queries.")
    thr = round((min_in + max_out) / 2, 4)
    print(f"  lowest in-scope={min_in:.4f}  highest out-of-scope={max_out:.4f}  -> threshold={thr}")
    out[strategy] = {"in": [s for _, s in ins], "out": [s for _, s in outs],
                     "min_in": min_in, "max_out": max_out, "threshold": thr}
CALIBRATION_FILE.write_text(json.dumps(out, indent=2))
print("\nwrote", CALIBRATION_FILE)
