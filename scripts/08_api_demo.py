import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from nykaa_agent import config  # noqa: F401  (sets CREWAI_DISABLE_TELEMETRY etc. first)

import json
from fastapi.testclient import TestClient
from nykaa_agent.api import app
from nykaa_agent.config import REQUEST_LOG
from nykaa_agent.pipeline import get_agent

n_before = len(REQUEST_LOG.read_text().splitlines()) if REQUEST_LOG.exists() else 0
with TestClient(app) as client:
    print("GET /health ->", client.get("/health").json())
    r = client.post("/ask", json={"question": "How long does delivery take to a remote pin code?", "session_id": "api-1"})
    print("POST /ask ->", r.status_code, json.dumps({k: r.json()[k] for k in ("answer", "sources", "grounded", "trace_id")}, indent=2))

    r = client.post("/add-document", json={"doc_id": "13_gift_wrapping", "title": "Gift Wrapping Policy",
        "text": "Gift Wrapping Policy. Nykaa offers gift wrapping on all Beauty orders for INR 49 per item. A personalised message card can be added for free. Gift-wrapped orders cannot be partially returned."})
    print("POST /add-document ->", r.status_code, r.json())
    r = client.post("/ask", json={"question": "How much does gift wrapping cost?", "session_id": "api-1"})
    print("POST /ask (new doc) ->", r.json()["answer"], "| sources:", r.json()["sources"])

    r = client.post("/ask", json={"question": "x" * 12000})
    print("POST /ask (oversized) ->", r.status_code, r.json())

    print("\nWebSocket multi-turn, then abrupt disconnect:")
    with client.websocket_connect("/ws/chat?session_id=ws-1") as ws:
        for q in ["What is the status of order NYK-0054?", "Is it eligible for a return?"]:
            ws.send_text(q); m = ws.receive_json()
            print(f"  USER: {q}\n  AGENT: {m['answer']}")
    # leaving the 'with' block closes the socket -> server must catch WebSocketDisconnect
    print("after disconnect, server still healthy:", client.get("/health").json())
    with client.websocket_connect("/ws/chat?session_id=ws-2") as ws:
        ws.send_text("Can I cancel a shipped order?"); print("  new client served:", ws.receive_json()["answer"][:80], "...")

get_agent().kb.remove_document("13_gift_wrapping")   # keep the KB clean for later scripts

lines = REQUEST_LOG.read_text().splitlines()[n_before:]
print(f"\n{len(lines)} new JSON-Lines log entries; sample:")
for ln in lines[:3]:
    print(" ", ln)
assert all(json.loads(l)["trace_id"] for l in lines)
