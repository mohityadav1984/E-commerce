"""FastAPI deployment (Task 11). Run:  uvicorn nykaa_agent.api:app --port 8000"""
from __future__ import annotations

from . import config  # noqa: F401
import asyncio
import uuid

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from .governance import BudgetExceeded
from .guardrails import detect_injection
from .logging_setup import log_request
from .pipeline import get_agent
from .schemas import (AddDocumentRequest, AddDocumentResponse, AskRequest, AskResponse)

app = FastAPI(title="Nykaa Domain Support Agent", version="1.0")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    try:
        res = await asyncio.to_thread(get_agent().ask, req.question, req.session_id, "POST /ask")
    except BudgetExceeded as exc:
        raise HTTPException(status_code=413, detail=f"budget_exceeded: {exc}")
    return AskResponse(**res)


@app.post("/add-document", response_model=AddDocumentResponse)
async def add_document(req: AddDocumentRequest) -> AddDocumentResponse:
    trace_id = uuid.uuid4().hex
    if detect_injection(req.text) or detect_injection(req.title):
        log_request({"trace_id": trace_id, "endpoint": "POST /add-document",
                     "doc_id": req.doc_id, "status": "blocked_injection"})
        raise HTTPException(status_code=400, detail="document rejected: prompt-injection pattern")
    counts = await asyncio.to_thread(get_agent().kb.add_document, req.doc_id, req.title, req.text)
    log_request({"trace_id": trace_id, "endpoint": "POST /add-document",
                 "doc_id": req.doc_id, "status": "ok", "chunks_indexed": counts})
    return AddDocumentResponse(trace_id=trace_id, doc_id=req.doc_id, chunks_indexed=counts)


@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket, session_id: str | None = None) -> None:
    """Multi-turn chat. A disconnect only ends THIS connection; the server keeps serving."""
    await ws.accept()
    sid = session_id or f"ws-{uuid.uuid4().hex[:8]}"
    try:
        while True:
            text = await ws.receive_text()
            try:
                res = await asyncio.to_thread(get_agent().ask, text, sid, "WS /ws/chat")
                await ws.send_json(AskResponse(**res).model_dump())
            except BudgetExceeded as exc:
                await ws.send_json({"error": "budget_exceeded", "detail": str(exc)})
            except WebSocketDisconnect:
                raise
            except Exception as exc:       # one bad message must not kill the socket
                await ws.send_json({"error": "internal_error", "detail": type(exc).__name__})
    except (WebSocketDisconnect, RuntimeError):
        log_request({"trace_id": uuid.uuid4().hex, "endpoint": "WS /ws/chat",
                     "session_id": sid, "status": "client_disconnected"})
