"""CrewAI tool wrappers. The RAG tool and the order-lookup tool."""
from __future__ import annotations

from . import config  # noqa: F401
from collections import Counter
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from . import markers
from .rag import get_kb
from .tools import check_order_status

TOOL_CALLS: Counter = Counter()   # evidence that each tool really ran


class RagInput(BaseModel):
    query: str = Field(..., description="The customer's policy question, in plain text.")


class OrderInput(BaseModel):
    record_id: str = Field(..., description="Order ID such as NYK-0007.")


class RagLookupTool(BaseTool):
    name: str = "rag_lookup"
    description: str = ("Search the Nykaa support policy knowledge base and return a grounded answer "
                        "with sources. Use for policy questions (returns, refunds, delivery, warranty...).")
    args_schema: Type[BaseModel] = RagInput

    def _run(self, query: str) -> str:
        TOOL_CALLS["rag_lookup"] += 1
        res = get_kb().grounded_answer(query)
        payload = {k: res[k] for k in ("answer", "grounded", "top_score", "threshold",
                                       "sources", "cache_hit")}
        payload["context"] = [c["text"] for c in res["chunks"]]
        return markers.encode(markers.RAG, payload)


class OrderStatusTool(BaseTool):
    name: str = "check_order_status"
    description: str = ("Look up one order by its ID and return its status, order value and an "
                        "escalation score. Use ONLY when the customer mentions an order ID.")
    args_schema: Type[BaseModel] = OrderInput

    def _run(self, record_id: str) -> str:
        TOOL_CALLS["check_order_status"] += 1
        return markers.encode(markers.ORDER, check_order_status(record_id))
