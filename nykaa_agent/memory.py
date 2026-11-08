"""Session memory (Task 8): LangChain InMemoryChatMessageHistory +
RunnableWithMessageHistory. In-process only, which the brief says is sufficient.
(RunnableWithMessageHistory emits a LangChainDeprecationWarning - expected.)"""
from __future__ import annotations

import re
from typing import Callable

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from .tools import ORDER_ID_RE

_STORE: dict[str, InMemoryChatMessageHistory] = {}
_FOLLOW_UP = re.compile(r"\b(it|that order|this order|the same order|my order|its)\b", re.I)


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _STORE:
        _STORE[session_id] = InMemoryChatMessageHistory()
    return _STORE[session_id]


def reset_session(session_id: str) -> None:
    _STORE.pop(session_id, None)


def build_chat_runnable(fn: Callable) -> RunnableWithMessageHistory:
    """`fn(inputs: dict, config) -> str`; inputs = {'input': str, 'history': [BaseMessage]}."""
    return RunnableWithMessageHistory(
        RunnableLambda(fn), get_session_history,
        input_messages_key="input", history_messages_key="history")


def resolve_with_history(question: str, history: list) -> tuple[str, str | None]:
    """If the user refers back ('is it eligible for return?') and names no order ID,
    carry the most recent order ID from earlier turns. Returns (question, carried_id)."""
    if ORDER_ID_RE.search(question) or not _FOLLOW_UP.search(question):
        return question, None
    for msg in reversed(history):
        if getattr(msg, "type", "") == "human":
            found = ORDER_ID_RE.findall(msg.content)
            if found:
                oid = found[-1].upper()
                return f"{question} (regarding order {oid})", oid
    return question, None
