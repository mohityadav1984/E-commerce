"""RAG core: SentenceTransformers embeddings + two ChromaDB collections
(fixed-overlap and sentence chunks), retrieval, grounded generation, caching."""
from __future__ import annotations

from . import config  # noqa: F401  (must be first: sets env vars)
from .cache import ResponseCache
from .chunking import chunk_document, load_documents
from .config import (CHROMA_DIR, COLLECTIONS, EMBED_MODEL, TOP_K,
                     active_strategy, load_threshold)
from .llm import FALLBACK, generate_grounded


class Counters:
    def __init__(self):
        self.retrieval_calls = 0
        self.generation_calls = 0

    def snapshot(self) -> dict:
        return {"retrieval_calls": self.retrieval_calls, "generation_calls": self.generation_calls}


class KnowledgeBase:
    def __init__(self, persist: bool = True):
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(EMBED_MODEL)
        self.client = (chromadb.PersistentClient(path=str(CHROMA_DIR)) if persist
                       else chromadb.EphemeralClient())
        self.collections = {
            s: self.client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
            for s, name in COLLECTIONS.items()
        }
        self.cache = ResponseCache()
        self.counters = Counters()
        self.docs: dict[str, dict] = {}

    # ---------------------------------------------------------------- indexing
    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def index_document(self, doc: dict) -> dict[str, int]:
        self.docs[doc["doc_id"]] = doc
        counts = {}
        for strategy, col in self.collections.items():
            chunks = chunk_document(doc, strategy)
            col.delete(where={"doc_id": doc["doc_id"]})   # drop stale chunks (e.g. after re-chunking)
            col.upsert(
                ids=[c["chunk_id"] for c in chunks],
                documents=[c["text"] for c in chunks],
                embeddings=self._embed([c["text"] for c in chunks]),
                metadatas=[{"doc_id": c["doc_id"], "title": c["title"],
                            "strategy": strategy, "chunk_index": c["chunk_index"]} for c in chunks],
            )
            counts[strategy] = len(chunks)
        return counts

    def build(self, docs: list[dict] | None = None) -> dict[str, int]:
        totals = {s: 0 for s in COLLECTIONS}
        for doc in docs or load_documents():
            for s, n in self.index_document(doc).items():
                totals[s] += n
        return totals

    def add_document(self, doc_id: str, title: str, text: str) -> dict[str, int]:
        counts = self.index_document({"doc_id": doc_id, "title": title, "text": text})
        self.cache.clear()      # cached answers may now be stale
        return counts

    def remove_document(self, doc_id: str) -> None:
        for col in self.collections.values():
            col.delete(where={"doc_id": doc_id})
        self.docs.pop(doc_id, None)
        self.cache.clear()

    # --------------------------------------------------------------- retrieval
    def retrieve(self, query: str, strategy: str | None = None, k: int = TOP_K) -> list[dict]:
        strategy = strategy or active_strategy()
        self.counters.retrieval_calls += 1
        res = self.collections[strategy].query(
            query_embeddings=self._embed([query]), n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {"chunk_id": cid, "doc_id": meta["doc_id"], "text": doc,
             "score": round(1.0 - dist, 4)}           # cosine similarity
            for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                            res["metadatas"][0], res["distances"][0])
        ]

    # -------------------------------------------------------------- generation
    def grounded_answer(self, query: str, strategy: str | None = None, k: int = TOP_K,
                        threshold: float | None = None, use_cache: bool = True) -> dict:
        strategy = strategy or active_strategy()
        threshold = load_threshold(strategy) if threshold is None else threshold
        key = self.cache.key(query, strategy, k, threshold)
        if use_cache:
            cached = self.cache.get(key)
            if cached is not None:
                cached["cache_hit"] = True
                return cached

        hits = self.retrieve(query, strategy, k)
        top = hits[0]["score"] if hits else 0.0
        if top < threshold:
            result = {"answer": FALLBACK, "grounded": False, "top_score": top,
                      "threshold": threshold, "sources": [], "chunks": hits}
        else:
            used = [h for h in hits if h["score"] >= threshold]
            self.counters.generation_calls += 1
            answer = generate_grounded(query, [h["text"] for h in used])
            grounded = answer != FALLBACK          # ranker found no sentence that answers it
            sources = list(dict.fromkeys(h["doc_id"] for h in used)) if grounded else []
            result = {"answer": answer, "grounded": grounded, "top_score": top,
                      "threshold": threshold, "sources": sources, "chunks": used}
        result["cache_hit"] = False
        if use_cache:
            self.cache.put(key, result)
        return result


_KB: KnowledgeBase | None = None


def get_kb() -> KnowledgeBase:
    """Process-wide singleton; (re)indexes the KB on first use (upsert = idempotent)."""
    global _KB
    if _KB is None:
        _KB = KnowledgeBase()
        _KB.build()
    return _KB
