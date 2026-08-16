"""
Retrieve step - method 2: hybrid search.
Combines vector (semantic) search with BM25 (keyword/lexical) search using
Reciprocal Rank Fusion (RRF). Catches exact names, codes, and IDs that pure
vector search sometimes misses.

Note: this is a client-side hybrid implementation using rank-bm25 so it
works with ANY vector store (Chroma included). Qdrant and Weaviate support
hybrid search natively server-side if you want to skip this layer later.
"""

from rank_bm25 import BM25Okapi

from embeddings.base import BaseEmbedder
from vectorstore.base import BaseVectorStore


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class HybridRetriever:
    def __init__(self, embedder: BaseEmbedder, store: BaseVectorStore, corpus: list[dict]):
        """
        corpus: list of {"id": ..., "text": ...} — the full chunk set, needed
        client-side because BM25 requires the whole corpus tokenized upfront.
        """
        self.embedder = embedder
        self.store = store
        self.corpus = corpus
        self._bm25 = BM25Okapi([_tokenize(c["text"]) for c in corpus])

    def retrieve(self, query: str, top_k: int = 5, vector_k: int = 20, bm25_k: int = 20, rrf_k: int = 60) -> list[dict]:
        # Vector side
        query_vec = self.embedder.embed_texts([query])[0]
        vector_results = self.store.query(query_vec, top_k=vector_k)
        vector_ranks = {r["id"]: rank for rank, r in enumerate(vector_results)}

        # BM25 side
        bm25_scores = self._bm25.get_scores(_tokenize(query))
        bm25_ranked_idx = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])[:bm25_k]
        bm25_ranks = {self.corpus[i]["id"]: rank for rank, i in enumerate(bm25_ranked_idx)}

        # Reciprocal Rank Fusion
        all_ids = set(vector_ranks) | set(bm25_ranks)
        fused = []
        text_by_id = {r["id"]: r["text"] for r in vector_results}
        text_by_id.update({c["id"]: c["text"] for c in self.corpus if c["id"] in all_ids})
        # Metadata (filename, page, ...) was previously dropped entirely here —
        # every hybrid result silently lost source attribution downstream.
        # vector_results carry metadata from the store; corpus entries (from
        # store.get_all()) do too, so fall back to corpus for BM25-only hits.
        meta_by_id = {r["id"]: r.get("metadata", {}) for r in vector_results}
        meta_by_id.update({c["id"]: c.get("metadata", {}) for c in self.corpus if c["id"] in all_ids and c["id"] not in meta_by_id})

        for doc_id in all_ids:
            score = 0.0
            if doc_id in vector_ranks:
                score += 1.0 / (rrf_k + vector_ranks[doc_id])
            if doc_id in bm25_ranks:
                score += 1.0 / (rrf_k + bm25_ranks[doc_id])
            fused.append({
                "id": doc_id,
                "text": text_by_id.get(doc_id, ""),
                "score": score,
                "metadata": meta_by_id.get(doc_id, {}),
            })

        fused.sort(key=lambda r: -r["score"])
        return fused[:top_k]


def hybrid_retrieve(query: str, embedder: BaseEmbedder, store: BaseVectorStore, top_k: int = 5,
                     vector_k: int = 20, bm25_k: int = 20, rrf_k: int = 60) -> list[dict]:
    """
    Convenience wrapper matching the vector_retrieve(query, embedder, store, top_k)
    shape used elsewhere (pipeline.py, the REST API) — pulls the full corpus via
    store.get_all() so callers don't need to manage it themselves. For very large
    corpora where fetching everything client-side gets expensive, build a
    HybridRetriever once and reuse it instead of calling this per-query.
    """
    corpus = store.get_all()
    if not corpus:
        return []
    retriever = HybridRetriever(embedder, store, corpus)
    return retriever.retrieve(query, top_k=top_k, vector_k=vector_k, bm25_k=bm25_k, rrf_k=rrf_k)


if __name__ == "__main__":
    print("This module is meant to be imported. See retrieval/benchmark_retrieval.py for a runnable comparison.")
