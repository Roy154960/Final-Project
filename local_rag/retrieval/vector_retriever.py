"""
Retrieve step - method 1: pure vector (semantic) search.
Embed the query, search the vector store for nearest neighbors.
"""

from embeddings.base import BaseEmbedder
from vectorstore.base import BaseVectorStore


def vector_retrieve(query: str, embedder: BaseEmbedder, store: BaseVectorStore, top_k: int = 5) -> list[dict]:
    query_vec = embedder.embed_texts([query])[0]
    return store.query(query_vec, top_k=top_k)


if __name__ == "__main__":
    print("This module is meant to be imported. See retrieval/benchmark_retrieval.py for a runnable comparison.")
