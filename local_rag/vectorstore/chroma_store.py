"""
Store step - ChromaDB (embedded, zero setup, local files, free).
Good for prototyping and small/medium projects.

Run directly to smoke-test:
    python -m vectorstore.chroma_store
"""

import numpy as np
import chromadb
from vectorstore.base import BaseVectorStore
from config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION,
    CHROMA_CLIENT_MODE,
    CHROMA_SERVER_HOST,
    CHROMA_SERVER_PORT,
)


def _build_chroma_client(persist_dir: str = None):
    """
    Single choke point every ChromaStore instance builds its client
    through, switched by config.CHROMA_CLIENT_MODE (env var
    CHROMA_CLIENT_MODE) -- see that constant's own docstring in config.py
    for the "embedded" vs "http" tradeoff. Centralized here rather than
    left to each caller so switching modes for the whole project is one
    env var, not a per-call-site edit -- every one of this project's
    ChromaStore(...) call sites (personal_rag.py, mcp_server/server.py,
    load_step.py, pipeline.py, the benchmark scripts, ...) goes through
    this same function via __init__ below.

    `persist_dir` only matters in "embedded" mode; it's a no-op in "http"
    mode since a Chroma server process, not this client, owns the actual
    on-disk path in that mode.
    """
    if CHROMA_CLIENT_MODE == "http":
        return chromadb.HttpClient(host=CHROMA_SERVER_HOST, port=CHROMA_SERVER_PORT)
    return chromadb.PersistentClient(path=str(persist_dir or CHROMA_PERSIST_DIR))


class ChromaStore(BaseVectorStore):
    def __init__(self, collection_name: str = CHROMA_COLLECTION, persist_dir: str = None):
        self.name = "chromadb"
        self._client = _build_chroma_client(persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, ids, vectors: np.ndarray, texts, metadatas) -> None:
        # Chroma's SQLite backend caps how many rows fit in a single upsert
        # call (commonly ~5,461 — a limit from SQLite's own parameter count,
        # not a Chroma design choice). A parent_child-sized corpus (15k+
        # chunks) blows straight through that in one call, so this batches
        # automatically instead of leaving it to the caller to know Chroma's
        # internal limit and split manually.
        try:
            max_batch = self._client.get_max_batch_size()
        except Exception:
            max_batch = 4000  # conservative fallback if this chromadb version doesn't expose it

        n = len(ids)
        for start in range(0, n, max_batch):
            end = min(start + max_batch, n)
            self._collection.upsert(
                ids=ids[start:end],
                embeddings=vectors[start:end].tolist(),
                documents=texts[start:end],
                metadatas=metadatas[start:end],
            )

    def query(self, vector: np.ndarray, top_k: int = 5, where: dict = None) -> list[dict]:
        result = self._collection.query(
            query_embeddings=[vector.tolist()],
            n_results=top_k,
            where=where,
        )
        out = []
        for i in range(len(result["ids"][0])):
            out.append({
                "id": result["ids"][0][i],
                "text": result["documents"][0][i],
                "score": 1 - result["distances"][0][i],  # cosine distance -> similarity
                "metadata": result["metadatas"][0][i],
            })
        return out

    def count(self) -> int:
        return self._collection.count()

    def delete(self, ids: list[str]) -> None:
        if ids:
            self._collection.delete(ids=ids)

    def delete_where(self, where: dict) -> None:
        """
        Remove every record matching a metadata filter, without the
        caller needing to know their ids first -- unlike delete() above
        (id-keyed, used by incremental re-indexing, which already has
        the exact ids of stale entries to remove), this is for the
        "forget everything tagged X" case, e.g.
        personal_rag.py's delete_thread_data(thread_id) removing every
        chunk uploaded into a conversation without first running a
        query to find their ids. Not part of BaseVectorStore -- this is
        a Chroma-specific convenience (the `where` shape is Chroma's
        own), same spirit as this class's own batching workaround above
        already being Chroma-specific rather than part of the shared
        interface.
        """
        if where:
            self._collection.delete(where=where)

    def get_all(self) -> list[dict]:
        result = self._collection.get(include=["documents", "metadatas"])
        return [
            {"id": result["ids"][i], "text": result["documents"][i], "metadata": result["metadatas"][i]}
            for i in range(len(result["ids"]))
        ]

    def get_where(self, where: dict) -> list[dict]:
        """
        Fetch every record matching a metadata filter directly -- no
        query vector, no similarity ranking, unlike query() above. Same
        shape as get_all(), just filtered, and same Chroma-native `where`
        shape delete_where() already uses.

        Exists for lookups where "closest match" is the WRONG question --
        e.g. local_rag/personal_rag.py's latest_uploaded_image(thread_id):
        "which image did this thread upload most recently" has nothing to
        do with embedding similarity to whatever the person's current
        message says, and forcing it through query() would mean
        comparing a text embedding of the CURRENT question against the
        image's caption embedding, which is exactly the kind of "merely
        resembles" ranking this project's own personal-upload feature was
        built to avoid relying on for "show me what I actually sent."
        """
        if not where:
            return []
        result = self._collection.get(where=where, include=["documents", "metadatas"])
        return [
            {"id": result["ids"][i], "text": result["documents"][i], "metadata": result["metadatas"][i]}
            for i in range(len(result["ids"]))
        ]


if __name__ == "__main__":
    store = ChromaStore(collection_name="smoke_test")
    dummy_vecs = np.random.default_rng(0).normal(size=(3, 8)).astype(np.float32)
    store.upsert(
        ids=["a", "b", "c"],
        vectors=dummy_vecs,
        texts=["chunk A", "chunk B", "chunk C"],
        metadatas=[{"source": "test"}] * 3,
    )
    print(f"count={store.count()}")
    print(store.query(dummy_vecs[0], top_k=2))
