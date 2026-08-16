"""
Store step - Qdrant (client-server, production-grade, Rust-based, free
open-source; run locally via `docker run -p 6333:6333 qdrant/qdrant`).

Run directly to smoke-test (requires Qdrant running locally):
    python -m vectorstore.qdrant_store
"""

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from vectorstore.base import BaseVectorStore
from config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION


class QdrantStore(BaseVectorStore):
    def __init__(self, collection_name: str = QDRANT_COLLECTION, dimensions: int = 384,
                 host: str = QDRANT_HOST, port: int = QDRANT_PORT):
        self.name = "qdrant"
        self._client = QdrantClient(host=host, port=port)
        existing = [c.name for c in self._client.get_collections().collections]
        if collection_name not in existing:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dimensions, distance=Distance.COSINE),
            )
        self._collection_name = collection_name

    def upsert(self, ids, vectors: np.ndarray, texts, metadatas) -> None:
        # Batched for the same reason as ChromaStore.upsert(): a parent_child-sized
        # corpus (15k+ chunks) sent as one request risks Qdrant's request-size
        # limits and a large memory spike client-side, even though Qdrant has
        # no fixed hard cap the way Chroma's SQLite backend does. 500 matches
        # Qdrant's own commonly-recommended batch size for indexing throughput.
        batch_size = 500
        n = len(ids)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            points = [
                PointStruct(
                    id=ids[i],
                    vector=vectors[i].tolist(),
                    payload={"text": texts[i], **metadatas[i]},
                )
                for i in range(start, end)
            ]
            self._client.upsert(collection_name=self._collection_name, points=points)

    def query(self, vector: np.ndarray, top_k: int = 5, where: dict = None) -> list[dict]:
        results = self._client.query_points(
            collection_name=self._collection_name,
            query=vector.tolist(),
            limit=top_k,
        ).points
        out = []
        for r in results:
            payload = dict(r.payload)
            text = payload.pop("text", "")
            out.append({"id": str(r.id), "text": text, "score": r.score, "metadata": payload})
        return out

    def count(self) -> int:
        return self._client.count(collection_name=self._collection_name).count

    def delete(self, ids: list[str]) -> None:
        if ids:
            self._client.delete(collection_name=self._collection_name, points_selector=ids)

    def get_all(self) -> list[dict]:
        out = []
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self._collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                payload = dict(p.payload or {})
                text = payload.pop("text", "")
                out.append({"id": str(p.id), "text": text, "metadata": payload})
            if offset is None:
                break
        return out


if __name__ == "__main__":
    # Note: Qdrant point IDs must be an unsigned int or a UUID string —
    # arbitrary strings (like our Chunk.chunk_id UUIDs) work fine since
    # uuid.uuid4() already produces valid UUID strings; plain "1"/"2"/"3"
    # would NOT work as string IDs, so we use ints here for the smoke test.
    store = QdrantStore(collection_name="smoke_test", dimensions=8)
    dummy_vecs = np.random.default_rng(0).normal(size=(3, 8)).astype(np.float32)
    store.upsert(
        ids=[1, 2, 3],
        vectors=dummy_vecs,
        texts=["chunk A", "chunk B", "chunk C"],
        metadatas=[{"source": "test"}] * 3,
    )

    print(f"count={store.count()}")
    print(store.query(dummy_vecs[0], top_k=2))
