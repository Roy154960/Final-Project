"""
Compare ChromaDB and Qdrant on the same synthetic dataset:
  - bulk upsert latency
  - single-query latency (avg over N queries)
  - result quality sanity check (nearest neighbor of a vector should be
    itself when it's already in the store)

Qdrant requires a running local server:
    docker run -p 6333:6333 qdrant/qdrant
ChromaDB needs nothing extra (embedded).

Run:
    python -m vectorstore.benchmark_stores
"""

import time
import uuid
import numpy as np

N_VECTORS = 500
DIM = 384
N_QUERIES = 20


def _make_dataset(n=N_VECTORS, dim=DIM, seed=0):
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, dim)).astype(np.float32)
    texts = [f"synthetic chunk {i}" for i in range(n)]
    metadatas = [{"idx": i} for i in range(n)]
    return vectors, texts, metadatas


def _bench_chroma():
    from vectorstore.chroma_store import ChromaStore
    store = ChromaStore(collection_name=f"bench_{uuid.uuid4().hex[:8]}")
    vectors, texts, metadatas = _make_dataset()
    ids = [str(uuid.uuid4()) for _ in range(len(vectors))]

    start = time.perf_counter()
    store.upsert(ids, vectors, texts, metadatas)
    upsert_time = time.perf_counter() - start

    start = time.perf_counter()
    correct = 0
    for i in range(N_QUERIES):
        result = store.query(vectors[i], top_k=1)
        if result and result[0]["id"] == ids[i]:
            correct += 1
    query_time = (time.perf_counter() - start) / N_QUERIES

    return {
        "store": "chromadb",
        "upsert_s_total": round(upsert_time, 3),
        "upsert_s_per_1k": round(upsert_time / (N_VECTORS / 1000), 3),
        "query_ms_avg": round(query_time * 1000, 2),
        "self_match_accuracy": correct / N_QUERIES,
    }


def _bench_qdrant():
    from vectorstore.qdrant_store import QdrantStore
    store = QdrantStore(collection_name=f"bench_{uuid.uuid4().hex[:8]}", dimensions=DIM)
    vectors, texts, metadatas = _make_dataset()
    ids = list(range(len(vectors)))  # Qdrant needs int or UUID ids

    start = time.perf_counter()
    store.upsert(ids, vectors, texts, metadatas)
    upsert_time = time.perf_counter() - start

    start = time.perf_counter()
    correct = 0
    for i in range(N_QUERIES):
        result = store.query(vectors[i], top_k=1)
        if result and result[0]["id"] == str(ids[i]):
            correct += 1
    query_time = (time.perf_counter() - start) / N_QUERIES

    return {
        "store": "qdrant",
        "upsert_s_total": round(upsert_time, 3),
        "upsert_s_per_1k": round(upsert_time / (N_VECTORS / 1000), 3),
        "query_ms_avg": round(query_time * 1000, 2),
        "self_match_accuracy": correct / N_QUERIES,
    }


def benchmark():
    results = []
    try:
        results.append(_bench_chroma())
    except Exception as e:
        print(f"[skip] ChromaDB: {e}")

    try:
        results.append(_bench_qdrant())
    except Exception as e:
        print(f"[skip] Qdrant (is a local server running? `docker run -p 6333:6333 qdrant/qdrant`): {e}")

    print(f"\n{'store':<12}{'upsert_s_/1k':<15}{'query_ms_avg':<15}{'self_match_acc':<15}")
    for r in results:
        print(f"{r['store']:<12}{r['upsert_s_per_1k']:<15}{r['query_ms_avg']:<15}{r['self_match_accuracy']:<15}")
    return results


if __name__ == "__main__":
    benchmark()
