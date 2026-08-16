"""
Compare chunking methods on ACTUAL RETRIEVAL QUALITY, not just chunk size
stats (that's what benchmark_chunkers.py does — this is the harder,
more useful question: does this chunking method retrieve the right
content for a real question?).

Why this needs its own script instead of reusing retrieval/benchmark_retrieval.py:
chunk IDs are meaningless across chunking methods — a "fixed_size" chunk
covering a sentence and a "semantic" chunk covering that same sentence have
completely different IDs, so an eval set built around specific chunk IDs
(like evaluation/build_eval_set.py produces) can't be reused here. Instead,
this checks whether the retrieved text CONTAINS expected keywords/phrases —
a method-agnostic relevance check, the same pattern vlm/benchmark_vlms.py
uses for image QA.

Process, per chunking method:
  1. Chunk your real documents with that method
  2. Embed every chunk, store in its own vector collection (so methods
     don't contaminate each other's results)
  3. For each (query, expected_keywords) example, retrieve top-k and check
     what fraction of expected keywords appear somewhere in the retrieved text
  4. Report keyword hit rate, chunk count (storage cost), and latency

FILL IN EVAL_QUERIES BELOW WITH REAL EXAMPLES FROM YOUR OWN CORPUS before
trusting these numbers — the placeholder is a smoke test, not a real signal.

Run (re-ingests from source every time):
    python -m chunking.benchmark_chunking_retrieval data/raw

Run (loads a previously-cached ingest instead of re-parsing/re-OCRing —
see ingestion/cache.py to build the cache first):
    python -m chunking.benchmark_chunking_retrieval --cache data/ingested_docs_cache.json
"""

import argparse
import sys
import time
import uuid

from ingestion.loader import ingest_directory, ingest_path
from chunking.fixed_size import chunk_fixed_size
from chunking.recursive import chunk_recursive
from chunking.sentence_based import chunk_sentence_based
from chunking.structure_aware import chunk_markdown_by_heading
from chunking.semantic import chunk_semantic

# Replace with real (question, expected_keywords) pairs from YOUR corpus —
# e.g. for a painting treatise: {"query": "How do you prepare a canvas for
# oil painting?", "expected_keywords": ["canvas", "priming", "ground"]}.
# Keywords should be distinctive words you KNOW appear in the passage that
# answers the question — not the question's own wording.
EVAL_QUERIES: list[dict] = [
    {"query": "What is the capital of France?", "expected_keywords": ["paris", "capital"]},
]


def _chunk_with_method(method_name: str, doc, embed_fn=None) -> list:
    if method_name == "fixed_size":
        return chunk_fixed_size(doc)
    if method_name == "recursive":
        return chunk_recursive(doc)
    if method_name == "sentence_based":
        return chunk_sentence_based(doc)
    if method_name == "structure_aware":
        # Only meaningful for documents with markdown headings — for plain
        # PDF page text with no headers, this collapses to one chunk per
        # page (see chunking/structure_aware.py's "whole_doc" fallback),
        # which is a legitimate but less interesting comparison point.
        return chunk_markdown_by_heading(doc)
    if method_name == "semantic":
        if embed_fn is None:
            raise ValueError("semantic chunking needs an embed_fn")
        return chunk_semantic(doc, embed_fn=embed_fn)
    raise ValueError(f"Unknown method: {method_name}")


def _keyword_hit_rate(text: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 1.0
    text_lower = text.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in text_lower)
    return hits / len(expected_keywords)


def build_index_for_method(method_name: str, text_docs: list, embedder) -> tuple:
    """Chunks every document with this method, embeds all chunks, stores
    them in a collection unique to this method. Returns (store, n_chunks)."""
    from vectorstore.chroma_store import ChromaStore

    embed_fn = embedder.embed_texts if method_name == "semantic" else None

    all_chunks = []
    for doc in text_docs:
        all_chunks.extend(_chunk_with_method(method_name, doc, embed_fn=embed_fn))

    if not all_chunks:
        return None, 0

    collection_name = f"chunkeval_{method_name}_{uuid.uuid4().hex[:8]}"
    store = ChromaStore(collection_name=collection_name)

    vectors = embedder.embed_texts([c.text for c in all_chunks])
    ids = [c.chunk_id for c in all_chunks]
    texts = [c.text for c in all_chunks]
    metadatas = [c.metadata for c in all_chunks]
    store.upsert(ids=ids, vectors=vectors, texts=texts, metadatas=metadatas)

    return store, len(all_chunks)


def evaluate_method(store, embedder, eval_queries: list[dict], top_k: int = 5) -> dict:
    hit_rates = []
    latencies = []

    for example in eval_queries:
        start = time.perf_counter()
        query_vec = embedder.embed_texts([example["query"]])[0]
        results = store.query(query_vec, top_k=top_k)
        elapsed = time.perf_counter() - start

        combined_text = " ".join(r["text"] for r in results)
        hit_rates.append(_keyword_hit_rate(combined_text, example["expected_keywords"]))
        latencies.append(elapsed)

    return {
        "avg_keyword_hit_rate": round(sum(hit_rates) / len(hit_rates), 2) if hit_rates else 0.0,
        "avg_latency_ms": round((sum(latencies) / len(latencies)) * 1000, 1) if latencies else 0.0,
    }


def benchmark(source: str = None, cache_path: str = None, embedder_name: str = "hf",
              top_k: int = 5, eval_queries: list[dict] = None):
    from pipeline import get_embedder

    eval_queries = eval_queries or EVAL_QUERIES
    embedder = get_embedder(embedder_name)

    if cache_path:
        from ingestion.cache import load_raw_documents
        docs = load_raw_documents(cache_path)
    else:
        docs = ingest_directory(source) if not source.endswith((".pdf", ".txt", ".md")) else ingest_path(source)

    text_docs = [d for d in docs if d.modality in ("text", "pdf_text")]
    if not text_docs:
        print("No text documents found.")
        return

    methods = ["fixed_size", "recursive", "sentence_based", "structure_aware", "semantic"]
    results = []

    for method_name in methods:
        print(f"Chunking + embedding with '{method_name}'...")
        start = time.perf_counter()
        store, n_chunks = build_index_for_method(method_name, text_docs, embedder)
        index_time = time.perf_counter() - start

        if store is None:
            print(f"  [skip] no chunks produced")
            continue

        metrics = evaluate_method(store, embedder, eval_queries, top_k=top_k)
        results.append({
            "method": method_name,
            "n_chunks": n_chunks,
            "index_time_s": round(index_time, 2),
            **metrics,
        })

    print(f"\n{'method':<18}{'n_chunks':<11}{'index_s':<10}{'hit_rate':<11}{'latency_ms':<12}")
    for r in results:
        print(f"{r['method']:<18}{r['n_chunks']:<11}{r['index_time_s']:<10}"
              f"{r['avg_keyword_hit_rate']:<11}{r['avg_latency_ms']:<12}")

    if len(eval_queries) < 10:
        print(f"\nNote: only {len(eval_queries)} eval example(s) — fill in EVAL_QUERIES at the "
              "top of this file with 15-30 real (question, expected_keywords) pairs from your "
              "own corpus before trusting these numbers. A single example is a smoke test.")

    if results:
        best = max(results, key=lambda r: r["avg_keyword_hit_rate"])
        smallest = min(results, key=lambda r: r["n_chunks"])
        print(f"\nHighest keyword hit rate: {best['method']} ({best['avg_keyword_hit_rate']})")
        print(f"Fewest chunks (lowest storage cost): {smallest['method']} ({smallest['n_chunks']})")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", default="data/raw", help="Folder to ingest (ignored if --cache is used)")
    parser.add_argument("--cache", default=None, help="Path to a cached ingest from ingestion/cache.py — skips re-ingesting")
    parser.add_argument("--embedder", choices=["hf", "ollama", "clip"], default="hf")
    parser.add_argument("--top-k", dest="top_k", type=int, default=5)
    args = parser.parse_args()
    benchmark(args.source, cache_path=args.cache, embedder_name=args.embedder, top_k=args.top_k)
