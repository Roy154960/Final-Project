"""
Compare all chunking methods on the same document(s) and report simple,
objective metrics so you can pick a method with evidence instead of a guess.

Metrics reported per method:
  - chunk count
  - avg / min / max chunk size (words)
  - std dev of chunk size (lower = more consistent, which can matter for
    downstream embedding truncation limits)
  - wall-clock time to chunk

Run (re-ingests from source every time):
    python -m chunking.benchmark_chunkers path/to/file_or_dir

Run (loads a previously-cached ingest instead of re-parsing/re-OCRing —
see ingestion/cache.py to build the cache first):
    python -m chunking.benchmark_chunkers --cache data/ingested_docs_cache.json
"""

import argparse
import sys
import time
import statistics

from ingestion.loader import ingest_directory, ingest_path
from chunking.fixed_size import chunk_fixed_size
from chunking.recursive import chunk_recursive
from chunking.sentence_based import chunk_sentence_based
from chunking.structure_aware import chunk_markdown_by_heading


def _stats(chunks) -> dict:
    sizes = [len(c.text.split()) for c in chunks] or [0]
    return {
        "count": len(chunks),
        "avg_words": round(statistics.mean(sizes), 1),
        "min_words": min(sizes),
        "max_words": max(sizes),
        "std_words": round(statistics.pstdev(sizes), 1) if len(sizes) > 1 else 0.0,
    }


def benchmark(target: str = None, cache_path: str = None):
    if cache_path:
        from ingestion.cache import load_raw_documents
        docs = load_raw_documents(cache_path)
    else:
        from pathlib import Path
        p = Path(target)
        docs = ingest_directory(target) if p.is_dir() else ingest_path(target)

    text_docs = [d for d in docs if d.modality in ("text", "pdf_text")]
    if not text_docs:
        print("No text documents found to chunk.")
        return

    methods = {
        "fixed_size": chunk_fixed_size,
        "recursive": chunk_recursive,
        "sentence_based": chunk_sentence_based,
        "structure_aware": chunk_markdown_by_heading,
        # semantic chunking is intentionally excluded from this benchmark:
        # it requires an embedding model call per sentence and is compared
        # separately once you've picked an embedder (see embeddings/benchmark_embedders.py)
    }

    print(f"Benchmarking {len(methods)} chunking methods over {len(text_docs)} document(s)\n")
    print(f"{'method':<18}{'count':<8}{'avg':<8}{'min':<6}{'max':<6}{'std':<8}{'time(s)':<8}")

    for name, fn in methods.items():
        start = time.perf_counter()
        all_chunks = []
        for doc in text_docs:
            all_chunks.extend(fn(doc))
        elapsed = time.perf_counter() - start
        s = _stats(all_chunks)
        print(
            f"{name:<18}{s['count']:<8}{s['avg_words']:<8}{s['min_words']:<6}"
            f"{s['max_words']:<6}{s['std_words']:<8}{elapsed:<8.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", default="data/raw", help="Folder to ingest (ignored if --cache is used)")
    parser.add_argument("--cache", default=None, help="Path to a cached ingest from ingestion/cache.py — skips re-ingesting")
    args = parser.parse_args()
    benchmark(args.source, cache_path=args.cache)
