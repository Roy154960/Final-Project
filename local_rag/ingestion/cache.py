"""
Ingestion cache - save/load ingested RawDocuments to/from disk.

Ingestion (especially OCR on scanned PDFs) can be slow — re-running it
every time you want to test a downstream step (chunking, embedding) wastes
real time on unchanged files. This lets you ingest once, then have every
downstream script load from the cached result instead of re-parsing PDFs.

Note: for ongoing work where files actually change over time, prefer
ingestion/incremental_indexer.py instead — this cache is a simple "ingest
once, reuse the result" snapshot, not a change-tracking system. It doesn't
know if your source files changed since it was written.

Usage:
    # One-time: ingest and cache
    python -m ingestion.cache data/raw data/ingested_cache.json

    # Anywhere else: load instantly, no re-ingestion
    from ingestion.cache import load_raw_documents
    docs = load_raw_documents("data/ingested_cache.json")
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

from ingestion.schema import RawDocument
from config import DATA_DIR

DEFAULT_CACHE_PATH = DATA_DIR / "ingested_docs_cache.json"


def save_raw_documents(docs: list[RawDocument], path: str = None) -> None:
    path = Path(path or DEFAULT_CACHE_PATH)
    data = [asdict(d) for d in docs]
    path.write_text(json.dumps(data, indent=2))
    print(f"Cached {len(docs)} ingested document(s) to {path}")


def load_raw_documents(path: str = None) -> list[RawDocument]:
    path = Path(path or DEFAULT_CACHE_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"No ingestion cache found at {path}. Build one first with:\n"
            f"    python -m ingestion.cache <your_docs_folder> {path}"
        )
    data = json.loads(path.read_text())
    return [RawDocument(**d) for d in data]


def cache_exists(path: str = None) -> bool:
    return Path(path or DEFAULT_CACHE_PATH).exists()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m ingestion.cache <source_folder> [cache_output_path]")
        sys.exit(1)

    from ingestion.loader import ingest_directory

    source = sys.argv[1]
    cache_path = sys.argv[2] if len(sys.argv) > 2 else str(DEFAULT_CACHE_PATH)

    print(f"Ingesting {source} (this is the slow part — OCR, PDF parsing, etc.)...")
    docs = ingest_directory(source)
    save_raw_documents(docs, cache_path)

    by_modality: dict[str, int] = {}
    for d in docs:
        by_modality[d.modality] = by_modality.get(d.modality, 0) + 1
    print(f"Done: {by_modality}")
    print(f"\nDownstream scripts can now load this instantly instead of re-ingesting:")
    print(f"    python -m chunking.benchmark_chunkers --cache {cache_path}")
    print(f"    python -m chunking.benchmark_chunking_retrieval --cache {cache_path}")
