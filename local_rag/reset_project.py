"""
reset_project.py - wipe all generated/processed data so the pipeline can be
re-run from a clean slate: the vector store(s), stages.py's checkpoint
files, the --incremental manifest, the --parent-child store, and (safely)
the duplicate copies POST /ingest stages under data/raw/ on every upload.

Deliberately self-contained, importing nothing from the rest of the
project — just stdlib. That's so it still runs even if something else in
the project is mid-edit or a dependency (torch, chromadb, ...) isn't
installed; all it needs to do is delete files, which needs none of that.
Paths below mirror config.py's own definitions (PROJECT_ROOT / "data" / ...)
rather than importing them, for the same reason.

What gets cleared BY DEFAULT (all regeneratable — re-run ingest/chunk/embed/
store to rebuild):
  - data/chroma_db/                  (both the text AND image collections —
                                       they live in the same persist dir)
  - Qdrant collections "rag_chunks" / "rag_images", best-effort — only if
    qdrant-client is installed AND a Qdrant server is actually reachable;
    silently skipped otherwise, since Chroma is this project's default
  - data/checkpoints/                (stages.py's 01-05 checkpoint files)
  - data/ingest_manifest.json        (the --incremental content-hash manifest)
  - data/parents_store.json          (the --parent-child parent-chunk store)
  - data/raw/<32-hex-chars>_*        ONLY files matching the exact pattern
                                       api.py's /ingest stages uploads under
                                       (uuid.uuid4().hex + "_" + filename) —
                                       these are internal duplicate copies
                                       that pile up on every API upload and
                                       are never cleaned up on success. Your
                                       own files in data/raw/ do NOT match
                                       this pattern and are left alone.

NOT cleared unless you ask for it (see flags below):
  - data/raw/*                       your actual source documents
  - data/benchmark_results/          benchmark/comparison output you may
                                       still want for a report

Usage:
    python reset_project.py                  # prompts for confirmation
    python reset_project.py -y                # no prompt
    python reset_project.py --dry-run          # show what would be removed, do nothing
    python reset_project.py --include-raw-docs # ALSO wipes data/raw/ entirely,
                                                 # including your own source files
    python reset_project.py --include-benchmarks  # also wipes data/benchmark_results/
    python reset_project.py -y --include-raw-docs --include-benchmarks  # everything
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DOCS_DIR = DATA_DIR / "raw"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"
BENCHMARK_RESULTS_DIR = DATA_DIR / "benchmark_results"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
MANIFEST_PATH = DATA_DIR / "ingest_manifest.json"
PARENTS_STORE_PATH = DATA_DIR / "parents_store.json"

QDRANT_COLLECTIONS = ["rag_chunks", "rag_images"]

# Exactly api.py's f"{uuid.uuid4().hex}_{file.filename}" pattern — 32 lowercase
# hex chars, an underscore, then the original filename.
_STAGED_UPLOAD_RE = re.compile(r"^[0-9a-f]{32}_.+")


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def _plan(include_raw_docs: bool, include_benchmarks: bool):
    """Returns a list of (label, path_or_paths, kind) describing what would
    be removed. kind is 'dir', 'file', or 'files' (a list of individual files)."""
    plan = []

    if CHROMA_PERSIST_DIR.exists():
        plan.append(("Chroma vector store (text + image collections)", CHROMA_PERSIST_DIR, "dir"))

    if CHECKPOINT_DIR.exists():
        plan.append(("stages.py checkpoint files", CHECKPOINT_DIR, "dir"))

    if MANIFEST_PATH.exists():
        plan.append(("--incremental manifest", MANIFEST_PATH, "file"))

    if PARENTS_STORE_PATH.exists():
        plan.append(("--parent-child parent store", PARENTS_STORE_PATH, "file"))

    if RAW_DOCS_DIR.exists():
        staged = [f for f in RAW_DOCS_DIR.iterdir() if f.is_file() and _STAGED_UPLOAD_RE.match(f.name)]
        if staged:
            plan.append((f"duplicate API-staged uploads in data/raw/ ({len(staged)} file(s))", staged, "files"))

    if include_raw_docs and RAW_DOCS_DIR.exists():
        others = [f for f in RAW_DOCS_DIR.iterdir()
                  if not (f.is_file() and _STAGED_UPLOAD_RE.match(f.name))]
        if others:
            plan.append(("ALL of data/raw/ including your own source documents", RAW_DOCS_DIR, "dir"))

    if include_benchmarks and BENCHMARK_RESULTS_DIR.exists():
        plan.append(("benchmark/comparison results", BENCHMARK_RESULTS_DIR, "dir"))

    return plan


def _remove(label: str, target, kind: str) -> int:
    freed = 0
    if kind == "dir":
        freed = _dir_size(target)
        shutil.rmtree(target, ignore_errors=True)
    elif kind == "file":
        freed = _dir_size(target)
        target.unlink(missing_ok=True)
    elif kind == "files":
        for f in target:
            freed += _dir_size(f)
            f.unlink(missing_ok=True)
    print(f"[reset]   removed: {label} ({_human(freed)})")
    return freed


def _reset_qdrant():
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        print("[reset] Qdrant: qdrant-client not installed — skipped (fine if you only use Chroma)")
        return
    try:
        client = QdrantClient(host="localhost", port=6333, timeout=2)
        existing = {c.name for c in client.get_collections().collections}
        for name in QDRANT_COLLECTIONS:
            if name in existing:
                client.delete_collection(name)
                print(f"[reset]   removed: Qdrant collection {name!r}")
        if not existing & set(QDRANT_COLLECTIONS):
            print("[reset] Qdrant: reachable, but no rag_chunks/rag_images collections found — nothing to do")
    except Exception as e:
        print(f"[reset] Qdrant: not reachable ({e}) — skipped (fine if you only use Chroma)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-y", "--yes", action="store_true", help="Don't ask for confirmation")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed, remove nothing")
    parser.add_argument("--include-raw-docs", action="store_true",
                         help="ALSO wipe data/raw/ entirely, including your own source documents")
    parser.add_argument("--include-benchmarks", action="store_true",
                         help="Also wipe data/benchmark_results/")
    args = parser.parse_args()

    plan = _plan(args.include_raw_docs, args.include_benchmarks)

    print(f"[reset] project root: {PROJECT_ROOT}")
    if not plan and not args.include_raw_docs:
        # Qdrant is checked separately below regardless, since it's an
        # external service and its presence can't be seen from the filesystem.
        pass

    if plan:
        print("[reset] the following will be removed:")
        for label, _target, _kind in plan:
            print(f"  - {label}")
    else:
        print("[reset] no local (filesystem) data found to remove.")
    print("[reset] will also attempt to clear Qdrant collections rag_chunks/rag_images "
          "if a Qdrant server is reachable on localhost:6333.")

    if args.dry_run:
        print("\n[reset] --dry-run: nothing was actually removed.")
        return

    if not args.yes:
        answer = input("\nProceed? [y/N] ").strip().lower()
        if answer != "y":
            print("[reset] aborted, nothing removed.")
            sys.exit(0)

    print()
    total_freed = 0
    for label, target, kind in plan:
        total_freed += _remove(label, target, kind)
    _reset_qdrant()

    # Recreate the directories the rest of the project expects to already
    # exist (config.py does this too on import, but nothing else has run yet
    # right after a reset).
    for d in (DATA_DIR, RAW_DOCS_DIR, CHROMA_PERSIST_DIR, BENCHMARK_RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n[reset] done — freed {_human(total_freed)} on disk. "
          f"Ready for a clean `python stages.py ingest --source data/raw` "
          f"(or `python pipeline.py ingest ...`).")


if __name__ == "__main__":
    main()
