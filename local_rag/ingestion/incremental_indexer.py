"""
Ingest enhancement - incremental re-indexing.

Re-embedding an entire corpus every time one file changes wastes time and
compute. This module tracks a content hash per source file in a small local
JSON manifest, and tells the pipeline which files are new, changed, or
deleted since the last run — so pipeline.py can embed only what's needed
and remove stale vectors for deleted/changed files.

Run directly to smoke-test:
    python -m ingestion.incremental_indexer data/raw
"""

import hashlib
import json
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

from config import DATA_DIR

MANIFEST_PATH = DATA_DIR / "ingest_manifest.json"


@dataclass
class ManifestEntry:
    path: str
    content_hash: str
    chunk_ids: list[str]  # so we know which vectors to delete if this file changes/is removed


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_manifest() -> dict[str, ManifestEntry]:
    if not MANIFEST_PATH.exists():
        return {}
    raw = json.loads(MANIFEST_PATH.read_text())
    return {k: ManifestEntry(**v) for k, v in raw.items()}


def save_manifest(manifest: dict[str, ManifestEntry]) -> None:
    raw = {k: asdict(v) for k, v in manifest.items()}
    MANIFEST_PATH.write_text(json.dumps(raw, indent=2))


def diff_against_manifest(source_dir: str, extensions=None):
    """
    Returns (new_files, changed_files, deleted_paths, unchanged_files).
    - new_files / changed_files: list[Path] that need (re-)ingesting
    - deleted_paths: list[str] whose old chunk_ids need removing from the vector store
    - unchanged_files: list[Path] that can be skipped entirely

    `source_dir` also accepts a path to a SINGLE FILE, not just a
    directory -- mirrors ingestion/loader.py's ingest_directory() own
    "despite the name, also accepts a single file" behavior, which
    pipeline.py's cmd_ingest relies on for exactly this (e.g. pointing
    --source at one specific troublesome PDF to re-ingest with
    --force-ocr). CONFIRMED bug this fixes: Path(a_file).rglob("*")
    silently returns an EMPTY list rather than the file itself or an
    error -- current_files would come back empty, which would make
    EVERY existing manifest entry for that file look "deleted" (nothing
    in current_files to match it against) with nothing in
    new_files/changed_files/unchanged_files to replace it -- i.e.
    pointing --source at a single already-ingested file would silently
    DELETE its vectors and never re-add them, the exact opposite of
    what re-ingesting it is supposed to do.

    `extensions`, when None (the default), is ingestion/loader.py's own
    TEXT_EXTS + IMAGE_EXTS + (".pdf",) -- imported directly rather than
    duplicated here as a second, separately-maintained list, which is
    exactly how this function's own extension list previously drifted
    out of sync with loader.py's (missing ".webp"/".bmp", both of which
    ingest_image.py has always supported) without either list's author
    necessarily noticing, since nothing enforced they stay identical.
    Pass an explicit tuple to override, same as before.
    """
    if extensions is None:
        from ingestion.loader import TEXT_EXTS, IMAGE_EXTS
        extensions = TEXT_EXTS + IMAGE_EXTS + (".pdf",)

    manifest = load_manifest()
    source_path = Path(source_dir)
    if source_path.is_file():
        current_files = {str(source_path)} if source_path.suffix.lower() in extensions else set()
    else:
        current_files = {
            str(f) for f in source_path.rglob("*") if f.is_file() and f.suffix.lower() in extensions
        }

    new_files, changed_files, unchanged_files = [], [], []
    for f in current_files:
        content_hash = _hash_file(Path(f))
        entry = manifest.get(f)
        if entry is None:
            new_files.append(Path(f))
        elif entry.content_hash != content_hash:
            changed_files.append(Path(f))
        else:
            unchanged_files.append(Path(f))

    deleted_paths = [f for f in manifest if f not in current_files]

    return new_files, changed_files, deleted_paths, unchanged_files


def update_manifest_entry(path: str, chunk_ids: list[str], key: Optional[str] = None) -> None:
    """
    Records that `chunk_ids` are currently stored for the source at
    `path`, so a future call can find and remove them again (see
    remove_manifest_entry). The content hash used to detect future
    changes is always computed from the actual bytes AT `path` -- that
    file must exist on disk when this is called.

    `key`, when given, is the manifest's LOOKUP KEY instead of `path`
    itself -- everything else (the content hash) is still computed from
    `path`. This exists for local_rag/api.py's /ingest endpoint: an
    uploaded file is staged to a UUID-prefixed disk path
    (`{uuid}_{filename}`) that's DIFFERENT on every single upload, so
    keying the manifest by that path (pipeline.py's CLI usage, where
    `path` genuinely is a stable, repeatedly-ingested source location)
    would never let a later re-upload of "the same" document (by name)
    find and clean up the earlier upload's chunk_ids -- every upload
    would look like a brand new, never-before-seen source. Passing
    key=file.filename instead means the manifest tracks "this uploaded
    filename currently maps to these chunk_ids," which is the identity
    an API caller re-uploading a document actually means by "the same
    file," while still hashing the REAL staged bytes at `path` for the
    record.
    """
    manifest = load_manifest()
    manifest_key = key if key is not None else path
    manifest[manifest_key] = ManifestEntry(path=manifest_key, content_hash=_hash_file(Path(path)), chunk_ids=chunk_ids)
    save_manifest(manifest)


def remove_manifest_entry(path: str) -> list[str]:
    """Returns the chunk_ids that were associated with this path, so the
    caller can delete them from the vector store, then removes the entry."""
    manifest = load_manifest()
    entry = manifest.pop(path, None)
    save_manifest(manifest)
    return entry.chunk_ids if entry else []


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    new, changed, deleted, unchanged = diff_against_manifest(target)
    print(f"new: {len(new)}, changed: {len(changed)}, deleted: {len(deleted)}, unchanged: {len(unchanged)}")
    for f in new:
        print(f"  [new] {f}")
    for f in changed:
        print(f"  [changed] {f}")
    for f in deleted:
        print(f"  [deleted] {f}")
