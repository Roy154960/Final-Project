"""
Ingest step - standalone image files (png, jpg, etc.)

These are passed through untouched at ingest time; the actual multimodal
representation happens in embeddings/clip_embedder.py, which embeds the raw
image into the same vector space as text (true CLIP-style multimodal RAG,
not OCR/captioning).

Run directly to smoke-test:
    python -m ingestion.ingest_image data/raw/sample.png
"""

import hashlib
import shutil
import sys
from pathlib import Path
from typing import Optional

from ingestion.schema import RawDocument

SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def ingest_image_file(path: str, copy_to_dir: Optional[str] = None) -> RawDocument:
    """
    copy_to_dir: when given, copies this image into that directory and
    records the COPY's path as image_path instead of the original's --
    see loader.py's own ingest_path() docstring for why (a standalone
    image ingested from outside the Docker bind mount is otherwise
    unreadable by the containers at query time; unlike a PDF's embedded
    images, there's no extraction step here to simply redirect, so this
    copies the bytes directly instead).

    Filename is prefixed with a short content hash: collision-safe
    against two different source images that happen to share a
    filename, AND idempotent -- re-ingesting the byte-identical file
    reuses the same destination rather than piling up duplicate copies
    on every re-run.
    """
    p = Path(path)
    image_path = p
    if copy_to_dir:
        dest_dir = Path(copy_to_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(p.read_bytes()).hexdigest()[:8]
        dest = dest_dir / f"{digest}_{p.name}"
        if not dest.exists():
            shutil.copyfile(p, dest)
        image_path = dest
    return RawDocument.new(
        source_path=str(p),
        modality="image",
        content="",
        image_path=str(image_path),
        filename=p.name,
    )


def ingest_image_dir(dir_path: str, copy_to_dir: Optional[str] = None) -> list[RawDocument]:
    p = Path(dir_path)
    docs = []
    for ext in SUPPORTED_EXTS:
        for file in p.rglob(f"*{ext}"):
            docs.append(ingest_image_file(str(file), copy_to_dir=copy_to_dir))
    return docs


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    if Path(target).is_dir():
        results = ingest_image_dir(target)
        print(f"Ingested {len(results)} image(s) from {target}")
    else:
        doc = ingest_image_file(target)
        print(f"Ingested image: {doc.image_path}")
