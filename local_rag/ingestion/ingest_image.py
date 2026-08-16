"""
Ingest step - standalone image files (png, jpg, etc.)

These are passed through untouched at ingest time; the actual multimodal
representation happens in embeddings/clip_embedder.py, which embeds the raw
image into the same vector space as text (true CLIP-style multimodal RAG,
not OCR/captioning).

Run directly to smoke-test:
    python -m ingestion.ingest_image data/raw/sample.png
"""

import sys
from pathlib import Path

from ingestion.schema import RawDocument

SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def ingest_image_file(path: str) -> RawDocument:
    p = Path(path)
    return RawDocument.new(
        source_path=str(p),
        modality="image",
        content="",
        image_path=str(p),
        filename=p.name,
    )


def ingest_image_dir(dir_path: str) -> list[RawDocument]:
    p = Path(dir_path)
    docs = []
    for ext in SUPPORTED_EXTS:
        for file in p.rglob(f"*{ext}"):
            docs.append(ingest_image_file(str(file)))
    return docs


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    if Path(target).is_dir():
        results = ingest_image_dir(target)
        print(f"Ingested {len(results)} image(s) from {target}")
    else:
        doc = ingest_image_file(target)
        print(f"Ingested image: {doc.image_path}")
