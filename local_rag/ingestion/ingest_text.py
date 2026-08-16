"""
Ingest step - plain text / markdown files.

Run directly to smoke-test:
    python -m ingestion.ingest_text data/raw/sample.txt
"""

import sys
from pathlib import Path

from ingestion.schema import RawDocument


def ingest_text_file(path: str) -> RawDocument:
    p = Path(path)
    content = p.read_text(encoding="utf-8", errors="ignore")
    return RawDocument.new(
        source_path=str(p),
        modality="text",
        content=content,
        filename=p.name,
    )


def ingest_text_dir(dir_path: str, extensions=(".txt", ".md")) -> list[RawDocument]:
    p = Path(dir_path)
    docs = []
    for ext in extensions:
        for file in p.rglob(f"*{ext}"):
            docs.append(ingest_text_file(str(file)))
    return docs


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    if Path(target).is_dir():
        results = ingest_text_dir(target)
        print(f"Ingested {len(results)} text file(s) from {target}")
        for r in results:
            print(f"  - {r.source_path}: {len(r.content)} chars")
    else:
        doc = ingest_text_file(target)
        print(f"Ingested {doc.source_path}: {len(doc.content)} chars")
        print(doc.content[:300])
