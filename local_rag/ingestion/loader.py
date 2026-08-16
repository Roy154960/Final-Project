"""
Unified ingestion entry point: given a directory of mixed files
(txt, md, pdf, png/jpg...), dispatch each to the right loader and
return one flat list of RawDocument.

PDFs get two extraction passes merged into one flat list: ingest_pdf()
(PyMuPDF — page text, OCR fallback, embedded images, page-complexity
flagging) and extract_tables() (pdfplumber — structured markdown tables,
kept as their own modality='pdf_table' docs alongside the page's
flattened text, not instead of it — see
chunking/structure_aware.py:chunk_pdf_table_as_unit). Table extraction is
on by default: it's local, fast (no model call), and its own module
docstring is explicit that skipping it silently loses table structure.
"""

from pathlib import Path
from typing import Literal

from ingestion.schema import RawDocument
from ingestion.ingest_text import ingest_text_file
from ingestion.ingest_pdf import ingest_pdf
from ingestion.ingest_image import ingest_image_file, SUPPORTED_EXTS as IMAGE_EXTS

TEXT_EXTS = (".txt", ".md")

PageVlmMode = Literal["auto", "always", "never"]


def _extract_pdf_tables_safely(path: str) -> list[RawDocument]:
    """pdfplumber is an optional dependency (see requirements.txt) — a
    missing install shouldn't take down the rest of ingestion, it should
    just mean no separate table chunks this run (the page's flattened
    text still has the table content, just unstructured)."""
    try:
        from ingestion.table_extraction import extract_tables
        return extract_tables(path)
    except ImportError as e:
        print(f"[warn] table extraction skipped for {path} (pdfplumber not installed: {e})")
        return []
    except Exception as e:
        print(f"[warn] table extraction failed for {path}: {e}")
        return []


def ingest_path(path: str, describe_pages: PageVlmMode = "auto", force_ocr: bool = False) -> list[RawDocument]:
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        docs = ingest_pdf(str(p), describe_pages=describe_pages, force_ocr=force_ocr)
        docs.extend(_extract_pdf_tables_safely(str(p)))
        return docs
    if p.suffix.lower() in TEXT_EXTS:
        return [ingest_text_file(str(p))]
    if p.suffix.lower() in IMAGE_EXTS:
        return [ingest_image_file(str(p))]
    raise ValueError(f"Unsupported file type: {p.suffix}")


def ingest_directory(dir_path: str, describe_pages: PageVlmMode = "auto", force_ocr: bool = False) -> list[RawDocument]:
    """Despite the name, also accepts a path to a single file (delegates to
    ingest_path) — so `--source path/to/one.pdf` works the same way
    `--source data/raw/` does, everywhere this function is used.

    force_ocr: passed straight through to ingest_pdf() for every PDF this
    call touches (see its own docstring for why you'd want this) -- there
    is no per-file override here, so point --source at just the one
    troublesome PDF (or a folder containing only it) rather than your
    whole corpus if only one source has this problem; forcing OCR on
    every page of every PDF in a large, otherwise-healthy corpus is much
    slower for no benefit on the pages that were already extracting
    correctly.
    """
    p = Path(dir_path)
    if p.is_file():
        return ingest_path(str(p), describe_pages=describe_pages, force_ocr=force_ocr)
    docs: list[RawDocument] = []
    all_exts = TEXT_EXTS + IMAGE_EXTS + (".pdf",)
    for file in p.rglob("*"):
        if file.is_file() and file.suffix.lower() in all_exts:
            try:
                docs.extend(ingest_path(str(file), describe_pages=describe_pages, force_ocr=force_ocr))
            except Exception as e:
                print(f"[warn] failed to ingest {file}: {e}")
    return docs


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    docs = ingest_directory(target)
    by_modality: dict[str, int] = {}
    for d in docs:
        by_modality[d.modality] = by_modality.get(d.modality, 0) + 1
    print(f"Ingested {len(docs)} raw document(s) from {target}: {by_modality}")
