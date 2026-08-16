"""
Ingest enhancement - table extraction from PDFs.

Flattening a table into paragraph text (what ingest_pdf.py does by default
via page.get_text()) destroys row/column structure — "Q3 | 12% | up" reads
fine to a human but is much harder for an embedding model or an LLM to
reason over correctly. This module extracts tables as structured data using
pdfplumber (free, local) and renders them as markdown tables instead, which
both embedding models and LLMs handle noticeably better than flattened rows.

Requires:
    pip install pdfplumber

Run directly to smoke-test:
    python -m ingestion.table_extraction data/raw/report_with_tables.pdf
"""

import sys
from pathlib import Path
from typing import Optional

from ingestion.schema import RawDocument

try:
    import pdfplumber
except ImportError:
    pdfplumber = None


def _table_to_markdown(table: list[list[Optional[str]]]) -> str:
    if not table:
        return ""
    header, *rows = table
    header_cells = [str(c) if c is not None else "" for c in header]
    md = ["| " + " | ".join(header_cells) + " |"]
    md.append("| " + " | ".join(["---"] * len(header_cells)) + " |")
    for row in rows:
        cells = [str(c) if c is not None else "" for c in row]
        md.append("| " + " | ".join(cells) + " |")
    return "\n".join(md)


def extract_tables(path: str) -> list[RawDocument]:
    """
    Returns one RawDocument per detected table, with modality='pdf_table' and
    content as a markdown table. Feed these through chunking as their own
    unit (a table usually shouldn't be split mid-row) rather than mixing
    them into the recursive text chunker.
    """
    if pdfplumber is None:
        raise ImportError("Run: pip install pdfplumber")

    p = Path(path)
    docs: list[RawDocument] = []

    with pdfplumber.open(str(p)) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for table_idx, table in enumerate(tables):
                if not table or len(table) < 2:  # need at least header + one row
                    continue
                md_table = _table_to_markdown(table)
                docs.append(
                    RawDocument.new(
                        source_path=str(p),
                        modality="pdf_table",
                        content=md_table,
                        filename=p.name,
                        page=page_num + 1,
                        table_index=table_idx,
                        n_rows=len(table) - 1,
                        n_cols=len(table[0]) if table else 0,
                    )
                )
    return docs


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample.pdf"
    results = extract_tables(target)
    print(f"Extracted {len(results)} table(s) from {target}")
    for d in results[:2]:
        print(f"\n--- page {d.metadata['page']}, table {d.metadata['table_index']} "
              f"({d.metadata['n_rows']}x{d.metadata['n_cols']}) ---")
        print(d.content)
