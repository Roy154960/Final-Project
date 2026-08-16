"""
Chunk method 5: Document-structure-aware chunking.
Exploits markdown headings (#, ##, ###) or, for PDFs, page boundaries
already present in metadata (see ingestion/ingest_pdf.py, which emits one
RawDocument per page). Best when your documents already have clear structure.
"""

import re
from ingestion.schema import RawDocument, Chunk

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def chunk_markdown_by_heading(doc: RawDocument) -> list[Chunk]:
    text = doc.content
    matches = list(HEADING_RE.finditer(text))

    # Carry doc-level metadata (filename, ...) forward so source attribution
    # works at generation time; "method"/"section" below override as needed.
    if not matches:
        # No headings found — treat whole doc as one chunk.
        return [Chunk.new(doc_id=doc.doc_id, text=text,
                           **{**doc.metadata, "method": "structure_aware", "section": "whole_doc"})]

    chunks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        heading_title = m.group(2)
        if section_text:
            chunks.append(
                Chunk.new(doc_id=doc.doc_id, text=section_text,
                          **{**doc.metadata, "method": "structure_aware", "section": heading_title})
            )
    return chunks


def chunk_pdf_page_as_unit(doc: RawDocument) -> list[Chunk]:
    """For PDFs ingested page-by-page (modality='pdf_text'), one chunk per page."""
    # doc.metadata already carries filename/page/extraction_method from
    # ingestion/ingest_pdf.py — spread it forward so source attribution
    # (filename + page) survives into generation/prompts.py.
    return [
        Chunk.new(
            doc_id=doc.doc_id,
            text=doc.content,
            **{**doc.metadata, "method": "structure_aware"},
        )
    ]


def chunk_pdf_table_as_unit(doc: RawDocument) -> list[Chunk]:
    """For PDF tables (modality='pdf_table', see ingestion/table_extraction.py),
    one chunk per table — a table is never split across chunks, since a row
    split mid-table breaks the markdown structure and loses the
    row/column alignment that both the embedder and the LLM rely on to
    read it correctly. This chunk is additive alongside that page's own
    plain-text chunk (which still contains the table flattened into prose
    via page.get_text()) rather than a replacement for it."""
    return [
        Chunk.new(
            doc_id=doc.doc_id,
            text=doc.content,
            **{**doc.metadata, "method": "structure_aware", "source_type": "table"},
        )
    ]


if __name__ == "__main__":
    md = "# Intro\nSome intro text.\n\n## Details\nMore detail here.\n\n## Conclusion\nWrap up."
    sample = RawDocument.new(source_path="sample.md", modality="text", content=md)
    result = chunk_markdown_by_heading(sample)
    for c in result:
        print(f"[{c.metadata['section']}] {c.text[:40]}...")
