"""
Ingest enhancement - OCR fallback for scanned PDFs.

NOTE: this capability is now built directly into ingestion/ingest_pdf.py
via its `ocr_on_empty_pages=True` default — that's the version pipeline.py
and ingestion/loader.py actually use. This standalone file is kept as a
simpler reference implementation (OCR only, no image extraction alongside
it) and still works fine on its own, but prefer ingest_pdf.ingest_pdf()
for anything going through the main pipeline so you're not maintaining two
separate OCR code paths that could drift apart.

ingest_pdf.py's page.get_text() returns an empty string for pages that are
just a scanned image (no embedded text layer). This module detects that
case and falls back to rendering the page to an image + running local OCR
(pytesseract, free, requires the system `tesseract-ocr` binary).

Requires:
    apt-get install tesseract-ocr      # or your OS equivalent
    pip install pytesseract pdf2image
    apt-get install poppler-utils      # pdf2image needs this to render pages

Run directly to smoke-test:
    python -m ingestion.ocr_fallback data/raw/scanned.pdf
"""

import sys
from pathlib import Path

from ingestion.schema import RawDocument

try:
    import fitz  # PyMuPDF
    import pytesseract
    from PIL import Image
except ImportError:
    fitz = None
    pytesseract = None


def _page_needs_ocr(page, min_chars: int = 20) -> bool:
    """Heuristic: if a page has almost no extractable text but does have
    image content, it's very likely a scan rather than a text page."""
    text = page.get_text().strip()
    if len(text) >= min_chars:
        return False
    return len(page.get_images(full=True)) > 0 or len(text) == 0


def ocr_page(page, dpi: int = 300) -> str:
    """Render a PDF page to a raster image at `dpi` and run local OCR on it.
    Higher DPI improves OCR accuracy on small print at the cost of speed."""
    pix = page.get_pixmap(dpi=dpi)
    img_bytes = pix.tobytes("png")
    import io
    image = Image.open(io.BytesIO(img_bytes))
    return pytesseract.image_to_string(image)


def ingest_pdf_with_ocr_fallback(path: str, ocr_dpi: int = 300) -> list[RawDocument]:
    """
    Same contract as ingest_pdf.ingest_pdf, but any page with no usable text
    layer is OCR'd instead of silently returning empty content.
    """
    if fitz is None or pytesseract is None:
        raise ImportError("Run: pip install pymupdf pytesseract pillow (and install system tesseract-ocr)")

    p = Path(path)
    docs: list[RawDocument] = []

    with fitz.open(str(p)) as pdf:
        for page_num, page in enumerate(pdf):
            native_text = page.get_text().strip()

            if _page_needs_ocr(page):
                ocr_text = ocr_page(page, dpi=ocr_dpi).strip()
                if ocr_text:
                    docs.append(
                        RawDocument.new(
                            source_path=str(p),
                            modality="pdf_text",
                            content=ocr_text,
                            filename=p.name,
                            page=page_num + 1,
                            extraction_method="ocr",
                        )
                    )
            elif native_text:
                docs.append(
                    RawDocument.new(
                        source_path=str(p),
                        modality="pdf_text",
                        content=native_text,
                        filename=p.name,
                        page=page_num + 1,
                        extraction_method="native",
                    )
                )
    return docs


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample.pdf"
    results = ingest_pdf_with_ocr_fallback(target)
    ocr_count = sum(1 for d in results if d.metadata.get("extraction_method") == "ocr")
    native_count = len(results) - ocr_count
    print(f"{native_count} page(s) via native text, {ocr_count} page(s) via OCR fallback")
    for d in results[:3]:
        print(f"  - page {d.metadata['page']} ({d.metadata['extraction_method']}): {d.content[:80]!r}")
