"""
Ingest step - PDF files.

Uses PyMuPDF (fitz) because it is free, local, fast, and — unlike text-only
extractors — lets us pull embedded images out of the same file so they can
be routed to the CLIP embedder later.

Handles TWO different kinds of scanned PDF, which need opposite handling:

1. "Text layer + page-scan image" (e.g. archive.org exports): the page
   already has a real, searchable text layer — get_text() works fine — but
   ALSO embeds one or more full-page images underneath it: a background
   scan, and often also a distorted/rotated preview thumbnail (Internet
   Archive's BookReader uses these for its page-turn animation). Both are
   redundant with the text layer. skip_full_page_scans=True (default)
   detects and drops them by checking how much of the PAGE AREA each image
   is actually placed to cover (via page.get_image_rects()), not by pixel
   dimensions — this is what makes it robust to low-resolution "reading
   copy" scans AND to warped preview thumbnails, both of which a pixel-size
   or aspect-ratio check on the raw image would miss.

2. "Text baked into the pixels, no text layer at all" (a true flat scan):
   get_text() returns nothing because there IS no text layer — the words
   are just part of the image. ocr_on_empty_pages=True (default) detects
   this (get_text() empty) and runs local OCR (pytesseract) on the page
   image to actually recover the text, instead of silently returning
   nothing for that page.

Both are on by default and don't conflict: case 1's image gets filtered
out AFTER its text was already captured natively; case 2 has no native
text to begin with, so OCR fills the gap.

Also flags pages for whole-page VLM description (Strategy 3: Vision LLMs
for Document Understanding — see ingestion/page_description.py). This is
deliberately split into a cheap half here and an expensive half there:

  - HERE (always cheap, no model call): describe_pages="auto" (default)
    runs _page_looks_visually_complex() — pure PyMuPDF geometry, just
    counts vector-drawn paths vs native text length — on every page. Only
    pages that look like they contain a chart/diagram drawn as vector
    graphics (which page.get_images() never catches, since it was never
    embedded as a raster image) get their full page rendered to a PNG and
    flagged via metadata["page_image_path"]. Everything else costs
    nothing extra.
  - THERE (expensive, one VLM call per flagged page): a downstream step —
    pipeline.py's describe_complex_pages(), gated behind --multimodal —
    is what actually calls the VLM, and only for docs this function
    already flagged. See that function's docstring for why the split.

describe_pages="always" flags every page (skip the heuristic, e.g. for a
one-off exhaustive run); describe_pages="never" disables this entirely
(zero rendering, zero flagging, matches old behavior).

Requires (only for OCR path):
    apt-get install tesseract-ocr
    pip install pytesseract

Run directly to smoke-test:
    python -m ingestion.ingest_pdf data/raw/sample.pdf
"""

import io
import re
import sys
from pathlib import Path
from typing import Literal, Optional

from ingestion.schema import RawDocument
from utils.image_sniff import describe_ext_mismatch
from config import (
    TESSERACT_CMD,
    PAGE_VISUAL_COMPLEXITY_MIN_DRAWINGS,
    PAGE_VISUAL_COMPLEXITY_MAX_TEXT_CHARS,
)

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pytesseract
    from PIL import Image
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
except ImportError:
    pytesseract = None

# How close an image's dimensions need to be to the full page to count as a
# "page scan" rather than a figure. Real figures are almost always
# meaningfully smaller than the full page (margins, captions, surrounding
# text), so this rarely false-positives on genuine content.
FULL_PAGE_SCAN_THRESHOLD = 0.9

# Below this many characters of native text, a page is treated as "empty"
# and eligible for OCR — catches pages with a stray page number or header
# but no real body text, not just literally zero characters.
MIN_NATIVE_TEXT_CHARS = 20

DEFAULT_OCR_DPI = 300


# Collapses ANY run of whitespace (including the erratic, layout-driven
# newlines Tesseract inserts between scattered caption fragments on a
# scanned page -- e.g. "بورتريه شخصي\nدورير\nبريشته\n58", one or two
# words per line, none of it a real paragraph break) down to a single
# space. Confirmed live-run problem this fixes: chunk_pdf_page_as_unit()
# (chunking/structure_aware.py) stores doc.content VERBATIM as a chunk's
# embedded text, so without this, those OCR-artifact newlines went
# straight into the embedding and the stored/returned source text
# unchanged -- splitting what's really one short, coherent phrase into
# several disconnected-looking lines both hurts embedding quality (a
# fragmented string embeds less coherently than the same words as one
# phrase) and looks broken when shown back to a person as a source.
# Applied to BOTH native and OCR text (see the page loop below) since a
# native-text layer can have its own PDF-generator-specific line-wrap
# artifacts too, not just OCR.
_WHITESPACE_RUN = re.compile(r"\s+")


def _normalize_extracted_text(text: str) -> str:
    """Collapse all internal whitespace (including newlines) to single
    spaces and strip the ends. See _WHITESPACE_RUN's own comment for the
    confirmed problem this fixes. Deliberately simple -- this does NOT
    attempt to detect or fix right-to-left/bidi ordering issues (see
    ingest_pdf()'s own `force_ocr` parameter docstring for that
    different, harder problem); it only removes spurious line breaks."""
    return _WHITESPACE_RUN.sub(" ", text).strip()


# Fraction of the page area an image must cover, by its ACTUAL PLACEMENT on
# the page (not its raw pixel dimensions), to count as a background/full-page
# image rather than a genuine figure. Using placement instead of pixel size
# is what makes this robust to two failure modes pixel-size checks miss:
#   - lower-resolution "reading copy" scans (can be well under 800px on a
#     side, but still placed to cover the whole page)
#   - Internet Archive's warped/skewed "page-curl" thumbnail images (used
#     for their BookReader's page-turn animation) — pixel aspect ratio looks
#     nothing like the page's rectangular aspect ratio, but the bounding box
#     of where it's actually DRAWN still spans nearly the full page
PAGE_COVERAGE_THRESHOLD = 0.6


def _looks_like_full_page_scan(page, xref: int, threshold: float) -> bool:
    """
    Checks how much of the page area this image's placement(s) actually
    cover, using page.get_image_rects() — the rectangle(s) where the image
    is drawn on the page, in page coordinates. This is placement-based
    rather than pixel-based, so it correctly catches both low-resolution
    background scans AND distorted/rotated preview thumbnails that a
    pixel-dimension check would miss.
    """
    page_area = page.rect.width * page.rect.height
    if page_area == 0:
        return False

    try:
        rects = page.get_image_rects(xref)
    except Exception:
        return False
    if not rects:
        return False

    covered_area = sum(r.width * r.height for r in rects)
    return (covered_area / page_area) >= threshold

def _too_small_or_decorative(page, xref: int, min_coverage: float = 0.01, extreme_aspect_ratio: float = 50.0) -> bool:
    """
    Returns True if an image is too small or too oddly-shaped on the page to
    be a meaningful figure -- catches things like bullet-point icons, thin
    horizontal rule lines, or decorative dividers, which are technically
    embedded images but not content worth keeping.

    min_coverage: an image covering less than this fraction of the page area
    is treated as decorative rather than a real figure.
    extreme_aspect_ratio: an image whose width:height (or height:width) ratio
    exceeds this is treated as a thin line/rule rather than a real figure.
    """
    page_area = page.rect.width * page.rect.height
    if page_area == 0:
        return False

    try:
        rects = page.get_image_rects(xref)
    except Exception:
        return False
    if not rects:
        return False

    covered_area = sum(r.width * r.height for r in rects)
    if (covered_area / page_area) < min_coverage:
        return True

    for r in rects:
        if r.height == 0 or r.width == 0:
            continue
        aspect_ratio = r.width / r.height
        if aspect_ratio > extreme_aspect_ratio or aspect_ratio < (1 / extreme_aspect_ratio):
            return True

    return False


# Small bound on how much "surrounding text" gets attached to each
# extracted image's metadata (see _nearby_page_text below) -- enough for
# a caption-length paragraph to genuinely help text search find the
# image via vocabulary the VLM's own short caption alone might lack
# ("a bowl of fruit" vs. the surrounding prose's "still life,"
# "chiaroscuro," "underpainting"), not enough to turn every image's
# metadata into a second copy of most of the page.
IMAGE_NEARBY_TEXT_CHARS = 300


def _nearby_page_text(page, xref: int, page_text: str, max_chars: int = IMAGE_NEARBY_TEXT_CHARS) -> str:
    """
    A small snippet of whichever text BLOCK on this page sits closest
    (by vertical position) to the image at `xref`'s own placement --
    genuine spatial proximity, not just "the start of the page's text,"
    since a page with more than one figure would otherwise attach the
    exact same, first-on-the-page text to every image on it regardless
    of which one it's actually next to.

    Uses page.get_image_rects(xref) (the SAME call
    _looks_like_full_page_scan/_too_small_or_decorative above already
    use for this image's own placement) to find the image's own
    location, and page.get_text("blocks") (each block's own bounding
    box, in the same page-coordinate space) to find the text block
    whose vertical center sits closest to the image's -- a cheap,
    reasonable proxy for "the caption or paragraph right above/below
    this figure" without needing full reading-order/column-layout
    analysis.

    Falls back to a plain prefix of `page_text` (this page's own
    already-extracted native/OCR text, passed in rather than
    re-extracted here -- see ingest_pdf's own `text` variable) whenever
    geometry lookup finds nothing usable -- a missing/failed proximity
    match should degrade to "some text is better than none," never to
    an empty string, the same "one image's own quirk shouldn't lose an
    otherwise-good result" tradeoff every other per-image step in this
    file already makes (see e.g. the try/except around
    pdf.extract_image() just below). Never raises.
    """
    try:
        rects = page.get_image_rects(xref)
        if not rects:
            return page_text[:max_chars]
        image_center_y = (rects[0].y0 + rects[0].y1) / 2

        blocks = page.get_text("blocks")
        best_text: Optional[str] = None
        best_distance: Optional[float] = None
        for block in blocks:
            block_text = block[4] if len(block) > 4 else ""
            if not block_text or not block_text.strip():
                continue
            block_center_y = (block[1] + block[3]) / 2
            distance = abs(block_center_y - image_center_y)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_text = block_text

        if best_text:
            return " ".join(best_text.split())[:max_chars]
    except Exception as e:  # noqa: BLE001 -- geometry lookup failing must never break image extraction itself, see docstring above
        print(f"[ingest_pdf] nearby-text lookup failed for an image (falling back to "
              f"page-text prefix): {e}", file=sys.stderr)

    return page_text[:max_chars]


def _page_looks_visually_complex(
    page,
    native_text: str,
    n_extracted_images_on_page: int,
    min_drawings: int = PAGE_VISUAL_COMPLEXITY_MIN_DRAWINGS,
    max_text_chars: int = PAGE_VISUAL_COMPLEXITY_MAX_TEXT_CHARS,
) -> bool:
    """
    Cheap, local, no-model-call heuristic behind describe_pages="auto" (see
    module docstring). Returns True only when a page looks like it likely
    contains a chart/diagram drawn as vector graphics rather than a raster
    image — the one case native text extraction AND the existing
    per-image CLIP/caption pipeline both miss entirely.

    n_extracted_images_on_page: if this page already yielded a real kept
    raster image (see the extract_images loop below), that image already
    gets its own VLM caption via ingestion/image_captioning.py — a whole
    -page description would be redundant, so this always returns False in
    that case rather than paying for two VLM calls describing the same
    content.
    """
    if n_extracted_images_on_page > 0:
        return False
    try:
        n_drawings = len(page.get_drawings())
    except Exception:
        # Some malformed/unusual pages can raise here; treat as "not
        # complex" rather than letting one bad page abort ingestion.
        return False
    return n_drawings >= min_drawings and len(native_text) <= max_text_chars


def _ocr_page(page, dpi: int = DEFAULT_OCR_DPI) -> str:
    if pytesseract is None:
        raise ImportError("Run: pip install pytesseract pillow (and install system tesseract-ocr)")
    pix = page.get_pixmap(dpi=dpi)
    image = Image.open(io.BytesIO(pix.tobytes("png")))
    try:
        return pytesseract.image_to_string(image)
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError(
            "Tesseract binary not found. This means it's either not installed, or installed but "
            "not on your system PATH (very common on Windows even right after installing). Fix: "
            "set TESSERACT_CMD in config.py to your tesseract.exe's full path, e.g.\n"
            r'    TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"' + "\n"
            "Install it from https://github.com/UB-Mannheim/tesseract/wiki if you haven't yet."
        )


def ingest_pdf(
    path: str,
    extract_images: bool = True,
    image_out_dir: Optional[str] = None,
    skip_full_page_scans: bool = True,
    skip_small_pictures: bool = True,
    full_page_scan_threshold: float = PAGE_COVERAGE_THRESHOLD,
    ocr_on_empty_pages: bool = True,
    ocr_dpi: int = DEFAULT_OCR_DPI,
    describe_pages: Literal["auto", "always", "never"] = "auto",
    force_ocr: bool = False,
) -> list[RawDocument]:
    """
    Returns a list of RawDocuments:
      - one per page (modality='pdf_text'), from the native text layer where
        one exists, or from OCR where it doesn't (ocr_on_empty_pages=True).
        Pages flagged by describe_pages carry metadata["page_image_path"]
        (a full-page PNG render) for pipeline.py's describe_complex_pages()
        to pick up downstream — see module docstring above.
      - one per embedded image (modality='image') if extract_images=True,
        excluding full-page background scans when skip_full_page_scans=True

    describe_pages: "auto" (default) flags a page only when
        _page_looks_visually_complex() says so (cheap, no model call —
        see that function and PAGE_VISUAL_COMPLEXITY_* in config.py);
        "always" flags every page; "never" flags none (skips the
        heuristic and the page-render step entirely).

    force_ocr: CONFIRMED live problem this exists for -- some PDFs
        (observed with a Noor-Book.com-sourced Arabic PDF, but this is a
        known category of issue with certain Arabic PDF generation/scan
        pipelines generally, not specific to one source) have a REAL
        native text layer that get_text() happily returns, but that
        layer stores Arabic glyphs in visual (mirrored) order rather
        than correct logical Unicode reading order -- reversing the
        WHOLE extracted string character-by-character turns it back
        into coherent Arabic (confirmed by hand against a real example),
        which means the PDF's own content stream, not this extraction
        code, has the character order backwards. A blind, automatic
        "reverse every string" fix is NOT applied here on purpose: mixed
        content (embedded Latin words, digits, page numbers) does NOT
        reverse the same way Arabic prose does, so a blanket reversal
        would silently break those instead of fixing anything -- correct
        handling needs a real bidi-aware pass (e.g. python-bidi +
        arabic_reshaper) applied selectively, which this function does
        NOT attempt.

        The practical, low-risk workaround: OCR reads the RENDERED PAGE
        IMAGE, not the PDF's own (possibly broken) internal character
        order, so Tesseract's Arabic model reconstructs correct logical
        reading order regardless of what the native layer's glyph order
        looks like. force_ocr=True skips trusting native_text/
        MIN_NATIVE_TEXT_CHARS entirely and runs OCR on every page,
        specifically so a PDF like this can be re-ingested cleanly
        without needing bidi-correction code at all. Slower (a real OCR
        pass on every page instead of only on native-text-empty ones)
        and pays the same "OCR needs pytesseract on PATH" dependency
        _ocr_page already documents -- use it deliberately on a specific
        file/source known to have this problem, not as a new default.
    """
    if fitz is None:
        raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")

    p = Path(path)
    image_out_dir = Path(image_out_dir or (p.parent / f"{p.stem}_images"))
    docs: list[RawDocument] = []
    n_skipped_as_page_scan = 0
    n_ocr_pages = 0
    n_pages_flagged_for_vlm = 0
    n_by_ext: dict[str, int] = {}

    with fitz.open(str(p)) as pdf:
        for page_num, page in enumerate(pdf):
            native_text = page.get_text().strip()
            extraction_method = "native"
            text = native_text

            if (force_ocr or len(native_text) < MIN_NATIVE_TEXT_CHARS) and ocr_on_empty_pages:
                # Never let one page's OCR failure take the WHOLE document
                # down. Confirmed, not hypothetical, failure mode on
                # Windows specifically: tesseract installed but not on
                # PATH (very common right after installing it, per
                # _ocr_page's own RuntimeError message below) means the
                # FIRST page with no native text layer raises here --
                # and, uncaught, that aborts this entire `with
                # fitz.open(...)` loop before `return docs` is ever
                # reached, silently discarding every page already
                # extracted, including any real native-text pages earlier
                # in the same document. A personal-RAG upload of a mixed
                # PDF (some pages with a text layer, some scanned) could
                # look, from the outside, exactly like "the PDF's text
                # never got ingested at all" -- ingest_upload() would
                # simply never be called with a partial result to work
                # with, because ingest_pdf() itself never returned one.
                # Degrading to "keep native_text as-is for this one page
                # (however short) and move on" is strictly better than
                # losing the entire document over a missing PATH entry.
                try:
                    ocr_text = _ocr_page(page, dpi=ocr_dpi).strip()
                    # Under force_ocr, always PREFER the OCR result over
                    # native_text even if it's shorter -- the whole point
                    # is that native_text's own character order can't be
                    # trusted for a page like this (see force_ocr's own
                    # docstring above), so "longer wins" would keep
                    # picking the broken native text right back on any
                    # page where OCR happens to extract fewer characters.
                    if force_ocr or len(ocr_text) > len(native_text):
                        text = ocr_text
                        extraction_method = "ocr"
                        n_ocr_pages += 1
                except Exception as e:
                    print(f"[ingest_pdf] {p.name} page {page_num + 1}: OCR failed "
                          f"({e}) -- keeping this page's native text as-is "
                          f"({len(native_text)} char(s)) rather than aborting "
                          f"the whole document", file=sys.stderr)

            # Collapse spurious line breaks (OCR layout artifacts, or a
            # native PDF generator's own line-wrap quirks) now, once,
            # before `text` is used for anything below -- the page's own
            # text chunk (text_doc.content), and _nearby_page_text's
            # plain-prefix fallback both read from this same variable, so
            # normalizing it here means neither needs to repeat the same
            # cleanup separately. See _normalize_extracted_text's own
            # docstring for exactly what this does and doesn't fix.
            text = _normalize_extracted_text(text)

            # Images are extracted before the text RawDocument is finalized
            # so _page_looks_visually_complex() below can see how many real
            # raster images this page already yielded (a page with a kept
            # figure doesn't also need a whole-page VLM description of the
            # same content).
            page_image_docs: list[RawDocument] = []
            if extract_images:
                for img_index, img in enumerate(page.get_images(full=True)):
                    xref = img[0]

                    if skip_full_page_scans and _looks_like_full_page_scan(
                        page, xref, full_page_scan_threshold
                    ):
                        n_skipped_as_page_scan += 1
                        continue

                    if skip_small_pictures and _too_small_or_decorative(
                        page, xref
                    ):
                        n_skipped_as_page_scan += 1
                        continue

                    try:
                        base_image = pdf.extract_image(xref)
                    except Exception:
                        continue

                    image_out_dir.mkdir(parents=True, exist_ok=True)
                    # This "ext" is PyMuPDF's own detection of the image's
                    # filter chain in the PDF (DCTDecode -> jpeg,
                    # JPXDecode -> jpx, etc.) — NOT re-verified against the
                    # bytes actually written below. JPXDecode (JPEG2000)
                    # shows up often in scanned/archival PDFs (e.g. Internet
                    # Archive book exports) because it compresses scans
                    # better than plain JPEG — that's expected, not a bug.
                    # What we DO check for here is the rarer case where
                    # PyMuPDF's ext doesn't match what the saved bytes
                    # actually are (see utils/image_sniff.py's docstring) —
                    # that mismatch is what later makes PIL fail with
                    # "cannot identify image file" in
                    # embeddings/clip_embedder.py's embed_images(), with no
                    # explanation, so it's worth surfacing immediately here
                    # instead of only discovering it downstream.
                    ext = base_image.get("ext", "png")
                    img_path = image_out_dir / f"page{page_num + 1}_img{img_index + 1}.{ext}"
                    img_path.write_bytes(base_image["image"])
                    n_by_ext[ext] = n_by_ext.get(ext, 0) + 1

                    mismatch = describe_ext_mismatch(str(img_path), ext)
                    if mismatch:
                        print(f"[ingest_pdf] {p.name} page {page_num + 1} img {img_index + 1} "
                              f"({img_path.name}): {mismatch} — this image will likely fail to "
                              f"CLIP-embed later (--multimodal), but everything else still proceeds")

                    page_image_docs.append(
                        RawDocument.new(
                            source_path=str(p),
                            modality="image",
                            content="",  # filled in by CLIP/captioning at embed time
                            image_path=str(img_path),
                            filename=p.name,
                            page=page_num + 1,
                            # A small snippet of whatever text sits closest
                            # to this image on the page (see
                            # _nearby_page_text's own docstring) -- used at
                            # --multimodal embed time (pipeline.py's
                            # build_caption_chunks) to combine with this
                            # image's VLM caption before it's embedded and
                            # dual-indexed into the TEXT store, so a search
                            # can find the image via vocabulary the short
                            # caption alone might not use. `text` here is
                            # this SAME page's already-computed native/OCR
                            # text (see the top of this page loop) -- never
                            # re-extracted.
                            nearby_text=_nearby_page_text(page, xref, text),
                        )
                    )

            text_doc: Optional[RawDocument] = None
            if text:
                text_doc = RawDocument.new(
                    source_path=str(p),
                    modality="pdf_text",
                    content=text,
                    filename=p.name,
                    page=page_num + 1,
                    extraction_method=extraction_method,
                )

            if text_doc is not None and describe_pages != "never":
                should_flag = (
                    describe_pages == "always"
                    or _page_looks_visually_complex(page, text, len(page_image_docs))
                )
                if should_flag:
                    try:
                        image_out_dir.mkdir(parents=True, exist_ok=True)
                        page_img_path = image_out_dir / f"page{page_num + 1}_full.png"
                        page.get_pixmap(dpi=ocr_dpi).save(str(page_img_path))
                        text_doc.metadata["page_image_path"] = str(page_img_path)
                        text_doc.metadata["page_vlm_trigger"] = describe_pages
                        n_pages_flagged_for_vlm += 1
                    except Exception as e:
                        print(f"[warn] failed to render page {page_num + 1} of {p.name} "
                              f"for VLM description: {e}")

            if text_doc is not None:
                docs.append(text_doc)
            docs.extend(page_image_docs)

    if n_skipped_as_page_scan:
        print(f"[ingest_pdf] {p.name}: skipped {n_skipped_as_page_scan} full-page scan image(s) "
              f"(pass skip_full_page_scans=False to keep them)")
    if n_ocr_pages:
        print(f"[ingest_pdf] {p.name}: recovered {n_ocr_pages} page(s) via OCR (no native text layer found)")
    if n_pages_flagged_for_vlm:
        print(f"[ingest_pdf] {p.name}: flagged {n_pages_flagged_for_vlm} page(s) as visually complex "
              f"(describe_pages={describe_pages!r}) — VLM description only runs if --multimodal is also passed")
    if n_by_ext:
        ext_summary = ", ".join(f"{count} .{ext}" for ext, count in sorted(n_by_ext.items()))
        print(f"[ingest_pdf] {p.name}: extracted {sum(n_by_ext.values())} image(s) -> {ext_summary}")
        n_jpx = n_by_ext.get("jpx", 0) + n_by_ext.get("jp2", 0)
        if n_jpx:
            print(f"[ingest_pdf] {p.name}: {n_jpx} image(s) are JPEG2000 (.jpx/.jp2) — normal for scanned/"
                  f"archival PDFs (better compression than plain JPEG for scans), but Pillow needs JPEG2000 "
                  f"support to open them; if --multimodal CLIP-embedding warns \"cannot identify image "
                  f"file\" for these, see embeddings/clip_embedder.py's warning for the fix")
    return docs


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample.pdf"
    results = ingest_pdf(target)
    text_docs = [d for d in results if d.modality == "pdf_text"]
    image_docs = [d for d in results if d.modality == "image"]
    print(f"Ingested {len(text_docs)} page(s) of text and {len(image_docs)} embedded image(s) from {target}")
    for d in text_docs[:3]:
        print(f"  - page {d.metadata['page']} ({d.metadata['extraction_method']}): {len(d.content)} chars")
    for d in image_docs[:3]:
        print(f"  - image: {d.image_path}")
