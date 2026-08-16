"""
Ingest enhancement - VLM-generated whole-page descriptions
(Strategy 3: Vision LLMs for Document Understanding).

This is genuinely different from ingestion/image_captioning.py: that
module only ever looks at individually-extracted embedded RASTER images
(page.get_images()). This module hands the VLM a full RENDER of the page
itself, so it can read/interpret content raster extraction can never
catch — a chart or diagram drawn as vector paths (matplotlib/Excel/
PowerPoint-style output embedded directly in the PDF, never a raster
image), or a dense multi-column layout that's easier to describe from a
picture than to reconstruct from PyMuPDF's flattened text order.

Cost control (this is the slow, expensive path — one real VLM call per
page it runs on):
    This module NEVER decides which pages to run on. It only ever
    processes RawDocuments that ingestion/ingest_pdf.py has ALREADY
    flagged (metadata["page_image_path"] set), via its cheap,
    no-model-call describe_pages="auto" heuristic (vector-drawing density
    vs. native text length — see PAGE_VISUAL_COMPLEXITY_* in config.py).
    So a 200-page mostly-text PDF with 3 chart-heavy pages costs exactly
    3 VLM calls here, not 200 — the expensive half only ever sees the
    pre-filtered subset. See pipeline.py's describe_complex_pages() for
    how the two halves are wired together and gated behind --multimodal.

Requires an Ollama or HF VLM (see vlm/ollama_vlm.py, vlm/hf_vlm.py) — the
same backends already used for ingestion/image_captioning.py and
--generator vlm.

Run directly to smoke-test:
    python -m ingestion.page_description data/raw/sample_page_render.png
"""

from typing import Optional

from config import PAGE_DESCRIPTION_PROMPT
from ingestion.schema import RawDocument


def load_vlm(vlm_backend: str, vlm_model: Optional[str] = None):
    """Thin re-export of image_captioning.load_vlm() — same VLM backends,
    same loading logic. Kept as its own function (rather than a bare
    import alias) so callers can `from ingestion.page_description import
    load_vlm` without needing to know it happens to delegate."""
    from ingestion.image_captioning import load_vlm as _load_vlm
    return _load_vlm(vlm_backend, vlm_model)


def describe_pages(
    page_docs: list[RawDocument],
    vlm_backend: str = "ollama",
    vlm_model: Optional[str] = None,
    vlm=None,
) -> list[str]:
    """
    page_docs: RawDocument objects with modality == "pdf_text" whose
    metadata["page_image_path"] is set — i.e. already flagged by
    ingest_pdf.py's describe_pages heuristic (or an explicit "always"
    override). Callers are expected to pre-filter to this subset before
    calling (see pipeline.py's describe_complex_pages()) — this function
    does not re-check the heuristic, it just describes whatever it's given.

    vlm: pass an already-loaded VLM instance to skip reloading one — same
    reasoning as image_captioning.caption_images(): matters most for
    --vlm-backend hf (a real in-process model load), free either way for
    ollama's thin client. Reuse the SAME instance as image captioning
    where possible (see pipeline.py) rather than loading two.

    Returns "" (not a raised exception) for any page that fails to
    describe, so one bad page/VLM hiccup doesn't abort the rest of the
    batch — same fail-soft pattern as image_captioning.caption_images();
    the caller should skip dual-indexing an empty description.
    """
    if not page_docs:
        return []
    if vlm is None:
        vlm = load_vlm(vlm_backend, vlm_model)

    descriptions: list[str] = []
    for doc in page_docs:
        page_image_path = doc.metadata.get("page_image_path")
        if not page_image_path:
            # Defensive: shouldn't happen if the caller filtered correctly
            # (see docstring above), but fail soft rather than crash the
            # batch over a caller bug.
            print(f"[warn] describe_pages() got a doc with no page_image_path "
                  f"(page {doc.metadata.get('page')!r}) — skipping")
            descriptions.append("")
            continue
        try:
            descriptions.append(vlm.answer_with_image(PAGE_DESCRIPTION_PROMPT, page_image_path).strip())
        except Exception as e:
            print(f"[warn] page description failed for {page_image_path}: {e}")
            descriptions.append("")
    return descriptions


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample_page_render.png"
    fake_doc = RawDocument.new(source_path=path, modality="pdf_text", content="placeholder", page=1)
    fake_doc.metadata["page_image_path"] = path
    result = describe_pages([fake_doc])
    print(result[0] if result and result[0] else "(no description — is Ollama running with a VLM pulled?)")
