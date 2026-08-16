"""
Ingest enhancement - VLM-generated image captions.

Used by pipeline.py's --multimodal ingest path. Each image gets a short
VLM-written description at ingest time, for two purposes:
  1. Embedded via the TEXT embedder and dual-indexed into the TEXT vector
     store (see pipeline.py cmd_ingest), so images are also reachable by
     plain semantic text search — not only by CLIP's own (weaker,
     cross-modal) text-to-image similarity.
  2. Stashed as metadata on the image's own entry in the CLIP image store,
     so generation/dual_modality_generator.py's image branch can hand the
     VLM a bit of pre-written context alongside the raw image at answer time.

Requires an Ollama or HF VLM (see vlm/ollama_vlm.py, vlm/hf_vlm.py) — the
same backends already used for --generator vlm.

Run directly to smoke-test:
    python -m ingestion.image_captioning data/raw/sample.png
"""

from typing import Optional

import time

from config import IMAGE_CAPTION_PROMPT
from ingestion.schema import RawDocument

# Conservative pacing between consecutive Groq calls WITHIN a captioning
# batch -- see caption_images()'s own comment on where this is applied
# and why it's Groq-only (ollama/hf have no external rate limit to
# protect, so pacing those would be a pure, pointless slowdown). Groq's
# free tier is roughly 30 requests/minute (see groq_client.py's own
# retry-after handling for what happens when a call blows through that
# budget anyway) -- pacing one call every _GROQ_CAPTION_PACING_SECONDS
# keeps a whole batch comfortably under that ceiling for its ENTIRE
# duration, rather than racing through the first ~10-15 images, hitting
# 429 repeatedly, and silently falling back to local Ollama for most of
# a large batch -- which is what a 39-image batch was observed doing
# live (2 real Groq captions, then 429s the rest of the way). Set
# somewhat above the exact 2.0s/request math (60s / 30 requests) as a
# safety margin against other Groq traffic sharing the same
# organization-level budget (see groq_client.py's own docs link).
_GROQ_CAPTION_PACING_SECONDS = 2.2


def load_vlm(vlm_backend: str, vlm_model: Optional[str] = None):
    if vlm_backend == "ollama":
        from vlm.ollama_vlm import OllamaVLM
        return OllamaVLM(vlm_model or "llava")
    if vlm_backend == "hf":
        from vlm.hf_vlm import HFVLM
        return HFVLM(vlm_model or "moondream2")
    if vlm_backend == "groq":
        # Groq-first, automatic local-Ollama-fallback -- see
        # vlm/fallback_vlm.py's own module docstring. This is the
        # default for config.PERSONAL_RAG_SINGLE_IMAGE_VLM_BACKEND, and
        # the ONLY hosted/online backend this project uses -- Groq or
        # local, no other option.
        from vlm.fallback_vlm import FallbackVLM
        return FallbackVLM(vlm_model) if vlm_model else FallbackVLM()
    raise ValueError(f"Unknown vlm_backend: {vlm_backend}")


def caption_images(image_docs: list[RawDocument], vlm_backend: str = "ollama",
                    vlm_model: Optional[str] = None, vlm=None) -> list[str]:
    """
    image_docs: RawDocument objects with modality == "image" (image_path set).

    vlm: pass an already-loaded VLM instance (see load_vlm() above) to skip
    reloading one — matters most for --vlm-backend hf, where each fresh
    instance is a real in-process model load; for ollama it's a thin client
    object either way, but reuse is still free where the caller already has
    one (see api.py's _get_caption_vlm()).

    Otherwise loads the VLM ONCE and reuses it across every image in the
    batch rather than reloading per image — the model load is the expensive
    part, same reasoning as embeddings/hf_embedder.py loading its model once
    in __init__.

    Paces itself between calls when the loaded VLM's own `.name` mentions
    "groq" (FallbackVLM's name is "fallback:groq-vlm:{model}->ollama" --
    see vlm/fallback_vlm.py -- so this catches it regardless of whether
    the caller got here via vlm_backend="groq" or passed an already-
    loaded `vlm` directly) -- see _GROQ_CAPTION_PACING_SECONDS' own
    comment for why. ollama/hf batches run at full speed, unpaced, same
    as before.

    Returns "" (not a raised exception) for any image that fails to caption,
    so one bad image doesn't abort captioning of the rest of the batch — the
    caller should skip dual-indexing an empty caption (see pipeline.py).
    """
    if not image_docs:
        return []
    if vlm is None:
        vlm = load_vlm(vlm_backend, vlm_model)

    is_groq = "groq" in getattr(vlm, "name", "").lower()

    n = len(image_docs)
    print(f"[caption] captioning {n} image(s) via {vlm_backend}"
          f"{f' ({vlm_model})' if vlm_model else ''}"
          f"{f' -- paced at one call/{_GROQ_CAPTION_PACING_SECONDS}s to stay under the Groq free-tier rate limit' if is_groq else ''}...")

    captions = []
    for i, doc in enumerate(image_docs, start=1):
        label = f"page {doc.metadata.get('page')}" if doc.metadata.get("page") else doc.image_path
        try:
            caption = vlm.answer_with_image(IMAGE_CAPTION_PROMPT, doc.image_path).strip()
            captions.append(caption)
            preview = (caption[:70] + "...") if len(caption) > 70 else caption
            print(f"[caption] {i}/{n} {label} ({doc.image_path}) -> {preview!r}")
        except Exception as e:
            print(f"[warn] captioning failed for {doc.image_path} ({i}/{n}): {e}")
            captions.append("")
        if is_groq and i < n:
            time.sleep(_GROQ_CAPTION_PACING_SECONDS)
    print(f"[caption] done -> {sum(1 for c in captions if c)}/{n} captioned successfully")
    return captions


if __name__ == "__main__":
    import sys
    from ingestion.schema import RawDocument

    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample.png"
    fake_doc = RawDocument.new(source_path=path, modality="image", content="", image_path=path)
    result = caption_images([fake_doc])
    print(result[0] if result and result[0] else "(no caption — is Ollama running with a VLM pulled?)")
