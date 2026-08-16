"""
Multimodal RAG generation - the piece that actually closes the loop on
"multimodal AI app": embeddings/clip_embedder.py retrieves relevant images
by similarity, but a VLM is what actually looks at the retrieved image and
reasons over it alongside any retrieved text context, in one answer.

Wraps either VLM backend (vlm/ollama_vlm.py or vlm/hf_vlm.py) behind the
same .generate(question, retrieved_chunks) interface used by
generation/ollama_generator.py and generation/hf_generator.py, so
pipeline.py can swap generators without special-casing multimodal results.
"""

from generation.prompts import build_rag_prompt


class MultimodalGenerator:
    def __init__(self, vlm_backend: str = "ollama", vlm_model: str = "llava"):
        if vlm_backend == "ollama":
            from vlm.ollama_vlm import OllamaVLM
            self._vlm = OllamaVLM(vlm_model)
        elif vlm_backend == "hf":
            from vlm.hf_vlm import HFVLM
            self._vlm = HFVLM(vlm_model)
        else:
            raise ValueError(f"Unknown vlm_backend: {vlm_backend}")
        self.name = f"multimodal:{self._vlm.name}"

    def generate(self, question: str, retrieved_chunks: list[dict]) -> str:
        """
        retrieved_chunks may mix text chunks (dict with 'text') and image
        chunks (dict with metadata['image_path'] set, e.g. from CLIP
        retrieval over ingested images/PDF figures). Text chunks become
        context passed alongside the image; if multiple images were
        retrieved, only the top-scoring one is passed to the VLM call itself
        (most local VLMs handle a single image per call reliably) — the
        rest are noted by filename in the text context instead of dropped
        silently.
        """
        image_chunks = [c for c in retrieved_chunks if c.get("metadata", {}).get("image_path")]
        text_chunks = [c for c in retrieved_chunks if not c.get("metadata", {}).get("image_path")]

        if not image_chunks:
            # No image in this result set — fall back to plain text generation
            # via the same VLM's underlying chat call isn't wired for pure
            # text here by design (that's what OllamaGenerator/HFGenerator
            # are for); surface this clearly instead of silently degrading.
            raise ValueError(
                "MultimodalGenerator was called with no image chunks in the retrieved "
                "results — use generation/ollama_generator.py or hf_generator.py for "
                "text-only questions instead."
            )

        primary_image = image_chunks[0]
        other_image_names = [c["metadata"].get("filename", "unknown") for c in image_chunks[1:]]

        text_context = ""
        if text_chunks:
            text_context = build_rag_prompt(question, text_chunks)
        if other_image_names:
            note = f"\n(Additional retrieved images not shown to the model: {', '.join(other_image_names)})"
            text_context += note

        return self._vlm.answer_with_image(question, primary_image["metadata"]["image_path"], text_context)


if __name__ == "__main__":
    print("This module needs a live VLM backend (Ollama llava/moondream, or HF moondream2). "
          "See vlm/benchmark_vlms.py for a runnable smoke test, or pipeline.py's --generator vlm flag.")
