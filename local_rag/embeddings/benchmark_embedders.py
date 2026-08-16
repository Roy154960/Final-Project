"""
Compare embedding models across BOTH runtimes (Ollama and Hugging Face),
plus CLIP for the multimodal path, on the same set of sample texts.

Metrics reported per model:
  - dimensions
  - avg time per text (embedding latency)
  - self-similarity sanity check (a text should be closer to a paraphrase
    of itself than to an unrelated sentence — flags a broken/misconfigured
    model rather than a genuinely "bad" one)

This does NOT rank retrieval quality (that needs labeled query/relevant-chunk
pairs — see evaluation/metrics.py for that once you have a test set). It's a
fast first pass to confirm every backend is reachable and behaving sanely
before you commit to one for the full pipeline.

Run:
    python -m embeddings.benchmark_embedders
"""

import time
import numpy as np

from config import OLLAMA_EMBED_MODELS, HF_EMBED_MODELS

SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast, dark-colored fox leaps above a sleepy dog.",  # paraphrase of above
    "Quarterly revenue grew by twelve percent year over year.",  # unrelated
]


def _cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _run_model(name: str, embed_fn) -> dict:
    start = time.perf_counter()
    vecs = embed_fn(SAMPLE_TEXTS)
    elapsed = time.perf_counter() - start

    sim_paraphrase = _cosine(vecs[0], vecs[1])
    sim_unrelated = _cosine(vecs[0], vecs[2])
    sane = sim_paraphrase > sim_unrelated

    return {
        "model": name,
        "dims": vecs.shape[1],
        "time_per_text_ms": round((elapsed / len(SAMPLE_TEXTS)) * 1000, 1),
        "sim_paraphrase": round(sim_paraphrase, 3),
        "sim_unrelated": round(sim_unrelated, 3),
        "sane": sane,
    }


def benchmark():
    results = []

    # --- Ollama models ---
    try:
        from embeddings.ollama_embedder import OllamaEmbedder
        for model in OLLAMA_EMBED_MODELS:
            try:
                embedder = OllamaEmbedder(model)
                results.append(_run_model(embedder.name, embedder.embed_texts))
            except Exception as e:
                print(f"[skip] {model} (Ollama): {e}")
    except ImportError as e:
        print(f"[skip] Ollama backend unavailable: {e}")

    # --- HF models ---
    try:
        from embeddings.hf_embedder import HFEmbedder
        for model in HF_EMBED_MODELS:
            try:
                embedder = HFEmbedder(model)
                results.append(_run_model(embedder.name, embedder.embed_texts))
            except Exception as e:
                print(f"[skip] {model} (HF): {e}")
    except ImportError as e:
        print(f"[skip] HF backend unavailable: {e}")

    # --- CLIP (text side, for comparability; image side tested separately) ---
    try:
        from embeddings.clip_embedder import ClipEmbedder
        try:
            embedder = ClipEmbedder()
            results.append(_run_model(embedder.name, embedder.embed_texts))
        except Exception as e:
            print(f"[skip] CLIP: {e}")
    except ImportError as e:
        print(f"[skip] CLIP backend unavailable: {e}")

    print(f"\n{'model':<40}{'dims':<8}{'ms/text':<10}{'sim_para':<10}{'sim_unrel':<10}{'sane':<6}")
    for r in results:
        print(
            f"{r['model']:<40}{r['dims']:<8}{r['time_per_text_ms']:<10}"
            f"{r['sim_paraphrase']:<10}{r['sim_unrelated']:<10}{r['sane']!s:<6}"
        )
    return results


if __name__ == "__main__":
    benchmark()
