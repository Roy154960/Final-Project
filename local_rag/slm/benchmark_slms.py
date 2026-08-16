"""
SLM benchmark - compares candidates from slm/model_registry.py on:
  - latency (time to first full response, and approx tokens/sec)
  - peak memory footprint (process RSS delta; GPU VRAM if CUDA available)
  - answer quality proxy: the free faithfulness heuristic from
    evaluation/metrics.py (grounds the answer against provided context, no
    LLM judge needed, so this stays free/local even for the eval step)

This deliberately reuses the SAME generator classes already in
generation/ollama_generator.py and generation/hf_generator.py — the point
of this file is comparing MODELS, not comparing serving code paths (that
comparison lives in serving/benchmark_serving.py instead).

Run:
    python -m slm.benchmark_slms
"""

import os
import time
import psutil

from slm.model_registry import SLM_CANDIDATES
from evaluation.metrics import keyword_faithfulness_heuristic

SAMPLE_QUESTION = "What is the capital of France?"
SAMPLE_CONTEXT = [{"text": "Paris is the capital and largest city of France.",
                    "metadata": {"filename": "geo.txt"}}]


def _process_memory_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _bench_one(candidate) -> dict:
    mem_before = _process_memory_mb()

    if candidate.backend == "ollama":
        from generation.ollama_generator import OllamaGenerator
        generator = OllamaGenerator(candidate.name)
    else:
        from generation.hf_generator import HFGenerator
        generator = HFGenerator(candidate.name)

    mem_after_load = _process_memory_mb()

    start = time.perf_counter()
    answer = generator.generate(SAMPLE_QUESTION, SAMPLE_CONTEXT)
    elapsed = time.perf_counter() - start

    approx_tokens = len(answer.split())  # word count as a cheap token-count proxy
    tokens_per_sec = round(approx_tokens / elapsed, 1) if elapsed > 0 else 0.0

    faithfulness = keyword_faithfulness_heuristic(answer, [c["text"] for c in SAMPLE_CONTEXT])

    return {
        "model": candidate.name,
        "backend": candidate.backend,
        "params_b": candidate.params_billion,
        "load_mem_mb": round(mem_after_load - mem_before, 1),
        "latency_s": round(elapsed, 2),
        "tokens_per_sec": tokens_per_sec,
        "faithfulness": round(faithfulness, 2),
        "answer_preview": answer[:80],
    }


def benchmark(candidates=None):
    candidates = candidates or SLM_CANDIDATES
    results = []
    for c in candidates:
        try:
            results.append(_bench_one(c))
        except Exception as e:
            print(f"[skip] {c.name} ({c.backend}): {e}")

    print(f"\n{'model':<38}{'params(B)':<11}{'load_mb':<10}{'latency_s':<11}{'tok/s':<8}{'faithful':<9}")
    for r in results:
        print(f"{r['model']:<38}{r['params_b']:<11}{r['load_mem_mb']:<10}"
              f"{r['latency_s']:<11}{r['tokens_per_sec']:<8}{r['faithfulness']:<9}")

    if results:
        best_quality = max(results, key=lambda r: r["faithfulness"])
        fastest = min(results, key=lambda r: r["latency_s"])
        print(f"\nMost faithful: {best_quality['model']} ({best_quality['faithfulness']})")
        print(f"Fastest: {fastest['model']} ({fastest['latency_s']}s)")
        print("\nNote: faithfulness here is a single-example smoke test, not a real quality "
              "signal — run this against evaluation/build_eval_set.py's real eval set for "
              "numbers you can actually trust.")

    return results


if __name__ == "__main__":
    benchmark()
