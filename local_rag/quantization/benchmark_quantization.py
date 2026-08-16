"""
Compare quantization approaches on the SAME underlying question, so you can
see the actual memory/latency/quality tradeoff rather than trusting
marketing numbers from any one library.

Two independent axes get compared here:
  1. GGUF quant levels via Ollama (CPU-friendly, works on any machine)
  2. BitsAndBytes 4-bit vs 8-bit vs full precision via HF transformers
     (GPU-only for the memory/speed benefit)

Metrics:
  - process memory delta on load (RSS for Ollama/CPU paths; actual CUDA
    VRAM allocated for BitsAndBytes, which is the more meaningful number
    when a GPU is present)
  - generation latency
  - faithfulness heuristic (free, local, from evaluation/metrics.py) as a
    fast quality proxy — NOT a substitute for a real eval set once you have one

Run:
    python -m quantization.benchmark_quantization
"""

import os
import time
import psutil

from evaluation.metrics import keyword_faithfulness_heuristic

SAMPLE_QUESTION = "What is the capital of France?"
SAMPLE_CONTEXT = [{"text": "Paris is the capital and largest city of France.",
                    "metadata": {"filename": "geo.txt"}}]


def _process_memory_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _run_and_score(generator, memory_fn=None) -> dict:
    mem_before = memory_fn() if memory_fn else _process_memory_mb()
    start = time.perf_counter()
    answer = generator.generate(SAMPLE_QUESTION, SAMPLE_CONTEXT)
    elapsed = time.perf_counter() - start
    mem_after = memory_fn() if memory_fn else _process_memory_mb()

    faithfulness = keyword_faithfulness_heuristic(answer, [c["text"] for c in SAMPLE_CONTEXT])
    return {
        "config": generator.name,
        "mem_delta_mb": round(mem_after - mem_before, 1),
        "latency_s": round(elapsed, 2),
        "faithfulness": round(faithfulness, 2),
    }


def benchmark_gguf_levels(base_model: str = "phi3", levels: list[str] = None):
    from quantization.gguf_quant import build_quant_tags, get_generator_for_tag

    levels = levels or ["q4_0", "q8_0"]  # keep the smoke test small by default; expand as needed
    tags = build_quant_tags(base_model, levels)

    results = []
    for tag in tags:
        try:
            generator = get_generator_for_tag(tag)
            results.append(_run_and_score(generator))
        except Exception as e:
            print(f"[skip] {tag.full_tag}: {e} (did you `ollama pull {tag.full_tag}`?)")
    return results


def benchmark_bitsandbytes(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    from quantization.bitsandbytes_quant import BitsAndBytesGenerator

    results = []
    for bits in (4, 8):
        try:
            generator = BitsAndBytesGenerator(model_name, bits=bits)
            result = _run_and_score(generator, memory_fn=generator.gpu_memory_footprint_mb)
            results.append(result)
        except Exception as e:
            print(f"[skip] bnb-{bits}bit: {e}")
    return results


def _print_table(results: list[dict], title: str):
    if not results:
        return
    print(f"\n--- {title} ---")
    print(f"{'config':<45}{'mem_delta_mb':<15}{'latency_s':<12}{'faithfulness':<12}")
    for r in results:
        print(f"{r['config']:<45}{r['mem_delta_mb']:<15}{r['latency_s']:<12}{r['faithfulness']:<12}")


def benchmark():
    gguf_results = benchmark_gguf_levels()
    _print_table(gguf_results, "GGUF quant levels (Ollama, CPU-friendly)")

    bnb_results = benchmark_bitsandbytes()
    _print_table(bnb_results, "BitsAndBytes 4-bit vs 8-bit (HF, GPU required)")

    if not gguf_results and not bnb_results:
        print("\nNo results — pull at least one Ollama quant tag or run on a CUDA machine "
              "with bitsandbytes installed to see real numbers.")

    return gguf_results, bnb_results


if __name__ == "__main__":
    benchmark()
