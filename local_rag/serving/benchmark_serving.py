"""
Compare SERVING infrastructure (not models) on the same underlying model
where possible: how fast can each backend answer a batch of questions?

This is the deliberate counterpart to slm/benchmark_slms.py, which compares
different MODELS on one serving path. Here the model is held constant
(Qwen2.5-1.5B-Instruct where every backend supports it) and only the
serving engine changes:

  - generation/hf_generator.py    : plain transformers, sequential .generate() calls
  - generation/ollama_generator.py: Ollama server, sequential calls
  - serving/vllm_offline.py       : vLLM, TRUE batch — this is where it should win

The gap between vLLM's batch throughput and the other two sequential paths
is the actual point of this benchmark: it should widen as N_QUESTIONS grows,
since continuous batching's advantage compounds with concurrency while
sequential request handling doesn't.

Run:
    python -m serving.benchmark_serving
"""

import time

N_QUESTIONS = 8  # small by default so this is a fast smoke test; raise this
                 # to see vLLM's batching advantage widen with more concurrency

QUESTIONS = [f"What is {i} plus {i}?" for i in range(1, N_QUESTIONS + 1)]
CONTEXT = [{"text": "This is a simple arithmetic question with no retrieved document needed.",
            "metadata": {"filename": "none"}}]


def _bench_sequential(name: str, generator) -> dict:
    start = time.perf_counter()
    for q in QUESTIONS:
        generator.generate(q, CONTEXT)
    elapsed = time.perf_counter() - start
    return {"backend": name, "total_s": round(elapsed, 2),
            "questions_per_sec": round(N_QUESTIONS / elapsed, 2) if elapsed > 0 else 0.0}


def _bench_vllm_batch(generator) -> dict:
    start = time.perf_counter()
    generator.generate_batch(QUESTIONS, [CONTEXT] * N_QUESTIONS)
    elapsed = time.perf_counter() - start
    return {"backend": "vllm-offline (true batch)", "total_s": round(elapsed, 2),
            "questions_per_sec": round(N_QUESTIONS / elapsed, 2) if elapsed > 0 else 0.0}


def benchmark(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    results = []

    try:
        from generation.hf_generator import HFGenerator
        results.append(_bench_sequential("hf-transformers (sequential)", HFGenerator(model_name)))
    except Exception as e:
        print(f"[skip] HF transformers: {e}")

    try:
        from generation.ollama_generator import OllamaGenerator
        # Ollama doesn't ship Qwen2.5-1.5B by that exact tag; use its closest
        # small local equivalent for this comparison. Swap to a matching
        # model on your machine if you want a strictly apples-to-apples run.
        results.append(_bench_sequential("ollama (sequential)", OllamaGenerator("llama3.2")))
    except Exception as e:
        print(f"[skip] Ollama: {e}")

    try:
        from serving.vllm_offline import VLLMOfflineGenerator
        results.append(_bench_vllm_batch(VLLMOfflineGenerator(model_name)))
    except Exception as e:
        print(f"[skip] vLLM offline: {e} (requires a CUDA GPU + `pip install vllm`)")

    print(f"\nBenchmarking {N_QUESTIONS} questions per backend\n")
    print(f"{'backend':<32}{'total_s':<12}{'questions/sec':<15}")
    for r in results:
        print(f"{r['backend']:<32}{r['total_s']:<12}{r['questions_per_sec']:<15}")

    if len(results) > 1:
        print("\nNote: HF transformers and Ollama here process requests sequentially "
              "(no batching), which is the realistic default for both. vLLM's number "
              "reflects true concurrent batch processing — the gap should widen further "
              "if you raise N_QUESTIONS.")

    return results


if __name__ == "__main__":
    benchmark()
