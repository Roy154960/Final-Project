"""
Compare VLM backends/models on the same image + question.

Like the other benchmarks in this project, quality here needs real labeled
data to mean anything — this ships with a runnable smoke-test harness
(latency + a basic non-empty/non-error sanity check), and a spot to plug in
your own (image_path, question, expected_keywords) tuples once you have
real images to test against.

Run:
    python -m vlm.benchmark_vlms path/to/image.png "What is shown in this image?"
"""

import sys
import time
from pathlib import Path

# Fill in with your own labeled examples for real quality signal:
#   {"image": "data/raw/invoice.png", "question": "What is the total amount?",
#    "expected_keywords": ["total", "$"]}
EVAL_EXAMPLES: list[dict] = []


def _keyword_hit_rate(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 1.0  # no keywords specified — can't score, don't penalize
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits / len(expected_keywords)


def _bench_backend(name: str, vlm, image_path: str, question: str, expected_keywords: list[str] = None) -> dict:
    start = time.perf_counter()
    try:
        answer = vlm.answer_with_image(question, image_path)
        elapsed = time.perf_counter() - start
        hit_rate = _keyword_hit_rate(answer, expected_keywords or [])
        return {"backend": name, "latency_s": round(elapsed, 2), "keyword_hit_rate": round(hit_rate, 2),
                "answer_preview": answer[:100], "error": None}
    except Exception as e:
        return {"backend": name, "latency_s": None, "keyword_hit_rate": None,
                "answer_preview": None, "error": str(e)}


def benchmark(image_path: str, question: str, expected_keywords: list[str] = None):
    results = []

    try:
        from vlm.ollama_vlm import OllamaVLM
        results.append(_bench_backend("ollama:llava", OllamaVLM("llava"), image_path, question, expected_keywords))
    except Exception as e:
        print(f"[skip] ollama:llava setup failed: {e}")

    try:
        from vlm.hf_vlm import HFVLM
        results.append(_bench_backend("hf:moondream2", HFVLM("moondream2"), image_path, question, expected_keywords))
    except Exception as e:
        print(f"[skip] hf:moondream2 setup failed: {e}")

    print(f"\n{'backend':<18}{'latency_s':<12}{'keyword_hit_rate':<18}{'answer_preview'}")
    for r in results:
        if r["error"]:
            print(f"{r['backend']:<18}FAILED: {r['error']}")
        else:
            print(f"{r['backend']:<18}{r['latency_s']:<12}{r['keyword_hit_rate']:<18}{r['answer_preview']}")

    return results


def run_full_eval_set():
    """Run every labeled example in EVAL_EXAMPLES across all backends, for a
    real (not single-example smoke-test) quality signal once you've filled
    EVAL_EXAMPLES in above."""
    if not EVAL_EXAMPLES:
        print("EVAL_EXAMPLES is empty — add a few (image, question, expected_keywords) "
              "tuples at the top of this file for a real quality signal.")
        return
    for example in EVAL_EXAMPLES:
        print(f"\n=== {example['image']} :: {example['question']} ===")
        benchmark(example["image"], example["question"], example.get("expected_keywords"))


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample.png"
    question = sys.argv[2] if len(sys.argv) > 2 else "What is shown in this image?"

    if not Path(image_path).exists():
        print(f"No image found at {image_path} — pass a real image path to test this.")
    else:
        benchmark(image_path, question)
