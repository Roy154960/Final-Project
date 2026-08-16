"""
SLM registry - candidate small language models, all free/local, across both
serving backends already in this project (Ollama and HF transformers).

"Small" here means roughly sub-8B parameters: small enough to run on a
laptop CPU or a modest consumer GPU, which matters a lot for a RAG pipeline
where generation happens on every single question.

This is metadata only (no model loading) so you can reason about tradeoffs
before spending time pulling/downloading anything.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class SLMCandidate:
    name: str
    backend: Literal["ollama", "hf"]
    params_billion: float
    context_length: int
    notes: str


SLM_CANDIDATES: list[SLMCandidate] = [
    SLMCandidate(
        name="phi3",
        backend="ollama",
        params_billion=3.8,
        context_length=4096,
        notes="Microsoft Phi-3-mini, GGUF-quantized by default via Ollama. Fast, strong for its size.",
    ),
    SLMCandidate(
        name="llama3.2",
        backend="ollama",
        params_billion=3.0,
        context_length=128_000,
        notes="Meta's smallest Llama 3.2 variant. Long context, good general instruction-following.",
    ),
    SLMCandidate(
        name="mistral",
        backend="ollama",
        params_billion=7.0,
        context_length=8192,
        notes="Larger end of 'small' — strongest reasoning of the Ollama trio, slowest of the three.",
    ),
    SLMCandidate(
        name="Qwen/Qwen2.5-1.5B-Instruct",
        backend="hf",
        params_billion=1.5,
        context_length=32_768,
        notes="Smallest model in this registry. Good for fast CPU-only iteration during development.",
    ),
    SLMCandidate(
        name="Qwen/Qwen2.5-0.5B-Instruct",
        backend="hf",
        params_billion=0.5,
        context_length=32_768,
        notes="Extremely small — useful as a latency floor / worst-case-quality baseline in benchmarks.",
    ),
    SLMCandidate(
        name="microsoft/Phi-3-mini-4k-instruct",
        backend="hf",
        params_billion=3.8,
        context_length=4096,
        notes="Same model family as Ollama's phi3, but full-precision HF weights instead of GGUF — useful\n"
              "for isolating 'quantization effect' vs 'model effect' when compared against ollama:phi3.",
    ),
]


def get_candidate(name: str) -> SLMCandidate:
    for c in SLM_CANDIDATES:
        if c.name == name:
            return c
    raise ValueError(f"Unknown SLM candidate: {name}. Options: {[c.name for c in SLM_CANDIDATES]}")


if __name__ == "__main__":
    print(f"{'name':<38}{'backend':<10}{'params(B)':<12}{'context':<10}")
    for c in SLM_CANDIDATES:
        print(f"{c.name:<38}{c.backend:<10}{c.params_billion:<12}{c.context_length:<10}")
