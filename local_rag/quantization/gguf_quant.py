"""
Quantization method 2: GGUF quant-tag comparison (Ollama backend).

Ollama models are already shipped as GGUF, pre-quantized — the model tag
after the colon selects the quantization level, e.g.:
    ollama pull phi3:3.8b-mini-4k-instruct-q4_0     # ~4-bit
    ollama pull phi3:3.8b-mini-4k-instruct-q5_K_M   # ~5-bit, K-means quantization
    ollama pull phi3:3.8b-mini-4k-instruct-q8_0     # ~8-bit
    ollama pull phi3:3.8b-mini-4k-instruct-fp16     # full precision baseline

This is the free, CPU-friendly quantization path — unlike BitsAndBytes
(quantization/bitsandbytes_quant.py), no GPU is required to see the benefit;
smaller GGUF quant levels are both smaller AND meaningfully faster on CPU.

Run directly to smoke-test (needs the tags pulled first):
    python -m quantization.gguf_quant
"""

from dataclasses import dataclass
from generation.ollama_generator import OllamaGenerator

# Common GGUF quant naming, roughly ordered smallest/lowest-quality to
# largest/highest-quality. Exact tag availability depends on what's
# published for a given base model on Ollama's library.
GGUF_QUANT_LEVELS = [
    "q2_K",   # smallest, largest quality loss - rarely worth it except extreme memory constraints
    "q4_0",   # classic 4-bit, good default
    "q4_K_M", # 4-bit with K-means clustering, usually better quality than q4_0 at similar size
    "q5_K_M", # 5-bit, noticeably better quality, still much smaller than fp16
    "q8_0",   # 8-bit, close to full quality
    "fp16",   # full precision baseline for comparison
]


@dataclass
class GGUFQuantTag:
    base_model: str
    quant_level: str

    @property
    def full_tag(self) -> str:
        return f"{self.base_model}:{self.quant_level}"


def build_quant_tags(base_model: str, levels: list[str] = None) -> list[GGUFQuantTag]:
    levels = levels or GGUF_QUANT_LEVELS
    return [GGUFQuantTag(base_model, level) for level in levels]


def get_generator_for_tag(tag: GGUFQuantTag) -> OllamaGenerator:
    """Returns an OllamaGenerator pinned to a specific quant level. Requires
    `ollama pull <tag.full_tag>` to have been run first."""
    return OllamaGenerator(tag.full_tag)


if __name__ == "__main__":
    tags = build_quant_tags("phi3", levels=["q4_0", "q8_0"])
    print("Example tags (pull these first with `ollama pull <tag>`):")
    for t in tags:
        print(f"  {t.full_tag}")
    print("\nSee quantization/benchmark_quantization.py to compare them once pulled.")
