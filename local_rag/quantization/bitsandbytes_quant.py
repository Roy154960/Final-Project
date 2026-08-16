"""
Quantization method 1: BitsAndBytes (HF transformers backend).

Loads an HF causal LM in 4-bit (NF4) or 8-bit instead of full fp16/fp32,
cutting VRAM/RAM footprint roughly in half (8-bit) or to a quarter
(4-bit) at some quality cost. Free, local, no separate model download —
quantizes the same weights on load.

Requires a CUDA GPU for the 4-bit/8-bit kernels to actually help (bitsandbytes'
int8/int4 matmul kernels are GPU-only); on CPU-only machines this will still
load correctly but bitsandbytes will not provide a memory or speed benefit —
use quantization/gguf_quant.py's approach (Ollama) instead for CPU-only setups.

Run directly to smoke-test (needs a CUDA GPU + `pip install bitsandbytes`):
    python -m quantization.bitsandbytes_quant
"""

from generation.prompts import RAG_SYSTEM_PROMPT, build_rag_prompt

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except ImportError:
    torch = None


class BitsAndBytesGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", bits: int = 4, max_new_tokens: int = 512):
        if torch is None:
            raise ImportError("Run: pip install transformers torch bitsandbytes")
        if bits not in (4, 8):
            raise ValueError("bits must be 4 or 8")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "BitsAndBytes quantized kernels require a CUDA GPU. "
                "On CPU-only machines, use quantization/gguf_quant.py (Ollama) instead."
            )

        self.name = f"bnb-{bits}bit:{model_name}"
        self.model_name = model_name
        self.bits = bits
        self.max_new_tokens = max_new_tokens

        quant_config = (
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
            if bits == 4
            else BitsAndBytesConfig(load_in_8bit=True)
        )

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=quant_config, device_map="auto"
        )

    def generate(self, question: str, retrieved_chunks: list[dict]) -> str:
        prompt = build_rag_prompt(question, retrieved_chunks)
        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        input_ids = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self._model.device)

        output = self._model.generate(input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        new_tokens = output[0][input_ids.shape[-1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def gpu_memory_footprint_mb(self) -> float:
        """Actual allocated VRAM for the loaded model — the real payoff metric
        for quantization, more meaningful than a generic process RSS reading."""
        return torch.cuda.memory_allocated(self._model.device) / (1024 * 1024)


if __name__ == "__main__":
    generator = BitsAndBytesGenerator("Qwen/Qwen2.5-1.5B-Instruct", bits=4)
    fake_chunks = [{"text": "Paris is the capital of France.", "metadata": {"filename": "geo.txt"}}]
    print(generator.generate("What is the capital of France?", fake_chunks))
    print(f"GPU memory: {generator.gpu_memory_footprint_mb():.1f} MB")
