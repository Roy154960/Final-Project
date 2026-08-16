"""
Serving method 1: vLLM offline batch inference.

vLLM's main advantage over plain HF transformers or Ollama for this
project is throughput under concurrent load: PagedAttention + continuous
batching let many in-flight requests share GPU memory efficiently, instead
of processing one request fully before starting the next. For a RAG app
serving multiple users, that's the difference between "fast for one
question" and "fast for a hundred questions at once."

This module uses vLLM's Python API directly for OFFLINE batch generation
(you hand it a list of prompts, get back a list of completions) — good for
benchmarking throughput, or for a batch job like "regenerate answers for
every question in an eval set." For an actual live server other apps can
call, see serving/vllm_openai_server.py instead.

Requires a CUDA GPU (vLLM's core kernels are GPU-only) and:
    pip install vllm

Run directly to smoke-test:
    python -m serving.vllm_offline
"""

from generation.prompts import RAG_SYSTEM_PROMPT, build_rag_prompt

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None


class VLLMOfflineGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", max_tokens: int = 512):
        if LLM is None:
            raise ImportError("Run: pip install vllm (requires a CUDA GPU)")
        self.name = f"vllm-offline:{model_name}"
        self.model_name = model_name
        self._llm = LLM(model=model_name)
        self._sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    def _build_prompt(self, question: str, retrieved_chunks: list[dict]) -> str:
        rag_prompt = build_rag_prompt(question, retrieved_chunks)
        # vLLM's offline API takes raw text, so we apply a simple chat
        # template by hand rather than pulling in a tokenizer just for this.
        return f"<|system|>\n{RAG_SYSTEM_PROMPT}\n<|user|>\n{rag_prompt}\n<|assistant|>\n"

    def generate(self, question: str, retrieved_chunks: list[dict]) -> str:
        """Single-question convenience wrapper — for the real throughput
        benefit, use generate_batch() with many questions at once instead."""
        return self.generate_batch([question], [retrieved_chunks])[0]

    def generate_batch(self, questions: list[str], retrieved_chunks_list: list[list[dict]]) -> list[str]:
        """
        This is the point of vLLM: hand it N prompts at once and continuous
        batching processes them far more efficiently than N sequential
        .generate() calls to a plain HF model or an Ollama server, which
        largely serialize requests instead.
        """
        prompts = [self._build_prompt(q, chunks) for q, chunks in zip(questions, retrieved_chunks_list)]
        outputs = self._llm.generate(prompts, self._sampling_params)
        return [output.outputs[0].text for output in outputs]


if __name__ == "__main__":
    generator = VLLMOfflineGenerator("Qwen/Qwen2.5-1.5B-Instruct")
    fake_chunks = [{"text": "Paris is the capital of France.", "metadata": {"filename": "geo.txt"}}]
    questions = ["What is the capital of France?", "What is the largest city in France?"]
    answers = generator.generate_batch(questions, [fake_chunks, fake_chunks])
    for q, a in zip(questions, answers):
        print(f"Q: {q}\nA: {a}\n")
