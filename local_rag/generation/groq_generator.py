"""
Generate step -- via Groq's hosted free tier (fast, no local GPU/CPU
cost). Matches OllamaGenerator's own .generate()/.name interface exactly
(see generation/ollama_generator.py's own docstring) so
mcp_server/server.py and pipeline.py's get_generator() can hold either
one behind the same variable without caring which backend actually
answered.

Used ONLY through generation/fallback_generator.py's FallbackGenerator
in practice -- see that module's own docstring for why nothing in this
project should construct GroqGenerator directly and skip the automatic
local-Ollama fallback it provides. Kept as its own standalone class
anyway (rather than folded straight into FallbackGenerator) for the same
reason OllamaGenerator/HFGenerator are each their own class: one small,
independently-smoke-testable unit per backend.

Model: llama-3.3-70b-versatile by default -- Groq's general-purpose
"versatile" tier (current free-tier RPM/RPD/TPM at
https://console.groq.com/docs/rate-limits), matching the LARGE reasoning
tier agents/llm_provider.py already picks for this project's other
Groq-first call sites, since grounding an answer in retrieved chunks is
squarely a "needs real reasoning," not a simple-lookup, job.

Run directly to smoke-test:
    python -m generation.groq_generator
"""

from config import GROQ_LARGE_MODEL
from generation.prompts import RAG_SYSTEM_PROMPT, build_rag_prompt
from groq_client import groq_chat_completion


class GroqGenerator:
    def __init__(self, model: str = GROQ_LARGE_MODEL):
        self.name = f"groq:{model}"
        self.model = model

    def generate(self, question: str, retrieved_chunks: list[dict]) -> str:
        prompt = build_rag_prompt(question, retrieved_chunks)
        data = groq_chat_completion(
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=self.model,
            tier="large",
            node="generate_answer",
        )
        return data["choices"][0]["message"]["content"]


if __name__ == "__main__":
    generator = GroqGenerator()
    fake_chunks = [{"text": "Paris is the capital of France.", "metadata": {"filename": "geo.txt"}}]
    print(f"[{generator.name}]")
    print(generator.generate("What is the capital of France?", fake_chunks))
