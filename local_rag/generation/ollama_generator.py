"""
Generate step - via local Ollama server (free, local, no API key).

Requires:
    ollama serve
    ollama pull llama3.2
    ollama pull mistral
    ollama pull phi3

Run directly to smoke-test:
    python -m generation.ollama_generator
"""

from config import OLLAMA_HOST, OLLAMA_NUM_CTX
from generation.prompts import RAG_SYSTEM_PROMPT, build_rag_prompt

try:
    import ollama as ollama_lib
except ImportError:
    ollama_lib = None


class OllamaGenerator:
    def __init__(self, model: str = "llama3.2", host: str = OLLAMA_HOST):
        if ollama_lib is None:
            raise ImportError("Run: pip install ollama")
        self.name = f"ollama:{model}"
        self.model = model
        self.client = ollama_lib.Client(host=host)

    def generate(self, question: str, retrieved_chunks: list[dict]) -> str:
        prompt = build_rag_prompt(question, retrieved_chunks)
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            # num_ctx=OLLAMA_NUM_CTX (config.py): with no explicit context
            # length, Ollama's own default for a model can be the model's
            # full trained max context (128K+ for some models), which asks
            # for a KV-cache buffer far bigger than most machines have
            # free RAM for -- a confirmed out-of-memory failure at
            # model-load time, not a clean error. Every local Ollama call
            # in this project shares this same config.py default.
            options={"num_ctx": OLLAMA_NUM_CTX},
        )
        return response["message"]["content"]


if __name__ == "__main__":
    generator = OllamaGenerator("llama3.2")
    fake_chunks = [{"text": "Paris is the capital of France.", "metadata": {"filename": "geo.txt"}}]
    print(generator.generate("What is the capital of France?", fake_chunks))
