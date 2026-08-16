"""
Serving method 2: vLLM's OpenAI-compatible API server.

Unlike vllm_offline.py (batch jobs, one Python process), this talks to a
long-running local server — the same pattern as your Ollama setup, but
with vLLM's continuous-batching engine underneath. This is the shape
you'd actually deploy: start the server once, then any number of
`ask` calls (or other apps, or a web frontend) hit it over HTTP.

No API key needed — this is a local server, and vLLM's OpenAI-compatible
endpoint accepts any placeholder string as the "key".

Start the server first (separate terminal, requires a CUDA GPU):
    vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000

Requires:
    pip install vllm openai

Run directly to smoke-test (server must already be running):
    python -m serving.vllm_openai_server
"""

from generation.prompts import RAG_SYSTEM_PROMPT, build_rag_prompt

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class VLLMServerGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 base_url: str = "http://localhost:8000/v1", max_tokens: int = 512):
        if OpenAI is None:
            raise ImportError("Run: pip install openai")
        self.name = f"vllm-server:{model_name}"
        self.model_name = model_name
        self.max_tokens = max_tokens
        # api_key is required by the client library but ignored by vLLM's
        # local server — any non-empty string works, no real key needed.
        self._client = OpenAI(base_url=base_url, api_key="not-needed")

    def generate(self, question: str, retrieved_chunks: list[dict], system_prompt: str = None) -> str:
        prompt = build_rag_prompt(question, retrieved_chunks)
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt or RAG_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.max_tokens,
            temperature=0.0,
        )
        return response.choices[0].message.content

    def health_check(self) -> bool:
        try:
            self._client.models.list()
            return True
        except Exception:
            return False


if __name__ == "__main__":
    generator = VLLMServerGenerator("Qwen/Qwen2.5-1.5B-Instruct")
    if not generator.health_check():
        print("vLLM server not reachable at http://localhost:8000 — start it first with:\n"
              "  vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000")
    else:
        fake_chunks = [{"text": "Paris is the capital of France.", "metadata": {"filename": "geo.txt"}}]
        print(generator.generate("What is the capital of France?", fake_chunks))
