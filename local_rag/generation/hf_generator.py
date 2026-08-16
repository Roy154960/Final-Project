"""
Generate step - via local Hugging Face weights (free, downloaded once
then cached, no API key). Slower than Ollama on CPU-only machines but
useful for direct comparison and for models Ollama doesn't package.

Run directly to smoke-test:
    python -m generation.hf_generator
"""

from generation.prompts import RAG_SYSTEM_PROMPT, build_rag_prompt

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    torch = None


class HFGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", max_new_tokens: int = 512):
        if torch is None:
            raise ImportError("Run: pip install transformers torch")
        self.name = f"hf:{model_name}"
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if not torch.cuda.is_available():
            self._model = self._model.to("cpu")

    def generate(self, question: str, retrieved_chunks: list[dict], system_prompt: str = None) -> str:
        prompt = build_rag_prompt(question, retrieved_chunks)
        messages = [
            {"role": "system", "content": system_prompt or RAG_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        input_ids = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self._model.device)

        output = self._model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        new_tokens = output[0][input_ids.shape[-1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)


if __name__ == "__main__":
    generator = HFGenerator("Qwen/Qwen2.5-1.5B-Instruct")
    fake_chunks = [{"text": "Paris is the capital of France.", "metadata": {"filename": "geo.txt"}}]
    print(generator.generate("What is the capital of France?", fake_chunks))
