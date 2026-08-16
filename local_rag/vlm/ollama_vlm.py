"""
VLM method 1: vision-language model via Ollama (free, local).

This is genuinely different from embeddings/clip_embedder.py: CLIP tells
you an image and a text query are *similar* (good for retrieval), but it
can't answer a question about what's actually in the image or reason over
it. A VLM (e.g. llava, moondream, bakllava) takes the raw image plus a text
prompt and produces a real, reasoned answer — so once CLIP or a query
retrieves the right image, the VLM is what actually looks at it and answers.

Requires:
    ollama pull llava        # or: ollama pull moondream (smaller, faster)

Run directly to smoke-test:
    python -m vlm.ollama_vlm data/raw/sample.png "What is shown in this image?"
"""

import sys
from pathlib import Path

from config import OLLAMA_HOST, OLLAMA_NUM_CTX

try:
    import ollama as ollama_lib
except ImportError:
    ollama_lib = None


class OllamaVLM:
    def __init__(self, model: str = "llava", host: str = OLLAMA_HOST):
        if ollama_lib is None:
            raise ImportError("Run: pip install ollama")
        self.name = f"ollama-vlm:{model}"
        self.model = model
        self.client = ollama_lib.Client(host=host)

    def describe_image(self, image_path: str, prompt: str = "Describe this image in detail.") -> str:
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt, "images": [image_path]}],
            # See answer_with_image's own comment on options={"num_ctx": ...}.
            options={"num_ctx": OLLAMA_NUM_CTX},
        )
        return response["message"]["content"]

    def answer_with_image(self, question: str, image_path: str, text_context: str = "") -> str:
        """
        The multimodal-RAG entry point: combines a retrieved image with any
        retrieved text context and the user's question in one VLM call, so
        the answer is genuinely grounded in what's actually in the image
        rather than a text caption someone else wrote.
        """
        prompt = f"{('Context: ' + text_context + chr(10) + chr(10)) if text_context else ''}Question: {question}"
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt, "images": [image_path]}],
            # num_ctx=OLLAMA_NUM_CTX (config.py): with no explicit context
            # length, Ollama's own default for a model can be the model's
            # full trained max context, which asks for a KV-cache buffer
            # far bigger than most machines have free RAM for -- a
            # confirmed out-of-memory failure at model-load time. Every
            # local Ollama call in this project shares this same default.
            options={"num_ctx": OLLAMA_NUM_CTX},
        )
        return response["message"]["content"]


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample.png"
    question = sys.argv[2] if len(sys.argv) > 2 else "What is shown in this image?"

    if not Path(image_path).exists():
        print(f"No image found at {image_path} — pass a real image path to test this.")
    else:
        vlm = OllamaVLM("llava")
        print(vlm.answer_with_image(question, image_path))
