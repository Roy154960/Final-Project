"""
Embed step - via a local Ollama server (free, fully local, no API key).

Requires:
    ollama serve                      # running in the background
    ollama pull nomic-embed-text
    ollama pull mxbai-embed-large

Run directly to smoke-test:
    python -m embeddings.ollama_embedder
"""

import numpy as np
from embeddings.base import BaseEmbedder
from config import OLLAMA_HOST

try:
    import ollama as ollama_lib
except ImportError:
    ollama_lib = None

MODEL_DIMS = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
}


class OllamaEmbedder(BaseEmbedder):
    def __init__(self, model: str = "nomic-embed-text", host: str = OLLAMA_HOST):
        if ollama_lib is None:
            raise ImportError("Run: pip install ollama")
        if model not in MODEL_DIMS:
            raise ValueError(f"Unknown Ollama embed model '{model}'. Options: {list(MODEL_DIMS)}")
        self.name = f"ollama:{model}"
        self.model = model
        self.dimensions = MODEL_DIMS[model]
        self.client = ollama_lib.Client(host=host)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for t in texts:
            resp = self.client.embeddings(model=self.model, prompt=t)
            vectors.append(resp["embedding"])
        return np.array(vectors, dtype=np.float32)


if __name__ == "__main__":
    embedder = OllamaEmbedder("nomic-embed-text")
    vecs = embedder.embed_texts(["local RAG pipelines are fun to build"])
    print(f"{embedder.name}: shape={vecs.shape}")
