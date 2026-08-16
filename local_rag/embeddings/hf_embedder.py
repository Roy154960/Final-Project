"""
Embed step - via local Hugging Face weights using sentence-transformers
(free, downloaded once then cached locally, no API key).

Run directly to smoke-test:
    python -m embeddings.hf_embedder
"""

import numpy as np
from embeddings.base import BaseEmbedder

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

MODEL_DIMS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
}


class HFEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise ImportError("Run: pip install sentence-transformers")
        self.name = f"hf:{model_name}"
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dimensions = MODEL_DIMS.get(model_name, self._model.get_sentence_embedding_dimension())

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        # BGE models recommend a query prefix for asymmetric search; harmless for
        # symmetric use cases too, so we skip it here to keep behavior uniform
        # across models. Add "query: " prefixing in retrieval/ if you need the
        # extra BGE retrieval boost.
        return self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


if __name__ == "__main__":
    embedder = HFEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    vecs = embedder.embed_texts(["local RAG pipelines are fun to build"])
    print(f"{embedder.name}: shape={vecs.shape}")
