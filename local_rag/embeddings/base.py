"""
Common interface every embedder implements, so the rest of the pipeline
(store/retrieve) doesn't care whether embeddings came from Ollama, HF, or CLIP.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseEmbedder(ABC):
    name: str
    dimensions: int

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Return shape (len(texts), dimensions)."""
        ...

    def embed_images(self, image_paths: list[str]) -> np.ndarray:
        """Only implemented by multimodal embedders (e.g. CLIP)."""
        raise NotImplementedError(f"{self.name} does not support image embedding")

    def supports_images(self) -> bool:
        return False
