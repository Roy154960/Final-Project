"""
Efficiency enhancement - embedding cache.

Re-embedding the same text (repeated boilerplate, re-running benchmarks
during development, incremental re-indexing hitting unchanged chunks that
still went through the embedder before the incremental-indexer check) costs
real time, especially for larger HF models on CPU. This wraps any embedder
with a local disk cache keyed by (model_name, text_hash), so identical
inputs are only embedded once, ever, across process restarts.
"""

import hashlib
import numpy as np
from diskcache import Cache

from embeddings.base import BaseEmbedder
from config import DATA_DIR

CACHE_DIR = DATA_DIR / "embedding_cache"


def _hash_text(model_name: str, text: str) -> str:
    return hashlib.sha256(f"{model_name}::{text}".encode()).hexdigest()


class CachedEmbedder(BaseEmbedder):
    """Wraps any BaseEmbedder with a transparent disk cache. Usage:

        embedder = CachedEmbedder(HFEmbedder("sentence-transformers/all-MiniLM-L6-v2"))
        vectors = embedder.embed_texts(texts)  # cache hits skip the model entirely
    """

    def __init__(self, inner: BaseEmbedder, cache_dir: str = None):
        self.inner = inner
        self.name = f"cached:{inner.name}"
        self.dimensions = inner.dimensions
        self._cache = Cache(str(cache_dir or CACHE_DIR))

    def supports_images(self) -> bool:
        return self.inner.supports_images()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        results: list[np.ndarray | None] = [None] * len(texts)
        # Dedupes repeats WITHIN this single batch too, not just across calls —
        # otherwise the same text appearing 2+ times in one embed_texts() call
        # would still hit the model repeatedly, since the disk cache isn't
        # written until after the whole batch finishes.
        indices_by_text: dict[str, list[int]] = {}
        for i, text in enumerate(texts):
            indices_by_text.setdefault(text, []).append(i)

        to_embed_texts = []
        for text, indices in indices_by_text.items():
            key = _hash_text(self.inner.name, text)
            cached = self._cache.get(key)
            if cached is not None:
                for i in indices:
                    results[i] = cached
            else:
                to_embed_texts.append(text)

        if to_embed_texts:
            fresh_vectors = self.inner.embed_texts(to_embed_texts)
            for text, vec in zip(to_embed_texts, fresh_vectors):
                key = _hash_text(self.inner.name, text)
                self._cache.set(key, vec)
                for i in indices_by_text[text]:
                    results[i] = vec

        return np.array(results, dtype=np.float32)

    def embed_images(self, image_paths: list[str]) -> np.ndarray:
        # Images are cached by file path + model name rather than content
        # hash, to avoid reading every image's bytes just to check the cache.
        # Swap to a content hash if your image files get overwritten in place.
        results: list[np.ndarray | None] = [None] * len(image_paths)
        to_embed_idx, to_embed_paths = [], []

        for i, path in enumerate(image_paths):
            key = _hash_text(self.inner.name, f"image::{path}")
            cached = self._cache.get(key)
            if cached is not None:
                results[i] = cached
            else:
                to_embed_idx.append(i)
                to_embed_paths.append(path)

        if to_embed_paths:
            fresh_vectors = self.inner.embed_images(to_embed_paths)
            for idx, path, vec in zip(to_embed_idx, to_embed_paths, fresh_vectors):
                key = _hash_text(self.inner.name, f"image::{path}")
                self._cache.set(key, vec)
                results[idx] = vec

        return np.array(results, dtype=np.float32)

    def cache_stats(self) -> dict:
        return {"cache_dir": str(self._cache.directory), "n_entries": len(self._cache)}


if __name__ == "__main__":
    # Smoke test with a fake inner embedder (no model download needed)
    class FakeEmbedder(BaseEmbedder):
        def __init__(self):
            self.name = "fake"
            self.dimensions = 4
            self.call_count = 0

        def embed_texts(self, texts):
            self.call_count += len(texts)
            return np.array([[hash(t) % 100 / 100] * 4 for t in texts], dtype=np.float32)

    fake = FakeEmbedder()
    cached = CachedEmbedder(fake, cache_dir="/tmp/local_rag_cache_test")

    texts = ["hello world", "goodbye world", "hello world"]  # 3rd is a repeat of 1st
    vecs = cached.embed_texts(texts)
    print(f"embedded {len(texts)} texts, inner embedder called {fake.call_count} time(s) (should be 2, not 3)")
    print(f"cache stats: {cached.cache_stats()}")
