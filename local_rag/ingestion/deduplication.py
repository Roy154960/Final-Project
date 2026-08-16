"""
Ingest enhancement - deduplication.

Near-identical chunks (repeated boilerplate, headers/footers, copy-pasted
sections across documents) waste storage and can dominate retrieval —
if 5 near-identical chunks all rank highly for a query, they crowd out
genuinely different relevant content in your top-k.

Two layers, cheapest first:
  1. Exact-duplicate removal via content hash (near-zero cost)
  2. Near-duplicate removal via embedding cosine similarity above a
     threshold (needs an embedder, catches paraphrased/reformatted repeats
     that hashing misses)
"""

import hashlib
import numpy as np

from ingestion.schema import Chunk


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()


def remove_exact_duplicates(chunks: list[Chunk]) -> list[Chunk]:
    seen = set()
    deduped = []
    for c in chunks:
        h = _hash_text(c.text)
        if h not in seen:
            seen.add(h)
            deduped.append(c)
    return deduped


def remove_near_duplicates(
    chunks: list[Chunk],
    embed_fn,  # callable: list[str] -> np.ndarray, e.g. an embeddings/ model's embed_texts
    similarity_threshold: float = 0.97,
) -> list[Chunk]:
    """
    Greedy near-duplicate removal: keeps the first occurrence of each
    "cluster" of highly similar chunks, drops the rest. O(n^2) similarity
    comparisons — fine for a few thousand chunks per ingest batch; for very
    large corpora, batch this or use an approximate nearest-neighbor index
    instead of the naive pairwise loop below.
    """
    if len(chunks) <= 1:
        return chunks

    vectors = embed_fn([c.text for c in chunks])
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.clip(norms, 1e-8, None)
    sim_matrix = normalized @ normalized.T

    keep = [True] * len(chunks)
    for i in range(len(chunks)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(chunks)):
            if keep[j] and sim_matrix[i, j] >= similarity_threshold:
                keep[j] = False

    return [c for c, k in zip(chunks, keep) if k]


def deduplicate(chunks: list[Chunk], embed_fn=None, similarity_threshold: float = 0.97) -> list[Chunk]:
    """Convenience wrapper: exact-dedup always, near-dedup only if an
    embed_fn is provided (it costs an embedding pass over every chunk)."""
    exact_deduped = remove_exact_duplicates(chunks)
    if embed_fn is None:
        return exact_deduped
    return remove_near_duplicates(exact_deduped, embed_fn, similarity_threshold)


if __name__ == "__main__":
    sample_chunks = [
        Chunk.new(doc_id="d1", text="The capital of France is Paris."),
        Chunk.new(doc_id="d1", text="The capital of France is Paris."),  # exact dup
        Chunk.new(doc_id="d2", text="Paris is the capital city of France."),  # near dup
        Chunk.new(doc_id="d3", text="Bananas are rich in potassium."),
    ]

    def fake_embed_fn(texts):
        # Toy embedder for the smoke test: bag-of-words counts over a fixed
        # vocabulary so "Paris"-ish sentences land close together in vector
        # space. Real usage passes embeddings/hf_embedder.py's embed_texts.
        vocab = ["capital", "france", "paris", "banana", "potassium"]
        vecs = []
        for t in texts:
            t_lower = t.lower()
            vecs.append([1.0 if w in t_lower else 0.0 for w in vocab])
        return np.array(vecs)

    print(f"before: {len(sample_chunks)} chunks")
    after_exact = remove_exact_duplicates(sample_chunks)
    print(f"after exact dedup: {len(after_exact)} chunks")
    after_near = deduplicate(sample_chunks, embed_fn=fake_embed_fn, similarity_threshold=0.65)
    print(f"after near dedup: {len(after_near)} chunks")
