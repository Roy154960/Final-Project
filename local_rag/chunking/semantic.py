"""
Chunk method 4: Semantic chunking.
Embed each sentence, then split where cosine similarity between
consecutive sentences drops below a threshold (a topic shift).
Most intelligent method, but slowest since it requires an embedding
model pass at chunk time.
"""

import nltk
import numpy as np
from ingestion.schema import RawDocument, Chunk

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def chunk_semantic(
    doc: RawDocument,
    embed_fn,  # callable: list[str] -> np.ndarray of shape (n, dim). Pass any embedder from embeddings/.
    similarity_threshold: float = 0.75,
    max_sentences_per_chunk: int = 10,
) -> list[Chunk]:
    sentences = nltk.sent_tokenize(doc.content)
    if not sentences:
        return []
    # Carry doc-level metadata (filename, page, ...) forward onto every chunk
    # so source attribution works at generation time.
    chunk_meta = {**doc.metadata, "method": "semantic"}

    if len(sentences) == 1:
        return [Chunk.new(doc_id=doc.doc_id, text=sentences[0], **chunk_meta)]

    embeddings = embed_fn(sentences)

    chunks = []
    current = [sentences[0]]
    for i in range(1, len(sentences)):
        sim = _cosine_sim(embeddings[i - 1], embeddings[i])
        if sim < similarity_threshold or len(current) >= max_sentences_per_chunk:
            chunks.append(Chunk.new(doc_id=doc.doc_id, text=" ".join(current), **chunk_meta))
            current = [sentences[i]]
        else:
            current.append(sentences[i])
    if current:
        chunks.append(Chunk.new(doc_id=doc.doc_id, text=" ".join(current), **chunk_meta))
    return chunks


if __name__ == "__main__":
    # Smoke test with a fake embed_fn (real usage passes an embeddings/ model)
    def fake_embed_fn(sentences):
        rng = np.random.default_rng(0)
        return rng.normal(size=(len(sentences), 32))

    sample_text = "Topic A sentence one. Topic A sentence two. Topic B begins here. Topic B continues."
    sample = RawDocument.new(source_path="sample", modality="text", content=sample_text)
    result = chunk_semantic(sample, embed_fn=fake_embed_fn)
    print(f"{len(result)} chunks (fake embeddings, real runs should use embeddings/hf_embedder.py)")
