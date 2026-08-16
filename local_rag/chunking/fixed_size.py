"""
Chunk method 1: Fixed-size chunking.
Split every N tokens (approximated by whitespace-split words here to avoid
a hard dependency on a specific tokenizer) with a sliding overlap.
Simple, fast, works surprisingly well as a baseline.
"""

from ingestion.schema import RawDocument, Chunk
from config import DEFAULT_CHUNK_SIZE_TOKENS, DEFAULT_CHUNK_OVERLAP_TOKENS


def chunk_fixed_size(
    doc: RawDocument,
    chunk_size: int = DEFAULT_CHUNK_SIZE_TOKENS,
    overlap: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
) -> list[Chunk]:
    words = doc.content.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        text = " ".join(words[start:end])
        # Carry doc-level metadata (filename, page, extraction_method, ...) forward
        # onto every chunk — without this, source attribution at generation time
        # falls back to "unknown source" for every chunk from this method.
        chunks.append(Chunk.new(doc_id=doc.doc_id, text=text, **{**doc.metadata, "method": "fixed_size"}))
        if end == len(words):
            break
        start = end - overlap  # slide back by overlap
    return chunks


if __name__ == "__main__":
    sample = RawDocument.new(source_path="sample", modality="text", content=" ".join(["word"] * 1200))
    result = chunk_fixed_size(sample)
    print(f"{len(result)} chunks, sizes: {[len(c.text.split()) for c in result]}")
