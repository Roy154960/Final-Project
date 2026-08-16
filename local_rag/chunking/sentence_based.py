"""
Chunk method 3: Sentence-based splitting.
Groups N sentences per chunk. Good for precise retrieval where you don't
want a chunk to straddle unrelated ideas mid-sentence.
"""

import nltk
from ingestion.schema import RawDocument, Chunk

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def chunk_sentence_based(doc: RawDocument, sentences_per_chunk: int = 4) -> list[Chunk]:
    sentences = nltk.sent_tokenize(doc.content)
    if not sentences:
        return []

    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        group = sentences[i : i + sentences_per_chunk]
        text = " ".join(group)
        # Carry doc-level metadata (filename, page, ...) forward so source
        # attribution works at generation time.
        chunks.append(Chunk.new(doc_id=doc.doc_id, text=text, **{**doc.metadata, "method": "sentence_based"}))
    return chunks


if __name__ == "__main__":
    sample_text = "This is sentence one. This is sentence two. This is sentence three. " * 20
    sample = RawDocument.new(source_path="sample", modality="text", content=sample_text)
    result = chunk_sentence_based(sample)
    print(f"{len(result)} chunks")
