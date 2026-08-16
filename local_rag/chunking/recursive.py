"""
Chunk method 2: Recursive splitting.
Tries paragraph breaks first, then sentences, then words, then raw
characters, so chunks respect natural text boundaries wherever possible.
Uses langchain-text-splitters (free, local, no API calls).
"""

from ingestion.schema import RawDocument, Chunk
from config import DEFAULT_CHUNK_SIZE_TOKENS, DEFAULT_CHUNK_OVERLAP_TOKENS

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Rough chars-per-token heuristic (~4 chars/token for English) so the
# chunk_size configured in tokens elsewhere stays comparable across methods.
CHARS_PER_TOKEN = 4


def chunk_recursive(
    doc: RawDocument,
    chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS,
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_tokens * CHARS_PER_TOKEN,
        chunk_overlap=overlap_tokens * CHARS_PER_TOKEN,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    pieces = splitter.split_text(doc.content)
    # Carry doc-level metadata (filename, page, extraction_method, ...) forward
    # onto every chunk — without this, source attribution at generation time
    # falls back to "unknown source" for every chunk from this method.
    return [Chunk.new(doc_id=doc.doc_id, text=t, **{**doc.metadata, "method": "recursive"}) for t in pieces]


if __name__ == "__main__":
    sample_text = "Paragraph one.\n\nParagraph two has more content. " * 50
    sample = RawDocument.new(source_path="sample", modality="text", content=sample_text)
    result = chunk_recursive(sample)
    print(f"{len(result)} chunks, avg chars: {sum(len(c.text) for c in result) // max(len(result),1)}")
