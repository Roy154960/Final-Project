"""
Chunk method 6: Parent-child chunking.

Small "child" chunks are embedded and stored for precise retrieval matching
(a narrow, specific chunk scores better against a narrow, specific query).
But at generation time, the LLM gets the larger "parent" chunk that
surrounds the matched child, so it has enough context to actually answer —
best of both worlds instead of choosing one chunk size for both jobs.

Design: parents are built first (bigger chunks, e.g. via recursive.py),
then each parent is split again into smaller children. Children carry a
`parent_id` in metadata; parents are kept in a separate lookup (not
embedded/stored in the vector index themselves) so retrieval always
searches over children, then resolves to parents before generation.
"""

from ingestion.schema import RawDocument, Chunk
from chunking.recursive import chunk_recursive

DEFAULT_PARENT_SIZE_TOKENS = 800
DEFAULT_PARENT_OVERLAP_TOKENS = 100
DEFAULT_CHILD_SIZE_TOKENS = 200
DEFAULT_CHILD_OVERLAP_TOKENS = 20


def build_parent_child_chunks(
    doc: RawDocument,
    parent_size: int = DEFAULT_PARENT_SIZE_TOKENS,
    parent_overlap: int = DEFAULT_PARENT_OVERLAP_TOKENS,
    child_size: int = DEFAULT_CHILD_SIZE_TOKENS,
    child_overlap: int = DEFAULT_CHILD_OVERLAP_TOKENS,
) -> tuple[list[Chunk], dict[str, Chunk]]:
    """
    Returns (children, parents_by_id).
      - children: what you embed and store in the vector index
      - parents_by_id: lookup table (parent_id -> parent Chunk), kept
        alongside the vector store (e.g. as a JSON file or a separate
        key-value table) so retrieval can resolve child -> parent at query time
    """
    parents = chunk_recursive(doc, chunk_size_tokens=parent_size, overlap_tokens=parent_overlap)

    children: list[Chunk] = []
    parents_by_id: dict[str, Chunk] = {}

    for parent in parents:
        parents_by_id[parent.chunk_id] = parent

        # Re-wrap the parent's text as a fake RawDocument so we can reuse
        # chunk_recursive for the child split too. Carry the parent's metadata
        # (which itself came from doc.metadata via the fixed chunk_recursive)
        # forward, or children lose filename/page and source attribution breaks.
        pseudo_doc = RawDocument.new(source_path=doc.source_path, modality="text", content=parent.text,
                                      **parent.metadata)
        child_chunks = chunk_recursive(pseudo_doc, chunk_size_tokens=child_size, overlap_tokens=child_overlap)

        for child in child_chunks:
            child.doc_id = doc.doc_id  # keep pointing at the real source doc, not the pseudo one
            child.metadata["parent_id"] = parent.chunk_id
            children.append(child)

    return children, parents_by_id


def resolve_to_parents(retrieved_children: list[dict], parents_by_id: dict[str, Chunk]) -> list[dict]:
    """
    Given retrieval results (each a dict with metadata containing
    'parent_id'), swap each child's text for its parent's larger text before
    handing off to generation. De-duplicates: if two retrieved children
    share the same parent, the parent is only included once.
    """
    seen_parent_ids = set()
    resolved = []
    for r in retrieved_children:
        parent_id = r.get("metadata", {}).get("parent_id")
        parent = parents_by_id.get(parent_id)
        if parent is None:
            resolved.append(r)  # fallback: no parent found, keep the child as-is
            continue
        if parent_id in seen_parent_ids:
            continue
        seen_parent_ids.add(parent_id)
        resolved.append({**r, "text": parent.text, "metadata": {**r.get("metadata", {}), **parent.metadata}})
    return resolved


if __name__ == "__main__":
    long_text = "Sentence about topic A. " * 100 + "Sentence about topic B. " * 100
    sample = RawDocument.new(source_path="sample.txt", modality="text", content=long_text)

    children, parents_by_id = build_parent_child_chunks(sample)
    print(f"{len(parents_by_id)} parent chunk(s), {len(children)} child chunk(s)")

    # Simulate retrieval returning 2 children from the same parent
    fake_retrieved = [
        {"id": children[0].chunk_id, "text": children[0].text, "metadata": children[0].metadata},
        {"id": children[1].chunk_id, "text": children[1].text, "metadata": children[1].metadata},
    ]
    resolved = resolve_to_parents(fake_retrieved, parents_by_id)
    print(f"resolved to {len(resolved)} parent(s) for generation (deduped if same parent)")
