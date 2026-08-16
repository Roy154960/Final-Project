"""
Retrieve enhancement - contextual compression.

Retrieved chunks are sized for good retrieval (e.g. 200-800 tokens), not
for minimal noise at generation time — a chunk can be relevant overall
while only 1-2 sentences of it actually answer the question. This uses the
local LLM to extract just the relevant sentences from each chunk BEFORE
generation, trading one extra (cheap, small) LLM call per chunk for a
cleaner context window and often better-grounded final answers.

Cost note: this adds len(chunks) extra LLM calls per question. Worth it
when your reranked top-k chunks are still large/noisy; skip it for
already-tight, high-precision retrieval setups where it's not worth the
extra latency.
"""

COMPRESSION_PROMPT = """Given the question and a passage, extract ONLY the sentences from the passage \
that are directly relevant to answering the question. Return the relevant sentences verbatim, \
unchanged. If nothing in the passage is relevant, return exactly: NOT_RELEVANT

Question: {question}

Passage:
{passage}

Relevant sentences:"""


def compress_chunk(question: str, chunk_text: str, generator) -> str | None:
    """Returns the compressed text, or None if the chunk was judged irrelevant."""
    prompt = COMPRESSION_PROMPT.format(question=question, passage=chunk_text)

    if hasattr(generator, "client"):  # OllamaGenerator
        response = generator.client.chat(model=generator.model, messages=[{"role": "user", "content": prompt}])
        result = response["message"]["content"].strip()
    else:  # HFGenerator-style fallback
        result = generator.generate(prompt, retrieved_chunks=[]).strip()

    if result == "NOT_RELEVANT" or not result:
        return None
    return result


def compress_retrieved_chunks(question: str, chunks: list[dict], generator) -> list[dict]:
    """
    Applies compress_chunk to each retrieved chunk. Drops chunks the LLM
    judges irrelevant entirely (this can happen even after retrieval scored
    them highly — embedding similarity and "does this actually help answer
    the question" aren't always the same thing).
    """
    compressed = []
    for chunk in chunks:
        result = compress_chunk(question, chunk["text"], generator)
        if result is not None:
            compressed.append({**chunk, "text": result, "original_text": chunk["text"]})
    return compressed


if __name__ == "__main__":
    print("This module needs a live generator. See pipeline.py's --compress flag for a runnable example.")
