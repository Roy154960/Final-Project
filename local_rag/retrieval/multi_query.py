"""
Retrieve enhancement - multi-query (query expansion).

A single embedding of the user's exact phrasing can miss chunks that
express the same idea differently. This asks the local LLM to generate a
few paraphrases/related framings of the question, retrieves for each
independently, then fuses the result lists with Reciprocal Rank Fusion —
the same fusion technique used in hybrid_retriever.py.
"""

import re

from embeddings.base import BaseEmbedder
from vectorstore.base import BaseVectorStore

MULTI_QUERY_PROMPT = """Generate {n} different ways to ask the following question. \
Each version should use different words/phrasing but ask for the same information. \
Return ONLY the {n} questions, one per line, no numbering, no extra commentary.

Original question: {question}"""


def generate_query_variants(question: str, generator, n: int = 3) -> list[str]:
    """
    generator: any object with a `.generate(question, retrieved_chunks)`-style
    interface won't fit here directly — instead we call the underlying LLM
    with a plain prompt. Pass an OllamaGenerator or HFGenerator instance;
    this function reaches into its raw chat/generate call.
    """
    prompt = MULTI_QUERY_PROMPT.format(n=n, question=question)

    if hasattr(generator, "client"):  # OllamaGenerator
        response = generator.client.chat(model=generator.model, messages=[{"role": "user", "content": prompt}])
        raw = response["message"]["content"]
    else:  # HFGenerator or similar — reuse .generate with no context, prompt as "question"
        raw = generator.generate(prompt, retrieved_chunks=[])

    lines = [ln.strip() for ln in raw.strip().split("\n") if ln.strip()]
    # Strip stray numbering like "1." or "-" the LLM sometimes adds despite instructions
    variants = [re.sub(r"^[\d\.\-\)\s]+", "", ln) for ln in lines]
    variants = [v for v in variants if v][:n]

    return [question] + variants  # always include the original


def multi_query_retrieve(
    question: str,
    embedder: BaseEmbedder,
    store: BaseVectorStore,
    generator,
    n_variants: int = 3,
    top_k_per_query: int = 10,
    top_k_final: int = 5,
    rrf_k: int = 60,
) -> list[dict]:
    queries = generate_query_variants(question, generator, n=n_variants)

    all_ranked_lists = []
    for q in queries:
        vec = embedder.embed_texts([q])[0]
        results = store.query(vec, top_k=top_k_per_query)
        all_ranked_lists.append(results)

    # Reciprocal Rank Fusion across all query variants' result lists
    fused_scores: dict[str, float] = {}
    text_by_id: dict[str, str] = {}
    meta_by_id: dict[str, dict] = {}

    for ranked_list in all_ranked_lists:
        for rank, r in enumerate(ranked_list):
            fused_scores[r["id"]] = fused_scores.get(r["id"], 0.0) + 1.0 / (rrf_k + rank)
            text_by_id[r["id"]] = r["text"]
            meta_by_id[r["id"]] = r.get("metadata", {})

    fused = [
        {"id": doc_id, "text": text_by_id[doc_id], "score": score, "metadata": meta_by_id[doc_id]}
        for doc_id, score in fused_scores.items()
    ]
    fused.sort(key=lambda r: -r["score"])
    return fused[:top_k_final]


if __name__ == "__main__":
    print("This module needs a live embedder + store + generator. "
          "See pipeline.py's --retrieval multi_query flag for a runnable end-to-end example.")
