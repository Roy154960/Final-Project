"""
Evaluation step - core metrics, no external dependencies.
For LLM-judged metrics (faithfulness, answer relevance), see ragas_eval.py.
"""


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(top_k)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(relevant_ids)


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Mean Reciprocal Rank of the FIRST relevant result."""
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def keyword_faithfulness_heuristic(answer: str, context_chunks: list[str]) -> float:
    """
    Cheap, model-free faithfulness proxy: what fraction of the answer's
    distinctive words also appear somewhere in the retrieved context.
    Not a substitute for LLM-judged faithfulness (see ragas_eval.py) but
    useful as a fast, free sanity check with zero extra inference cost.
    """
    context_words = set(" ".join(context_chunks).lower().split())
    answer_words = [w for w in answer.lower().split() if len(w) > 4]  # skip short/common words
    if not answer_words:
        return 1.0
    grounded = sum(1 for w in answer_words if w in context_words)
    return grounded / len(answer_words)


if __name__ == "__main__":
    retrieved = ["c1", "c2", "c3"]
    relevant = {"c2", "c4"}
    print(f"precision@3 = {precision_at_k(retrieved, relevant, 3)}")
    print(f"recall@3 = {recall_at_k(retrieved, relevant, 3)}")
    print(f"mrr = {mrr(retrieved, relevant)}")
