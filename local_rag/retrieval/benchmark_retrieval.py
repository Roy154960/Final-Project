"""
Compare retrieval strategies: vector-only vs hybrid vs hybrid+rerank vs
query-routed vs multi-query. This is the comparison harness for the
Retrieve step — every strategy implemented in retrieval/ should show up
here so choosing one over another is a measured decision, not a guess.

This needs a small labeled eval set: a handful of (query, relevant_chunk_ids)
pairs. Ship a few realistic examples from your own documents into
EVAL_SET below before trusting the numbers — the placeholder examples here
just prove the harness works end-to-end.

Metrics (see evaluation/metrics.py for implementations):
  - precision@k
  - recall@k
  - MRR (mean reciprocal rank of the first relevant result)

Run:
    python -m retrieval.benchmark_retrieval
"""

from evaluation.metrics import precision_at_k, recall_at_k, mrr
from evaluation.build_eval_set import load_eval_set

# Loads data/eval_set.json if you've built one with:
#   python -m evaluation.build_eval_set --interactive
# Falls back to a single placeholder example so this script still runs
# standalone — but treat any numbers from the placeholder as a smoke test,
# not a real signal. Build a real 20-50 example set before trusting this.
_real_eval_set = load_eval_set()
EVAL_SET = (
    [{"query": e["query"], "relevant_ids": set(e["relevant_ids"])} for e in _real_eval_set]
    if _real_eval_set
    else [{"query": "What is the capital of France?", "relevant_ids": {"chunk_paris"}}]
)


def _build_toy_index(embedder, store):
    """Populate the store with a tiny toy corpus so the harness is runnable
    standalone. Swap this for your real ingested + chunked + embedded corpus."""
    texts = [
        ("chunk_paris", "Paris is the capital and largest city of France."),
        ("chunk_berlin", "Berlin is the capital of Germany."),
        ("chunk_banana", "Bananas are rich in potassium and fiber."),
    ]
    ids = [t[0] for t in texts]
    contents = [t[1] for t in texts]
    vectors = embedder.embed_texts(contents)
    metadatas = [{"filename": "toy_corpus.txt"} for _ in ids]
    store.upsert(ids, vectors, contents, metadatas)
    return [{"id": i, "text": c, "metadata": m} for i, c, m in zip(ids, contents, metadatas)]


def benchmark(top_k: int = 3, include_multi_query: bool = True, generator=None):
    """
    include_multi_query: multi-query needs a live generator (LLM) to paraphrase
    the question, so it's the one strategy here that can fail if Ollama isn't
    running / no HF model is available. It's skipped (with a printed reason)
    rather than crashing the whole comparison — same defensive pattern as
    slm/benchmark_slms.py.
    """
    from embeddings.hf_embedder import HFEmbedder
    from vectorstore.chroma_store import ChromaStore
    from retrieval.vector_retriever import vector_retrieve
    from retrieval.hybrid_retriever import HybridRetriever
    from retrieval.reranker import Reranker
    from retrieval.query_router import rule_based_route, route_and_retrieve

    embedder = HFEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    store = ChromaStore(collection_name="retrieval_benchmark_toy")
    corpus = _build_toy_index(embedder, store)
    hybrid = HybridRetriever(embedder, store, corpus)
    reranker = Reranker()

    def _router_strategy(q):
        decision_preview = rule_based_route(q)
        corpus_for_hybrid = corpus if decision_preview.route == "keyword_hybrid" else None
        results, _decision = route_and_retrieve(q, embedder, store,
                                                 corpus_for_hybrid=corpus_for_hybrid, top_k=top_k)
        return results

    strategies = {
        "vector_only": lambda q: vector_retrieve(q, embedder, store, top_k=top_k),
        "hybrid": lambda q: hybrid.retrieve(q, top_k=top_k),
        "hybrid_plus_rerank": lambda q: reranker.rerank(q, hybrid.retrieve(q, top_k=top_k * 2), top_k=top_k),
        "router": _router_strategy,
    }

    if include_multi_query:
        try:
            from retrieval.multi_query import multi_query_retrieve
            gen = generator
            if gen is None:
                from generation.ollama_generator import OllamaGenerator
                gen = OllamaGenerator()  # raises/errors below if Ollama isn't reachable
            # Fail fast on a cheap call so a dead Ollama server shows up as one
            # [skip] line, not a stack trace per eval-set question further down.
            gen.generate("connectivity check", [])
            strategies["multi_query"] = lambda q: multi_query_retrieve(q, embedder, store, gen, top_k_final=top_k)
        except Exception as e:
            print(f"[skip] multi_query: needs a live generator (Ollama/HF) — {e}")

    print(f"{'strategy':<22}{'precision@k':<14}{'recall@k':<12}{'mrr':<8}")
    for name, fn in strategies.items():
        try:
            precisions, recalls, ranks = [], [], []
            for example in EVAL_SET:
                results = fn(example["query"])
                retrieved_ids = [r["id"] for r in results]
                precisions.append(precision_at_k(retrieved_ids, example["relevant_ids"], top_k))
                recalls.append(recall_at_k(retrieved_ids, example["relevant_ids"], top_k))
                ranks.append(mrr(retrieved_ids, example["relevant_ids"]))
            avg = lambda lst: round(sum(lst) / len(lst), 3) if lst else 0.0
            print(f"{name:<22}{avg(precisions):<14}{avg(recalls):<12}{avg(ranks):<8}")
        except Exception as e:
            print(f"[skip] {name}: {e}")


if __name__ == "__main__":
    benchmark()
