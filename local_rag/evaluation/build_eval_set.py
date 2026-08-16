"""
Evaluation enhancement - build a REAL labeled eval set.

The single highest-leverage thing you can do for this project: the
benchmarks in chunking/, embeddings/, vectorstore/, and retrieval/ are
only as trustworthy as the eval data behind them. This module gives you:

  1. A guided CLI to build one interactively against your own indexed corpus
  2. An auto-labeling mode (--auto-label) that keeps YOUR real questions but
     uses a local LLM to judge which retrieved passages are relevant,
     instead of you doing it by hand for every question
  3. A validated schema so retrieval/benchmark_retrieval.py can load it directly
  4. A stats summary so you know when you have "enough" for the numbers to
     stop being noisy (rule of thumb: 20-50 examples covering your real
     question types, per your existing project notes)

On auto-labeling and bias: a good eval set should reflect what YOU, the
actual user, would ask — that's why --auto-label still requires you to
write the actual questions yourself (in a plain text file, one per line);
it only automates the tedious "which of these 8 passages counts" judgment,
not the questions themselves. Even so, an LLM judge has its own blind
spots and will not perfectly match your own judgment — treat auto-labeled
sets as good for fast iteration during development, and spot-check a
sample by hand (re-run a few of the same questions with --interactive and
compare) before quoting numbers built entirely from auto-labels in a final
report.

Workflow (manual):
    1. Ingest your real documents (pipeline.py ingest ...)
    2. Run this in --interactive mode: it shows you retrieval results for a
       question you type, and you mark which ones are actually relevant
    3. Repeat for 20-50 realistic questions
    4. Point retrieval/benchmark_retrieval.py's EVAL_SET loader at the
       resulting file

Workflow (auto-labeled):
    1. Ingest your real documents
    2. Write your real questions into a text file, one per line
    3. Run this in --auto-label mode, pointing at that file and a local
       Ollama model to use as judge
    4. Same resulting file, same downstream usage as the manual workflow

Run:
    python -m evaluation.build_eval_set --interactive --embedder hf --store chroma
    python -m evaluation.build_eval_set --interactive --embedder hf --store chroma \
        --collection "parent_child__hf_sentence-transformers_all-MiniLM-L6-v2"
    python -m evaluation.build_eval_set --auto-label questions.txt --embedder hf --store chroma \
        --collection "parent_child__hf_sentence-transformers_all-MiniLM-L6-v2" --judge-model llama3.2
    python -m evaluation.build_eval_set --stats data/eval_set.json
"""

import argparse
import json
import re
from pathlib import Path

from config import DATA_DIR, OLLAMA_HOST

DEFAULT_EVAL_SET_PATH = DATA_DIR / "eval_set.json"

JUDGE_SYSTEM_PROMPT = (
    "You are judging search results for a retrieval system — you are NOT "
    "answering the question yourself. You will be given a question and a "
    "numbered list of retrieved text passages (some may be OCR'd, with "
    "broken words or formatting — judge the content, not the formatting). "
    "Identify ONLY the passages that directly contain information that "
    "answers the question. Respond with nothing but a comma-separated list "
    "of passage numbers (e.g. '0,2,5'), or the single word 'none' if no "
    "passage is relevant. No explanation, no other words."
)


def load_eval_set(path: str = None) -> list[dict]:
    p = Path(path or DEFAULT_EVAL_SET_PATH)
    if not p.exists():
        return []
    return json.loads(p.read_text())


def save_eval_set(eval_set: list[dict], path: str = None) -> None:
    p = Path(path or DEFAULT_EVAL_SET_PATH)
    p.write_text(json.dumps(eval_set, indent=2))


def print_stats(eval_set: list[dict]) -> None:
    n = len(eval_set)
    print(f"{n} labeled example(s)")
    if n < 20:
        print(f"  -> below the recommended 20-50 range; benchmark numbers will be noisy until you add more.")
    avg_relevant = sum(len(e["relevant_ids"]) for e in eval_set) / n if n else 0
    print(f"  avg relevant chunks per question: {avg_relevant:.1f}")


def _resolve_store(store_name: str, dimensions: int, collection: str = None):
    # get_store() always opens the default collection (config.CHROMA_COLLECTION /
    # QDRANT_COLLECTION) with no way to override it — which is exactly the
    # collection load_step.py's per-method/embedder experiments do NOT write
    # to (it names collections "<chunk_method>__<embedder_key>" on purpose,
    # so comparisons don't collide). Build against one of THOSE collections
    # directly here instead, rather than requiring a duplicate `store` run
    # into the default collection just to label an eval set.
    if collection:
        if store_name == "chroma":
            from vectorstore.chroma_store import ChromaStore
            return ChromaStore(collection_name=collection)
        elif store_name == "qdrant":
            from vectorstore.qdrant_store import QdrantStore
            return QdrantStore(collection_name=collection, dimensions=dimensions)
        raise ValueError(f"Unknown store: {store_name}")
    from pipeline import get_store
    return get_store(store_name, dimensions)


def _retrieve_candidates(question: str, embedder, store, retrieval: str, top_k: int) -> list[dict]:
    if retrieval == "vector":
        from retrieval.vector_retriever import vector_retrieve
        return vector_retrieve(question, embedder, store, top_k=top_k)
    if retrieval == "hybrid":
        from retrieval.hybrid_retriever import hybrid_retrieve
        return hybrid_retrieve(question, embedder, store, top_k=top_k)
    if retrieval == "router":
        from retrieval.query_router import rule_based_route, route_and_retrieve
        decision_preview = rule_based_route(question)
        corpus = store.get_all() if decision_preview.route == "keyword_hybrid" else None
        results, _decision = route_and_retrieve(question, embedder, store,
                                                 corpus_for_hybrid=corpus, top_k=top_k)
        return results
    raise ValueError(f"Unknown retrieval strategy: {retrieval}")


def interactive_build(embedder_name: str, store_name: str, top_k: int = 8, out_path: str = None,
                       collection: str = None, retrieval: str = "vector"):
    from pipeline import get_embedder

    embedder = get_embedder(embedder_name)
    store = _resolve_store(store_name, embedder.dimensions, collection)
    eval_set = load_eval_set(out_path)

    print("Building a labeled retrieval eval set. Type a question, then mark which "
          "retrieved chunks are actually relevant. Type 'done' to stop and save.\n")

    while True:
        question = input("Question (or 'done'): ").strip()
        if question.lower() == "done":
            break
        if not question:
            continue

        results = _retrieve_candidates(question, embedder, store, retrieval, top_k)
        if not results:
            print("  No results retrieved — index some documents first.")
            continue

        for i, r in enumerate(results):
            print(f"  [{i}] ({r['score']:.3f}) {r['text'][:120]}")

        raw = input("  Relevant indices (comma-separated, or blank for none): ").strip()
        relevant_indices = [int(x) for x in raw.split(",") if x.strip().isdigit()]
        relevant_ids = [results[i]["id"] for i in relevant_indices]

        eval_set.append({"query": question, "relevant_ids": relevant_ids})
        save_eval_set(eval_set, out_path)  # save after every example so nothing is lost
        print(f"  saved ({len(eval_set)} total)\n")

    print_stats(eval_set)


def _judge_relevance(question: str, results: list[dict], judge_model: str, host: str = None) -> list[int]:
    """Ask a local Ollama model which of the retrieved passages actually answer
    the question. Returns a list of passage indices (into `results`), same
    shape as what a human would type at the interactive prompt."""
    try:
        import ollama as ollama_lib
    except ImportError:
        raise ImportError("Run: pip install ollama")

    client = ollama_lib.Client(host=host or OLLAMA_HOST)

    lines = [f"Question: {question}", "", "Passages:"]
    for i, r in enumerate(results):
        lines.append(f"[{i}] {r['text'][:400]}")
    lines.append("")
    lines.append("Relevant passage numbers (comma-separated, or 'none'):")
    prompt = "\n".join(lines)

    response = client.chat(
        model=judge_model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    raw = response["message"]["content"].strip().lower()

    if "none" in raw and not re.search(r"\d", raw):
        return []
    indices = []
    for tok in re.findall(r"\d+", raw):
        idx = int(tok)
        if 0 <= idx < len(results) and idx not in indices:
            indices.append(idx)
    return indices


def auto_label_from_file(questions_path: str, embedder_name: str = "hf", store_name: str = "chroma",
                          top_k: int = 8, out_path: str = None, collection: str = None,
                          judge_model: str = "llama3.2", retrieval: str = "vector"):
    """
    Same output file and schema as interactive_build(), but the questions come
    from a plain text file (one per line, blank lines skipped) instead of
    stdin, and relevance judging is done by a local LLM instead of you typing
    indices by hand. See the module docstring for the bias tradeoff this
    involves before trusting these labels for a final report without spot-checking.
    """
    from pipeline import get_embedder

    questions = [line.strip() for line in Path(questions_path).read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    if not questions:
        print(f"No questions found in {questions_path} — one question per line, please.")
        return

    embedder = get_embedder(embedder_name)
    store = _resolve_store(store_name, embedder.dimensions, collection)
    eval_set = load_eval_set(out_path)

    print(f"Auto-labeling {len(questions)} question(s) from {questions_path}, "
          f"judged by Ollama model '{judge_model}'.\n"
          f"These are LLM-judged labels, not human-verified — see this module's docstring "
          f"before trusting them for a final report without spot-checking a sample.\n")

    for i, question in enumerate(questions):
        results = _retrieve_candidates(question, embedder, store, retrieval, top_k)
        if not results:
            print(f"[{i + 1}/{len(questions)}] {question!r} -> no results retrieved, skipping")
            continue

        try:
            relevant_indices = _judge_relevance(question, results, judge_model)
        except Exception as e:
            print(f"[{i + 1}/{len(questions)}] {question!r} -> judge failed ({e}), skipping")
            continue

        relevant_ids = [results[idx]["id"] for idx in relevant_indices]
        eval_set.append({"query": question, "relevant_ids": relevant_ids})
        save_eval_set(eval_set, out_path)  # save after every example so nothing is lost

        preview = ", ".join(f"[{idx}]" for idx in relevant_indices) or "none"
        print(f"[{i + 1}/{len(questions)}] {question!r} -> relevant: {preview} "
              f"({len(relevant_ids)}/{len(results)} passages)")

    print()
    print_stats(eval_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--auto-label", dest="auto_label", metavar="QUESTIONS_FILE",
                         help="Path to a text file of questions (one per line) to auto-label "
                              "using a local LLM as judge, instead of the interactive human flow.")
    parser.add_argument("--judge-model", dest="judge_model", default="llama3.2",
                         help="Ollama model to use as the relevance judge with --auto-label "
                              "(default: llama3.2). Needs `ollama serve` running.")
    parser.add_argument("--retrieval", choices=["vector", "hybrid", "router"], default="vector",
                         help="Retrieval strategy used to fetch candidates to label (default: vector)")
    parser.add_argument("--stats", metavar="PATH", help="Print stats for an existing eval set file")
    parser.add_argument("--embedder", choices=["hf", "ollama", "clip"], default="hf")
    parser.add_argument("--store", choices=["chroma", "qdrant"], default="chroma")
    parser.add_argument("--top-k", dest="top_k", type=int, default=8)
    parser.add_argument("--out", default=None)
    parser.add_argument("--collection", default=None,
                         help="Target a specific vector store collection by name (e.g. a "
                              "load_step.py combo like 'parent_child__hf_sentence-transformers_"
                              "all-MiniLM-L6-v2') instead of the default collection.")
    args = parser.parse_args()

    if args.stats:
        print_stats(load_eval_set(args.stats))
    elif args.auto_label:
        auto_label_from_file(args.auto_label, args.embedder, args.store, args.top_k, args.out,
                              args.collection, args.judge_model, args.retrieval)
    elif args.interactive:
        interactive_build(args.embedder, args.store, args.top_k, args.out, args.collection, args.retrieval)
    else:
        print("Pass --interactive to build a set by hand, --auto-label QUESTIONS_FILE to "
              "auto-label from a local LLM judge, or --stats PATH to inspect an existing one.")
