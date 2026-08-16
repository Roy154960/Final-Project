"""
load_step.py — run the RAG pipeline ONE STAGE AT A TIME.

The key property: every stage reads its input EXCLUSIVELY from a checkpoint
file under data/checkpoints/ — never from a variable held over from a
previous step in the same Python process. That means each command below is
independent of whichever ones ran before it in this session:

    python load_step.py ingest   --source data/raw
    python load_step.py chunk
    python load_step.py embed    --chunk-method recursive
    python load_step.py store    --chunk-method recursive --embedder "hf:sentence-transformers/all-MiniLM-L6-v2"
    python load_step.py retrieve --chunk-method recursive --embedder "hf:sentence-transformers/all-MiniLM-L6-v2" --store chroma
    python load_step.py generate --chunk-method recursive --embedder "hf:sentence-transformers/all-MiniLM-L6-v2" --store chroma --strategy hybrid
    python load_step.py status   # what's been run so far, and with what results

You can run `embed` today, close the terminal, and run `store` next week in
a totally fresh process — as long as data/checkpoints/step3_embeddings__*
exists, `store` doesn't care how or when it was produced. Persistence is
plain JSON + .npy on disk (via ingestion/cache.py for step 1, and the same
pattern extended here for the rest) plus Chroma/Qdrant's own on-disk
storage for step 4 — nothing lives only in memory between steps.

Every stage runs ALL applicable methods (not just one) and evaluates each
with the same metrics its corresponding benchmark_*.py already established
in this project (chunking/benchmark_chunkers.py, embeddings/benchmark_embedders.py,
vectorstore/benchmark_stores.py, retrieval/benchmark_retrieval.py,
slm/benchmark_slms.py) — this file's job is chaining those comparisons
together with disk checkpoints, not reinventing the metrics. A candidate
that errors (model not pulled, server not running, etc.) is skipped with a
printed reason, never crashes the whole stage — same defensive pattern
already used throughout this project.

Run everything in order, still checkpointing each stage as it goes
(a crash on stage 4 doesn't lose stages 1-3's results):
    python load_step.py all --source data/raw

Checkpoint files (data/checkpoints/):
    step1_raw_docs.json                                        (via ingestion/cache.py)
    step1_metrics.json
    step2_chunks__<method>.json                                 one per chunking method
    step2_parents__<method>.json                                only for parent_child
    step2_metrics.json
    step3_embeddings__<chunk_method>__<embedder_key>.npy/.json  one per (chunk_method, embedder) pair actually run
    step3_metrics__<chunk_method>.json
    step4_ref__<store>__<chunk_method>__<embedder_key>.json     pointer to the persisted Chroma/Qdrant collection
    step4_metrics__<chunk_method>__<embedder_key>.json
    step5_retrieval__<store>__<chunk_method>__<embedder_key>__<strategy>.json
    step5_metrics__<store>__<chunk_method>__<embedder_key>.json
    step6_generation__<generator_key>.json
    step6_metrics.json
"""

import argparse
import json
import time
import statistics
from dataclasses import asdict
from pathlib import Path

import numpy as np

from config import DATA_DIR, HF_EMBED_MODELS, OLLAMA_EMBED_MODELS, OLLAMA_GENERATION_MODELS, HF_GENERATION_MODELS
from ingestion.schema import Chunk
from evaluation.metrics import precision_at_k, recall_at_k, mrr, keyword_faithfulness_heuristic
from evaluation.build_eval_set import load_eval_set

CKPT_DIR = DATA_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_METHODS = ["fixed_size", "recursive", "sentence_based", "semantic", "structure_aware", "parent_child"]
EMBEDDER_CANDIDATES = [("hf", m) for m in HF_EMBED_MODELS] + [("ollama", m) for m in OLLAMA_EMBED_MODELS]
STORE_CANDIDATES = ["chroma", "qdrant"]
RETRIEVAL_STRATEGIES = ["vector", "hybrid", "router", "multi_query"]
GENERATOR_CANDIDATES = [("ollama", m) for m in OLLAMA_GENERATION_MODELS] + [("hf", m) for m in HF_GENERATION_MODELS]

SANITY_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast, dark-colored fox leaps above a sleepy dog.",   # paraphrase of the above
    "Quarterly revenue grew by twelve percent year over year.",  # unrelated
]


# ---------------------------------------------------------------------------
# Checkpoint I/O helpers
# ---------------------------------------------------------------------------
def _safe_name(s: str) -> str:
    return s.replace("/", "_").replace(":", "_").replace(" ", "_")


def _save_json(path, data) -> None:
    Path(path).write_text(json.dumps(data, indent=2))


def _load_json(path) -> dict:
    return json.loads(Path(path).read_text())


def _save_embedding_bundle(prefix: Path, ids, vectors, texts, metadatas) -> None:
    np.save(str(prefix) + ".npy", np.asarray(vectors))
    _save_json(str(prefix) + ".json", {"ids": ids, "texts": texts, "metadatas": metadatas})


def _load_embedding_bundle(prefix: Path):
    npy_path = Path(str(prefix) + ".npy")
    if not npy_path.exists():
        raise FileNotFoundError(
            f"No step3 checkpoint at {npy_path} — run:\n"
            f"    python load_step.py embed --chunk-method <method> --embedders <backend:model>"
        )
    vectors = np.load(str(npy_path))
    side = _load_json(str(prefix) + ".json")
    return side["ids"], vectors, side["texts"], side["metadatas"]


def _cosine(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _resolve_embedder(embedder_key: str):
    """embedder_key is exactly embedder.name, e.g. 'hf:sentence-transformers/all-MiniLM-L6-v2'."""
    backend, model = embedder_key.split(":", 1)
    if backend == "hf":
        from embeddings.hf_embedder import HFEmbedder
        return HFEmbedder(model)
    if backend == "ollama":
        from embeddings.ollama_embedder import OllamaEmbedder
        return OllamaEmbedder(model)
    raise ValueError(f"Unknown embedder backend in key {embedder_key!r}")


# ---------------------------------------------------------------------------
# Step 1: Ingest — parse the raw source folder once, cache to disk
# ---------------------------------------------------------------------------
STEP1_PATH = CKPT_DIR / "step1_raw_docs.json"


def step_ingest(source: str = "data/raw"):
    from ingestion.cache import save_raw_documents
    from ingestion.loader import ingest_directory

    print(f"[ingest] parsing {source} (the slow part — OCR, PDF parsing, etc.) ...")
    docs = ingest_directory(source)
    save_raw_documents(docs, str(STEP1_PATH))

    by_modality: dict = {}
    for d in docs:
        by_modality[d.modality] = by_modality.get(d.modality, 0) + 1
    metrics = {"n_docs": len(docs), "by_modality": by_modality}
    _save_json(CKPT_DIR / "step1_metrics.json", metrics)
    print(f"[ingest] done: {metrics}\n  -> {STEP1_PATH}")
    return metrics


# ---------------------------------------------------------------------------
# Step 2: Chunk — load persisted raw docs, run ALL chunking methods
# ---------------------------------------------------------------------------
def _chunk_stats(chunks) -> dict:
    sizes = [len(c.text.split()) for c in chunks] or [0]
    return {
        "count": len(chunks),
        "avg_words": round(statistics.mean(sizes), 1),
        "min_words": min(sizes),
        "max_words": max(sizes),
        "std_words": round(statistics.pstdev(sizes), 1) if len(sizes) > 1 else 0.0,
    }


def _run_one_chunk_method(name: str, text_docs, embed_fn=None):
    from chunking.fixed_size import chunk_fixed_size
    from chunking.recursive import chunk_recursive
    from chunking.sentence_based import chunk_sentence_based
    from chunking.semantic import chunk_semantic
    from chunking.structure_aware import chunk_markdown_by_heading, chunk_pdf_page_as_unit
    from chunking.parent_child import build_parent_child_chunks

    all_chunks, parents_by_id = [], {}
    for doc in text_docs:
        if name == "fixed_size":
            all_chunks.extend(chunk_fixed_size(doc))
        elif name == "recursive":
            all_chunks.extend(chunk_recursive(doc))
        elif name == "sentence_based":
            all_chunks.extend(chunk_sentence_based(doc))
        elif name == "semantic":
            all_chunks.extend(chunk_semantic(doc, embed_fn=embed_fn))
        elif name == "structure_aware":
            all_chunks.extend(chunk_pdf_page_as_unit(doc) if doc.modality == "pdf_text"
                               else chunk_markdown_by_heading(doc))
        elif name == "parent_child":
            children, doc_parents = build_parent_child_chunks(doc)
            all_chunks.extend(children)
            parents_by_id.update(doc_parents)
        else:
            raise ValueError(f"Unknown chunking method: {name}")
    return all_chunks, parents_by_id


def step_chunk(methods: list = None, embedder_for_semantic: str = "hf"):
    if not STEP1_PATH.exists():
        raise FileNotFoundError(f"No step1 checkpoint at {STEP1_PATH} — run `python load_step.py ingest` first.")

    from ingestion.cache import load_raw_documents
    docs = load_raw_documents(str(STEP1_PATH))
    text_docs = [d for d in docs if d.modality in ("text", "pdf_text")]
    if not text_docs:
        raise ValueError("No text/pdf_text documents in the step1 checkpoint — nothing to chunk.")

    methods = methods or CHUNK_METHODS
    embed_fn = None
    if "semantic" in methods:
        # Semantic chunking needs an embedding function to detect topic shifts
        # between sentences — this is a lightweight on-the-fly call, not a
        # dependency on step3 (which embeds whole CHUNKS, a different thing).
        from pipeline import get_embedder
        embed_fn = get_embedder(embedder_for_semantic).embed_texts

    metrics = {}
    for name in methods:
        try:
            start = time.perf_counter()
            chunks, parents = _run_one_chunk_method(name, text_docs, embed_fn=embed_fn)
            elapsed = time.perf_counter() - start

            _save_json(CKPT_DIR / f"step2_chunks__{name}.json", [asdict(c) for c in chunks])
            if parents:
                _save_json(CKPT_DIR / f"step2_parents__{name}.json", {pid: asdict(p) for pid, p in parents.items()})

            stats = _chunk_stats(chunks)
            stats["time_s"] = round(elapsed, 3)
            metrics[name] = stats
            print(f"[chunk] {name:<16}{stats}")
        except Exception as e:
            print(f"[skip] chunk method {name}: {e}")
            metrics[name] = {"error": str(e)}

    _save_json(CKPT_DIR / "step2_metrics.json", metrics)
    return metrics


def _load_chunks(chunk_method: str) -> list:
    path = CKPT_DIR / f"step2_chunks__{chunk_method}.json"
    if not path.exists():
        raise FileNotFoundError(f"No step2 checkpoint for '{chunk_method}' at {path} — "
                                 f"run `python load_step.py chunk` first.")
    return [Chunk(**d) for d in _load_json(path)]


# ---------------------------------------------------------------------------
# Step 3: Embed — load ONE persisted chunk set, run ALL embedder candidates
# ---------------------------------------------------------------------------
def step_embed(chunk_method: str = "recursive", backends: list = None):
    chunks = _load_chunks(chunk_method)
    text_chunks = [c for c in chunks if c.modality == "text"]
    if not text_chunks:
        raise ValueError(f"No text chunks in step2 checkpoint '{chunk_method}' — nothing to embed.")
    texts = [c.text for c in text_chunks]
    ids = [c.chunk_id for c in text_chunks]
    metadatas = [c.metadata for c in text_chunks]

    candidates = backends or EMBEDDER_CANDIDATES
    metrics = {}
    for backend, model in candidates:
        key = f"{backend}:{model}"
        try:
            embedder = _resolve_embedder(key)

            # Sanity check on fixed sentences (same three used in
            # embeddings/benchmark_embedders.py) — flags a broken/misconfigured
            # model independent of how well it does on YOUR corpus.
            sanity_vecs = embedder.embed_texts(SANITY_TEXTS)
            sim_para = _cosine(sanity_vecs[0], sanity_vecs[1])
            sim_unrel = _cosine(sanity_vecs[0], sanity_vecs[2])

            # Real corpus throughput — the number that actually matters for
            # deciding if this embedder is practical at your corpus size.
            start = time.perf_counter()
            vectors = embedder.embed_texts(texts)
            elapsed = time.perf_counter() - start

            prefix = CKPT_DIR / f"step3_embeddings__{_safe_name(chunk_method)}__{_safe_name(key)}"
            _save_embedding_bundle(prefix, ids, vectors, texts, metadatas)

            metrics[key] = {
                "dims": int(vectors.shape[1]),
                "n_chunks_embedded": len(texts),
                "total_time_s": round(elapsed, 2),
                "ms_per_chunk": round((elapsed / max(len(texts), 1)) * 1000, 2),
                "sanity_sim_paraphrase": round(sim_para, 3),
                "sanity_sim_unrelated": round(sim_unrel, 3),
                "sane": sim_para > sim_unrel,
            }
            print(f"[embed] {key:<45}{metrics[key]}")
        except Exception as e:
            print(f"[skip] embedder {key}: {e}")
            metrics[key] = {"error": str(e)}

    _save_json(CKPT_DIR / f"step3_metrics__{_safe_name(chunk_method)}.json", metrics)
    return metrics


# ---------------------------------------------------------------------------
# Step 4: Store — load ONE persisted embedding bundle, populate ALL vector stores
# ---------------------------------------------------------------------------
def step_store(chunk_method: str = "recursive",
               embedder_key: str = "hf:sentence-transformers/all-MiniLM-L6-v2",
               stores: list = None):
    prefix = CKPT_DIR / f"step3_embeddings__{_safe_name(chunk_method)}__{_safe_name(embedder_key)}"
    ids, vectors, texts, metadatas = _load_embedding_bundle(prefix)

    stores = stores or STORE_CANDIDATES
    metrics = {}
    for store_name in stores:
        try:
            collection = f"{_safe_name(chunk_method)}__{_safe_name(embedder_key)}"
            if store_name == "chroma":
                from vectorstore.chroma_store import ChromaStore
                store = ChromaStore(collection_name=collection)
            elif store_name == "qdrant":
                from vectorstore.qdrant_store import QdrantStore
                store = QdrantStore(collection_name=collection, dimensions=vectors.shape[1])
            else:
                raise ValueError(f"Unknown store: {store_name}")

            start = time.perf_counter()
            store.upsert(ids, vectors, texts, metadatas)
            upsert_time = time.perf_counter() - start

            # Self-match sanity + query latency, on the REAL embeddings this
            # time (vectorstore/benchmark_stores.py does the same check but
            # on synthetic random vectors — this is the corpus-grounded version).
            n_probe = min(20, len(ids))
            start = time.perf_counter()
            correct = 0
            for i in range(n_probe):
                result = store.query(vectors[i], top_k=1)
                if result and str(result[0]["id"]) == str(ids[i]):
                    correct += 1
            query_time = (time.perf_counter() - start) / max(n_probe, 1)

            metrics[store_name] = {
                "collection": collection,
                "n_vectors": len(ids),
                "upsert_s_total": round(upsert_time, 3),
                "query_ms_avg": round(query_time * 1000, 2),
                "self_match_accuracy": round(correct / max(n_probe, 1), 3),
            }
            print(f"[store] {store_name:<10}{metrics[store_name]}")

            # Pointer so step 5 can find this exact collection later, in a
            # totally separate process, without re-running steps 1-4.
            _save_json(
                CKPT_DIR / f"step4_ref__{store_name}__{_safe_name(chunk_method)}__{_safe_name(embedder_key)}.json",
                {"store": store_name, "collection": collection, "chunk_method": chunk_method,
                 "embedder_key": embedder_key, "dimensions": int(vectors.shape[1])},
            )
        except Exception as e:
            print(f"[skip] store {store_name}: {e}")
            metrics[store_name] = {"error": str(e)}

    _save_json(CKPT_DIR / f"step4_metrics__{_safe_name(chunk_method)}__{_safe_name(embedder_key)}.json", metrics)
    return metrics


def _resolve_store(store_name: str, chunk_method: str, embedder_key: str):
    ref_path = CKPT_DIR / f"step4_ref__{store_name}__{_safe_name(chunk_method)}__{_safe_name(embedder_key)}.json"
    if not ref_path.exists():
        raise FileNotFoundError(f"No step4 checkpoint for store={store_name}, chunk_method={chunk_method}, "
                                 f"embedder={embedder_key} at {ref_path} — run `python load_step.py store` first.")
    ref = _load_json(ref_path)
    if store_name == "chroma":
        from vectorstore.chroma_store import ChromaStore
        return ChromaStore(collection_name=ref["collection"]), ref
    from vectorstore.qdrant_store import QdrantStore
    return QdrantStore(collection_name=ref["collection"], dimensions=ref["dimensions"]), ref


# ---------------------------------------------------------------------------
# Step 5: Retrieve — load ONE persisted store, run ALL retrieval strategies
# against the real labeled eval set (data/eval_set.json)
# ---------------------------------------------------------------------------
def step_retrieve(store_name: str = "chroma", chunk_method: str = "recursive",
                   embedder_key: str = "hf:sentence-transformers/all-MiniLM-L6-v2",
                   strategies: list = None, top_k: int = 5, generator_name: str = "ollama"):
    store, _ref = _resolve_store(store_name, chunk_method, embedder_key)
    embedder = _resolve_embedder(embedder_key)

    eval_set = load_eval_set()
    if not eval_set:
        print("[retrieve] WARNING: no data/eval_set.json found. Build one first with:\n"
              "    python -m evaluation.build_eval_set --interactive\n"
              "  Falling back to a 1-example placeholder — treat these numbers as a smoke "
              "test, not a real signal.")
        eval_set = [{"query": "test question", "relevant_ids": []}]

    strategies = strategies or RETRIEVAL_STRATEGIES
    metrics = {}
    for strategy in strategies:
        try:
            gen = None
            if strategy == "multi_query":
                from pipeline import get_generator
                gen = get_generator(generator_name)  # raises below if unreachable -> [skip]

            precisions, recalls, ranks, latencies = [], [], [], []
            per_query_results = []
            for example in eval_set:
                start = time.perf_counter()
                if strategy == "vector":
                    from retrieval.vector_retriever import vector_retrieve
                    results = vector_retrieve(example["query"], embedder, store, top_k=top_k)
                elif strategy == "hybrid":
                    from retrieval.hybrid_retriever import hybrid_retrieve
                    results = hybrid_retrieve(example["query"], embedder, store, top_k=top_k)
                elif strategy == "router":
                    from retrieval.query_router import rule_based_route, route_and_retrieve
                    decision_preview = rule_based_route(example["query"])
                    corpus = store.get_all() if decision_preview.route == "keyword_hybrid" else None
                    results, _decision = route_and_retrieve(example["query"], embedder, store,
                                                             corpus_for_hybrid=corpus, top_k=top_k)
                elif strategy == "multi_query":
                    from retrieval.multi_query import multi_query_retrieve
                    results = multi_query_retrieve(example["query"], embedder, store, gen, top_k_final=top_k)
                else:
                    raise ValueError(f"Unknown retrieval strategy: {strategy}")
                latencies.append(time.perf_counter() - start)

                retrieved_ids = [r["id"] for r in results]
                relevant = set(example.get("relevant_ids", []))
                if relevant:
                    precisions.append(precision_at_k(retrieved_ids, relevant, top_k))
                    recalls.append(recall_at_k(retrieved_ids, relevant, top_k))
                    ranks.append(mrr(retrieved_ids, relevant))
                per_query_results.append({"query": example["query"], "retrieved": results})

            out_path = CKPT_DIR / (f"step5_retrieval__{store_name}__{_safe_name(chunk_method)}__"
                                    f"{_safe_name(embedder_key)}__{strategy}.json")
            _save_json(out_path, per_query_results)

            avg = lambda lst: round(sum(lst) / len(lst), 3) if lst else None
            metrics[strategy] = {
                "precision_at_k": avg(precisions),
                "recall_at_k": avg(recalls),
                "mrr": avg(ranks),
                "avg_latency_ms": round(sum(latencies) / len(latencies) * 1000, 1) if latencies else None,
                "n_queries": len(eval_set),
            }
            print(f"[retrieve] {strategy:<14}{metrics[strategy]}")
        except Exception as e:
            print(f"[skip] retrieval strategy {strategy}: {e}")
            metrics[strategy] = {"error": str(e)}

    _save_json(CKPT_DIR / (f"step5_metrics__{store_name}__{_safe_name(chunk_method)}__"
                            f"{_safe_name(embedder_key)}.json"), metrics)
    return metrics


def _pick_best_strategy(metrics: dict, default: str = "vector") -> str:
    """The one place this script auto-picks a 'winner' — because it's the
    one place a real, labeled ground truth (relevant_ids in eval_set.json)
    exists to justify picking automatically instead of just reporting
    numbers and leaving the decision to you."""
    scored = {k: v for k, v in metrics.items() if isinstance(v, dict) and v.get("mrr") is not None}
    if not scored:
        return default
    return max(scored, key=lambda k: scored[k]["mrr"])


# ---------------------------------------------------------------------------
# Step 6: Generate — load ONE persisted retrieval result set, run ALL generators
# ---------------------------------------------------------------------------
def step_generate(store_name: str = "chroma", chunk_method: str = "recursive",
                   embedder_key: str = "hf:sentence-transformers/all-MiniLM-L6-v2",
                   strategy: str = "hybrid", generators: list = None):
    path = CKPT_DIR / (f"step5_retrieval__{store_name}__{_safe_name(chunk_method)}__"
                       f"{_safe_name(embedder_key)}__{strategy}.json")
    if not path.exists():
        raise FileNotFoundError(f"No step5 checkpoint at {path} — run `python load_step.py retrieve` first.")
    per_query_results = _load_json(path)

    candidates = generators or GENERATOR_CANDIDATES
    metrics = {}
    for backend, model in candidates:
        key = f"{backend}:{model}"
        try:
            if backend == "ollama":
                from generation.ollama_generator import OllamaGenerator
                gen = OllamaGenerator(model)
            else:
                from generation.hf_generator import HFGenerator
                gen = HFGenerator(model)

            answers, latencies, faithfulness_scores = [], [], []
            for item in per_query_results:
                context = item["retrieved"]
                start = time.perf_counter()
                answer = gen.generate(item["query"], context)
                latencies.append(time.perf_counter() - start)
                faithfulness_scores.append(keyword_faithfulness_heuristic(answer, [c["text"] for c in context]))
                answers.append({"query": item["query"], "answer": answer})

            _save_json(CKPT_DIR / f"step6_generation__{_safe_name(key)}.json", answers)

            metrics[key] = {
                "avg_latency_s": round(sum(latencies) / len(latencies), 2) if latencies else None,
                "avg_faithfulness": round(sum(faithfulness_scores) / len(faithfulness_scores), 3)
                                    if faithfulness_scores else None,
                "n_answers": len(answers),
            }
            print(f"[generate] {key:<38}{metrics[key]}")
        except Exception as e:
            print(f"[skip] generator {key}: {e}")
            metrics[key] = {"error": str(e)}

    _save_json(CKPT_DIR / "step6_metrics.json", metrics)
    return metrics


# ---------------------------------------------------------------------------
# status — inspect what's been checkpointed so far, no state needed to ask
# ---------------------------------------------------------------------------
def status():
    if not any(CKPT_DIR.iterdir()):
        print(f"No checkpoints yet in {CKPT_DIR}. Start with: python load_step.py ingest")
        return
    print(f"Checkpoints in {CKPT_DIR}:\n")
    for f in sorted(CKPT_DIR.iterdir()):
        if f.suffix == ".json" and "metrics" in f.name:
            try:
                data = _load_json(f)
                print(f"  {f.name}")
                for k, v in data.items():
                    print(f"      {k}: {v}")
            except Exception:
                print(f"  {f.name} (unreadable)")
        elif f.suffix in (".json", ".npy"):
            print(f"  {f.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_kv_list(items, sep_backend=True):
    """'hf:model,ollama:model2' -> [('hf','model'), ('ollama','model2')]"""
    if not items:
        return None
    out = []
    for item in items.split(","):
        item = item.strip()
        if sep_backend and ":" in item:
            backend, model = item.split(":", 1)
            out.append((backend, model))
        else:
            out.append(item)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("--source", default="data/raw")

    p_chunk = sub.add_parser("chunk")
    p_chunk.add_argument("--methods", default=None, help=f"comma-separated subset of {CHUNK_METHODS}")
    p_chunk.add_argument("--embedder-for-semantic", default="hf", choices=["hf", "ollama", "clip"])

    p_embed = sub.add_parser("embed")
    p_embed.add_argument("--chunk-method", default="recursive", choices=CHUNK_METHODS)
    p_embed.add_argument("--backends", default=None, help="comma-separated 'backend:model' pairs, e.g. hf:BAAI/bge-small-en-v1.5")

    p_store = sub.add_parser("store")
    p_store.add_argument("--chunk-method", default="recursive", choices=CHUNK_METHODS)
    p_store.add_argument("--embedder", default="hf:sentence-transformers/all-MiniLM-L6-v2")
    p_store.add_argument("--stores", default=None, help=f"comma-separated subset of {STORE_CANDIDATES}")

    p_retrieve = sub.add_parser("retrieve")
    p_retrieve.add_argument("--store", default="chroma", choices=STORE_CANDIDATES)
    p_retrieve.add_argument("--chunk-method", default="recursive", choices=CHUNK_METHODS)
    p_retrieve.add_argument("--embedder", default="hf:sentence-transformers/all-MiniLM-L6-v2")
    p_retrieve.add_argument("--strategies", default=None, help=f"comma-separated subset of {RETRIEVAL_STRATEGIES}")
    p_retrieve.add_argument("--top-k", type=int, default=5)
    p_retrieve.add_argument("--generator", default="ollama", help="used only for the multi_query strategy")

    p_generate = sub.add_parser("generate")
    p_generate.add_argument("--store", default="chroma", choices=STORE_CANDIDATES)
    p_generate.add_argument("--chunk-method", default="recursive", choices=CHUNK_METHODS)
    p_generate.add_argument("--embedder", default="hf:sentence-transformers/all-MiniLM-L6-v2")
    p_generate.add_argument("--strategy", default="hybrid", choices=RETRIEVAL_STRATEGIES)
    p_generate.add_argument("--generators", default=None, help="comma-separated 'backend:model' pairs, e.g. ollama:phi3")

    p_all = sub.add_parser("all", help="run every stage in order with sensible defaults, still checkpointing each")
    p_all.add_argument("--source", default="data/raw")
    p_all.add_argument("--chunk-method", default="recursive", choices=CHUNK_METHODS)
    p_all.add_argument("--embedder", default="hf:sentence-transformers/all-MiniLM-L6-v2")
    p_all.add_argument("--store", default="chroma", choices=STORE_CANDIDATES)

    sub.add_parser("status")

    args = parser.parse_args()

    if args.stage == "ingest":
        step_ingest(args.source)
    elif args.stage == "chunk":
        step_chunk(_parse_kv_list(args.methods, sep_backend=False), args.embedder_for_semantic)
    elif args.stage == "embed":
        step_embed(args.chunk_method, _parse_kv_list(args.backends))
    elif args.stage == "store":
        step_store(args.chunk_method, args.embedder, _parse_kv_list(args.stores, sep_backend=False))
    elif args.stage == "retrieve":
        step_retrieve(args.store, args.chunk_method, args.embedder,
                       _parse_kv_list(args.strategies, sep_backend=False), args.top_k, args.generator)
    elif args.stage == "generate":
        step_generate(args.store, args.chunk_method, args.embedder, args.strategy,
                       _parse_kv_list(args.generators))
    elif args.stage == "status":
        status()
    elif args.stage == "all":
        print("=== STAGE 1: ingest ===")
        step_ingest(args.source)
        print("\n=== STAGE 2: chunk (all methods) ===")
        step_chunk()
        print("\n=== STAGE 3: embed (all embedders, on the chosen chunk method) ===")
        step_embed(args.chunk_method)
        print("\n=== STAGE 4: store (all vector stores, on the chosen embedder) ===")
        step_store(args.chunk_method, args.embedder)
        print("\n=== STAGE 5: retrieve (all strategies, on the chosen store) ===")
        retrieve_metrics = step_retrieve(args.store, args.chunk_method, args.embedder)
        best_strategy = _pick_best_strategy(retrieve_metrics)
        print(f"\n  -> best strategy by MRR on your eval set: {best_strategy!r} "
              f"(auto-selected; override with --strategy on the `generate` stage if you disagree)")
        print("\n=== STAGE 6: generate (all generators, using the best retrieval strategy) ===")
        step_generate(args.store, args.chunk_method, args.embedder, best_strategy)
        print("\nAll stages complete. Run `python load_step.py status` any time to review checkpoints/metrics.")


if __name__ == "__main__":
    main()
