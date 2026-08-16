"""
Run the pipeline ONE STAGE AT A TIME, each stage independent of the ones
before it, using only what's persisted to disk between them — no shared
Python process, no in-memory state carried over. You can run `ingest`
today, shut everything down, and run `chunk` next week; each command reads
its input from a checkpoint file, does its job, and writes the next
checkpoint file. This is pipeline.py's exact same logic (get_embedder,
get_store, get_generator, the same chunking/retrieval functions) — just
split into six independently-runnable commands instead of two fused ones
("ingest" = ingest+chunk+embed+store, "ask" = retrieve+generate).

Stages and what bridges them:

    ingest    -> data/checkpoints/01_raw_documents.json
    chunk     -> data/checkpoints/02_chunks.json (+ 02_parents.json if --parent-child)
    embed     -> data/checkpoints/03_text_vectors.npz, 03_image_vectors.npz
    store     -> the persisted vector store itself (data/chroma_db/, or Qdrant)
    retrieve  -> data/checkpoints/04_retrieved.json
    generate  -> data/checkpoints/05_answer.json (+ printed to stdout)

Once `store` has run, the checkpoint files from ingest/chunk/embed are no
longer needed for retrieval — the vector store on disk IS the durable
result of that half of the pipeline. `retrieve` and `generate` only need
the vector store (+ a fresh embedder instance to embed the query) and
`04_retrieved.json` respectively.

IMPORTANT: the embedder used in `embed`/`store` must match the one used in
`retrieve` (same model = same vector space). Nothing enforces this
automatically — pass the same `--embedder` value to every stage.

Example, one command at a time (each is a separate, independent invocation):

    python stages.py ingest --source data/raw
    python stages.py chunk
    python stages.py embed --embedder hf
    python stages.py store --store chroma
    python stages.py retrieve "What does the document say about X?" --embedder hf --store chroma --retrieval hybrid
    python stages.py generate --generator ollama

    python stages.py status     # see which checkpoints exist
    python stages.py clean      # wipe all checkpoints and start over
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from config import DATA_DIR
from ingestion.loader import ingest_directory
from ingestion.schema import RawDocument, Chunk
from chunking.fixed_size import chunk_fixed_size
from chunking.recursive import chunk_recursive
from chunking.sentence_based import chunk_sentence_based
from chunking.semantic import chunk_semantic
from chunking.structure_aware import (
    chunk_pdf_page_as_unit,
    chunk_pdf_table_as_unit,
    chunk_markdown_by_heading,
)
from chunking.parent_child import build_parent_child_chunks, resolve_to_parents
from safety.pii_redaction import redact_chunks
from safety.prompt_injection import flag_suspicious_chunks
from pipeline import (
    get_embedder, get_store, get_generator, get_image_store,
    build_caption_chunks, build_page_description_chunks, retrieve_multimodal,
)
from utils.logging_config import get_logger

logger = get_logger("local_rag.stages")

CHECKPOINT_DIR = DATA_DIR / "checkpoints"
RAW_DOCS_PATH = CHECKPOINT_DIR / "01_raw_documents.json"
CHUNKS_PATH = CHECKPOINT_DIR / "02_chunks.json"
PARENTS_PATH = CHECKPOINT_DIR / "02_parents.json"
TEXT_VECTORS_PATH = CHECKPOINT_DIR / "03_text_vectors.npz"
IMAGE_VECTORS_PATH = CHECKPOINT_DIR / "03_image_vectors.npz"
RETRIEVED_PATH = CHECKPOINT_DIR / "04_retrieved.json"
ANSWER_PATH = CHECKPOINT_DIR / "05_answer.json"


# ---------------------------------------------------------------------------
# (De)serialization — RawDocument/Chunk are plain dataclasses with a dict
# `metadata` field, so they round-trip through JSON cleanly. A "kind" tag
# disambiguates the two types, since images pass through the chunk stage as
# RawDocument objects unchanged (CLIP embeds the raw image, no chunking needed).
# ---------------------------------------------------------------------------
def _serialize_item(item) -> dict:
    if isinstance(item, RawDocument):
        return {"kind": "raw_document", **asdict(item)}
    if isinstance(item, Chunk):
        return {"kind": "chunk", **asdict(item)}
    raise TypeError(f"Cannot checkpoint object of type {type(item)}")


def _deserialize_item(d: dict):
    d = dict(d)
    kind = d.pop("kind")
    if kind == "raw_document":
        return RawDocument(**d)
    if kind == "chunk":
        return Chunk(**d)
    raise ValueError(f"Unknown checkpoint item kind: {kind!r}")


def _json_default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    return str(o)


def _require(path: Path, hint: str):
    if not path.exists():
        raise SystemExit(f"[stages] missing checkpoint {path} — {hint}")


# ---------------------------------------------------------------------------
# Stage 1: Ingest
# ---------------------------------------------------------------------------
def cmd_ingest(args):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    raw_docs = ingest_directory(args.source, describe_pages=args.page_vlm)
    RAW_DOCS_PATH.write_text(json.dumps([_serialize_item(d) for d in raw_docs], indent=2))
    by_modality = {}
    for d in raw_docs:
        by_modality[d.modality] = by_modality.get(d.modality, 0) + 1
    print(f"[stage:ingest] {len(raw_docs)} raw document(s) {by_modality} -> {RAW_DOCS_PATH}")


# ---------------------------------------------------------------------------
# Stage 2: Chunk
# ---------------------------------------------------------------------------
_CHUNK_METHODS = {
    "fixed_size": chunk_fixed_size,
    "recursive": chunk_recursive,
    "sentence_based": chunk_sentence_based,
    "structure_aware": chunk_markdown_by_heading,
    # "semantic" is handled separately in cmd_chunk below since it needs an
    # embed_fn (an embedder instance), unlike the other four.
}


def cmd_chunk(args):
    _require(RAW_DOCS_PATH, "run `python stages.py ingest --source ...` first.")
    raw_docs = [_deserialize_item(d) for d in json.loads(RAW_DOCS_PATH.read_text())]

    # Only load an embedder if semantic chunking actually needs one — every
    # other method here is embedding-free, so don't pay that cost otherwise.
    embedder = get_embedder(args.embedder, use_cache=args.cache) if args.method == "semantic" else None

    parents_by_id: dict = {}
    all_chunks = []
    for doc in raw_docs:
        if doc.modality == "image":
            all_chunks.append(doc)  # CLIP embeds the raw image directly, no chunking needed
        elif doc.modality == "pdf_table":
            all_chunks.extend(chunk_pdf_table_as_unit(doc))  # never split a table mid-row
        elif args.parent_child:
            children, doc_parents = build_parent_child_chunks(doc)
            all_chunks.extend(children)
            parents_by_id.update({pid: {"text": p.text, "metadata": p.metadata} for pid, p in doc_parents.items()})
        elif args.method == "auto":
            # Previous hardcoded default: page-as-unit for PDFs (keeps OCR'd
            # pages intact), recursive splitting for everything else.
            if doc.modality == "pdf_text":
                all_chunks.extend(chunk_pdf_page_as_unit(doc))
            else:
                all_chunks.extend(chunk_recursive(doc))
        elif args.method == "semantic":
            all_chunks.extend(chunk_semantic(doc, embed_fn=embedder.embed_texts))
        else:
            all_chunks.extend(_CHUNK_METHODS[args.method](doc))

    text_chunks = [c for c in all_chunks if getattr(c, "modality", "text") == "text"]
    image_chunks = [c for c in all_chunks if getattr(c, "modality", "text") == "image"]

    if args.scan_injection and text_chunks:
        flagged = flag_suspicious_chunks(text_chunks)
        if flagged:
            print(f"[stage:chunk] {len(flagged)} chunk(s) flagged for possible prompt injection (logged, not dropped)")

    if args.redact_pii and text_chunks:
        text_chunks = redact_chunks(text_chunks)

    if args.multimodal and image_chunks:
        from ingestion.image_captioning import caption_images
        captions = caption_images(image_chunks, vlm_backend=args.vlm_backend, vlm_model=args.vlm_model)
        caption_chunks = build_caption_chunks(image_chunks, captions)
        text_chunks = text_chunks + caption_chunks
        print(f"[stage:chunk] captioned {len(image_chunks)} image(s), "
              f"dual-indexed {len(caption_chunks)} caption chunk(s) into the text set")

    if args.multimodal:
        # Strategy 3 — same VLM backend/model as image captioning above,
        # but describing whole pages the ingest stage already flagged as
        # visually complex (metadata["page_image_path"], set by
        # ingest_directory(describe_pages=...) in cmd_ingest). Only ever
        # processes that pre-flagged subset, never every page.
        pages_needing_description = [d for d in raw_docs if d.modality == "pdf_text"
                                      and d.metadata.get("page_image_path")]
        if pages_needing_description:
            from ingestion.page_description import describe_pages
            descriptions = describe_pages(pages_needing_description,
                                           vlm_backend=args.vlm_backend, vlm_model=args.vlm_model)
            description_chunks = build_page_description_chunks(pages_needing_description, descriptions)
            text_chunks = text_chunks + description_chunks
            print(f"[stage:chunk] described {len(pages_needing_description)} visually-complex page(s), "
                  f"dual-indexed {len(description_chunks)} description chunk(s) into the text set")

    all_chunks = text_chunks + image_chunks

    if args.parent_child:
        PARENTS_PATH.write_text(json.dumps(parents_by_id, indent=2))

    CHUNKS_PATH.write_text(json.dumps([_serialize_item(c) for c in all_chunks], indent=2))
    method_used = "parent_child" if args.parent_child else args.method
    print(f"[stage:chunk] method={method_used} -> {len(all_chunks)} chunk(s) "
          f"({len(text_chunks)} text, {len(image_chunks)} image) -> {CHUNKS_PATH}")


# ---------------------------------------------------------------------------
# Stage 3: Embed
# ---------------------------------------------------------------------------
def cmd_embed(args):
    _require(CHUNKS_PATH, "run `python stages.py chunk` first.")
    items = [_deserialize_item(d) for d in json.loads(CHUNKS_PATH.read_text())]
    text_chunks = [c for c in items if getattr(c, "modality", "text") == "text"]
    image_chunks = [c for c in items if getattr(c, "modality", "text") == "image"]

    embedder = get_embedder(args.embedder, use_cache=args.cache)

    if text_chunks:
        vectors = np.asarray(embedder.embed_texts([c.text for c in text_chunks]))
        ids = np.array([c.chunk_id for c in text_chunks])
        np.savez(TEXT_VECTORS_PATH, ids=ids, vectors=vectors)
        print(f"[stage:embed] {len(text_chunks)} text chunk(s) via {embedder.name} -> {TEXT_VECTORS_PATH}")

    if image_chunks:
        if args.multimodal:
            # Images always go through CLIP when --multimodal, regardless of
            # this stage's --embedder (that flag only governs text/captions
            # here) — see pipeline.ingest_images_multimodal's docstring for
            # why text and images deliberately use different embedders.
            clip_embedder = get_embedder("clip")
            vectors = np.asarray(clip_embedder.embed_images([c.image_path for c in image_chunks]))
            ids = np.array([c.doc_id for c in image_chunks])
            np.savez(IMAGE_VECTORS_PATH, ids=ids, vectors=vectors)
            print(f"[stage:embed] {len(image_chunks)} image chunk(s) via {clip_embedder.name} "
                  f"(--multimodal, always CLIP) -> {IMAGE_VECTORS_PATH}")
        elif embedder.supports_images():
            vectors = np.asarray(embedder.embed_images([c.image_path for c in image_chunks]))
            ids = np.array([c.doc_id for c in image_chunks])
            np.savez(IMAGE_VECTORS_PATH, ids=ids, vectors=vectors)
            print(f"[stage:embed] {len(image_chunks)} image chunk(s) via {embedder.name} -> {IMAGE_VECTORS_PATH}")
        else:
            print(f"[stage:embed] {len(image_chunks)} image chunk(s) found but "
                  f"{embedder.name} doesn't support images — skipped (use --multimodal, or --embedder clip)")

    if not text_chunks and not image_chunks:
        print("[stage:embed] nothing to embed (0 chunks in checkpoint)")


# ---------------------------------------------------------------------------
# Stage 4: Store
# ---------------------------------------------------------------------------
def cmd_store(args):
    _require(CHUNKS_PATH, "run `python stages.py chunk` first.")
    if not TEXT_VECTORS_PATH.exists() and not IMAGE_VECTORS_PATH.exists():
        raise SystemExit(f"[stages] no embeddings found in {CHECKPOINT_DIR} — run `python stages.py embed` first.")

    items_by_key = {}
    for d in json.loads(CHUNKS_PATH.read_text()):
        item = _deserialize_item(d)
        key = item.chunk_id if isinstance(item, Chunk) else item.doc_id
        items_by_key[key] = item

    store = None
    if TEXT_VECTORS_PATH.exists():
        data = np.load(TEXT_VECTORS_PATH, allow_pickle=True)
        text_dims = data["vectors"].shape[1]
        store = get_store(args.store, text_dims)
        ids = [str(i) for i in data["ids"]]
        vectors = data["vectors"]
        texts = [items_by_key[i].text for i in ids]
        metadatas = [items_by_key[i].metadata for i in ids]
        store.upsert(ids=ids, vectors=vectors, texts=texts, metadatas=metadatas)
        print(f"[stage:store] stored {len(ids)} text vector(s) into {args.store}")

    if IMAGE_VECTORS_PATH.exists():
        data = np.load(IMAGE_VECTORS_PATH, allow_pickle=True)
        # Read this vector file's OWN dimensionality rather than reusing
        # text_dims above — with --multimodal, images are CLIP-embedded
        # (e.g. 512d) while text may use a different embedder entirely
        # (e.g. 384d HF MiniLM), so the two can genuinely differ in size.
        image_dims = data["vectors"].shape[1]
        if args.multimodal:
            # Separate collection, kept apart from the text store above —
            # see config.CHROMA_IMAGE_COLLECTION / QDRANT_IMAGE_COLLECTION.
            image_store = get_image_store(args.store, image_dims)
        else:
            # Legacy behavior: images share the same collection as text
            # (this is the plain "--embedder clip embeds everything" path).
            image_store = store if store is not None else get_store(args.store, image_dims)
        ids = [str(i) for i in data["ids"]]
        vectors = data["vectors"]
        texts = ["" for _ in ids]
        metadatas = [{**items_by_key[i].metadata, "image_path": items_by_key[i].image_path} for i in ids]
        image_store.upsert(ids=ids, vectors=vectors, texts=texts, metadatas=metadatas)
        destination = "separate image collection" if args.multimodal else f"'{args.store}' (shared with text)"
        print(f"[stage:store] stored {len(ids)} image vector(s) into {destination}")
        if args.multimodal:
            print(f"[stage:store] image collection now has {image_store.count()} total vector(s)")

    if store is not None:
        print(f"[stage:store] text collection now has {store.count()} total vector(s) in '{args.store}'. "
              f"This is now the durable index — retrieve/generate no longer need the checkpoint files above.")


# ---------------------------------------------------------------------------
# Stage 5: Retrieve
# ---------------------------------------------------------------------------
def cmd_retrieve(args):
    embedder = get_embedder(args.embedder, use_cache=args.cache)
    store = get_store(args.store, embedder.dimensions)

    if store.count() == 0 and not args.multimodal:
        raise SystemExit("[stages] vector store is empty — run `python stages.py store` first.")

    if args.retrieval == "vector":
        from retrieval.vector_retriever import vector_retrieve
        results = vector_retrieve(args.question, embedder, store, top_k=args.top_k)

    elif args.retrieval == "hybrid":
        from retrieval.hybrid_retriever import hybrid_retrieve
        results = hybrid_retrieve(args.question, embedder, store, top_k=args.top_k)

    elif args.retrieval == "router":
        from retrieval.query_router import rule_based_route, route_and_retrieve
        decision_preview = rule_based_route(args.question)
        corpus = store.get_all() if decision_preview.route == "keyword_hybrid" else None
        results, decision = route_and_retrieve(args.question, embedder, store,
                                                corpus_for_hybrid=corpus, top_k=args.top_k)
        print(f"[stage:retrieve] router picked '{decision.route}' ({decision.reason})")

    elif args.retrieval == "multi_query":
        # multi_query needs a generator just to paraphrase the question — a
        # detail internal to this retrieval strategy, distinct from the
        # `generate` stage's final-answer generation later on.
        from retrieval.multi_query import multi_query_retrieve
        gen = get_generator(args.generator, vlm_backend=args.vlm_backend, vlm_model=args.vlm_model)
        results = multi_query_retrieve(args.question, embedder, store, gen, top_k_final=args.top_k)

    else:
        raise SystemExit(f"[stages] unknown retrieval strategy: {args.retrieval}")

    if args.parent_child:
        if not PARENTS_PATH.exists():
            print("[stage:retrieve] --parent-child set but no 02_parents.json checkpoint found — "
                  "was `chunk` run with --parent-child? Continuing with child chunks as-is.")
        else:
            parents_by_id_raw = json.loads(PARENTS_PATH.read_text())
            parents_lookup = {pid: Chunk.new(doc_id="", text=p["text"], **p["metadata"])
                               for pid, p in parents_by_id_raw.items()}
            results = resolve_to_parents(results, parents_lookup)

    if args.rerank:
        from retrieval.reranker import Reranker
        results = Reranker().rerank(args.question, results, top_k=args.top_k)

    if args.multimodal:
        # Added AFTER rerank/parent_child on purpose — see
        # pipeline.retrieve_multimodal()'s docstring (a cross-encoder
        # reranker scoring an image's empty text against the question would
        # likely rank it out of top_k).
        results = retrieve_multimodal(args.question, args.store, results, top_k=args.top_k)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RETRIEVED_PATH.write_text(json.dumps({"question": args.question, "results": results},
                                          indent=2, default=_json_default))
    print(f"[stage:retrieve] {len(results)} chunk(s) for {args.question!r} -> {RETRIEVED_PATH}")
    for r in results:
        preview = r["text"][:100] or f"[image: {r.get('metadata', {}).get('image_path', '?')}]"
        print(f"  [{r.get('score', 0):.3f}] {preview}")


# ---------------------------------------------------------------------------
# Stage 6: Generate
# ---------------------------------------------------------------------------
def cmd_generate(args):
    _require(RETRIEVED_PATH, 'run `python stages.py retrieve "your question"` first.')
    payload = json.loads(RETRIEVED_PATH.read_text())
    question, results = payload["question"], payload["results"]

    if args.compress:
        # 04_retrieved.json may already contain image-branch chunks (empty
        # text, tagged metadata["retrieval_branch"]=="image") if `retrieve`
        # was run with --multimodal — compressing an empty passage would
        # just get it judged NOT_RELEVANT and silently dropped, so only the
        # text branch goes through contextual compression.
        from retrieval.contextual_compression import compress_retrieved_chunks
        gen_for_compress = get_generator(args.generator, vlm_backend=args.vlm_backend, vlm_model=args.vlm_model)
        text_only = [r for r in results if r.get("metadata", {}).get("retrieval_branch") != "image"]
        image_only = [r for r in results if r.get("metadata", {}).get("retrieval_branch") == "image"]
        results = compress_retrieved_chunks(question, text_only, gen_for_compress) + image_only

    generator = get_generator(args.generator, vlm_backend=args.vlm_backend, vlm_model=args.vlm_model)

    if args.multimodal:
        if args.generator == "vlm":
            raise SystemExit(
                "--multimodal already handles the image branch itself (see "
                "generation/dual_modality_generator.py) — pass --generator ollama or hf, "
                "not vlm, so the text branch has a plain text generator to draft from."
            )
        from generation.dual_modality_generator import DualModalityGenerator
        generator = DualModalityGenerator(generator, vlm_backend=args.vlm_backend, vlm_model=args.vlm_model)

    answer = generator.generate(question, results)

    print(f"\n--- Answer ({generator.name}) ---\n{answer}")
    ANSWER_PATH.write_text(json.dumps({"question": question, "answer": answer, "generator": generator.name},
                                       indent=2))
    print(f"\n[stage:generate] saved -> {ANSWER_PATH}")


# ---------------------------------------------------------------------------
# Convenience: status / clean
# ---------------------------------------------------------------------------
def cmd_status(_args):
    checkpoints = [RAW_DOCS_PATH, CHUNKS_PATH, PARENTS_PATH, TEXT_VECTORS_PATH,
                   IMAGE_VECTORS_PATH, RETRIEVED_PATH, ANSWER_PATH]
    print(f"Checkpoint directory: {CHECKPOINT_DIR}\n")
    for p in checkpoints:
        mark = "✓" if p.exists() else " "
        size = f"({p.stat().st_size} bytes)" if p.exists() else ""
        print(f"  [{mark}] {p.name} {size}")


def cmd_clean(_args):
    if not CHECKPOINT_DIR.exists():
        print("Nothing to clean.")
        return
    for f in CHECKPOINT_DIR.glob("*"):
        f.unlink()
    print(f"Cleared all checkpoints in {CHECKPOINT_DIR}. "
          f"(The vector store itself is untouched — delete data/chroma_db/ separately if you want a clean index too.)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run the RAG pipeline one independent stage at a time.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="Stage 1: parse raw documents -> 01_raw_documents.json")
    p.add_argument("--source", required=True)
    p.add_argument("--page-vlm", dest="page_vlm", choices=["auto", "always", "never"], default="auto",
                    help="Strategy 3 (whole-page VLM description) flagging. 'auto' (default): cheap, local, "
                         "no-model-call heuristic flags only chart/diagram-heavy pages. 'always' flags every "
                         "page. 'never' disables flagging entirely. The VLM call itself happens in `chunk "
                         "--multimodal`, not here — this stage only flags + renders candidate pages.")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("chunk", help="Stage 2: chunk raw documents -> 02_chunks.json")
    p.add_argument("--method", choices=["auto", "fixed_size", "recursive", "sentence_based",
                                         "semantic", "structure_aware"], default="auto",
                   help="auto = page-as-unit for PDFs / recursive otherwise (previous hardcoded behavior). "
                        "Pick one explicitly to actually use the other methods — see "
                        "chunking/benchmark_chunkers.py to compare them on your own corpus first. "
                        "Ignored if --parent-child is set.")
    p.add_argument("--embedder", choices=["hf", "ollama", "clip"], default="hf",
                   help="Only used when --method semantic (it embeds sentences to find topic breaks)")
    p.add_argument("--cache", action="store_true")
    p.add_argument("--parent-child", dest="parent_child", action="store_true")
    p.add_argument("--redact-pii", dest="redact_pii", action="store_true")
    p.add_argument("--scan-injection", dest="scan_injection", action="store_true")
    p.add_argument("--multimodal", action="store_true",
                    help="Caption each image with a VLM and dual-index the caption into the text set — "
                         "see pipeline.build_caption_chunks. Pairs with embed/store/retrieve/generate --multimodal.")
    p.add_argument("--vlm-backend", dest="vlm_backend", choices=["ollama", "hf"], default="ollama",
                    help="Only used when --multimodal (captions each image)")
    p.add_argument("--vlm-model", dest="vlm_model", default=None,
                    help="Only used when --multimodal (defaults to llava/moondream2 per backend)")
    p.set_defaults(func=cmd_chunk)

    p = sub.add_parser("embed", help="Stage 3: embed chunks -> 03_*_vectors.npz")
    p.add_argument("--embedder", choices=["hf", "ollama", "clip"], default="hf")
    p.add_argument("--cache", action="store_true")
    p.add_argument("--multimodal", action="store_true",
                    help="Images always embed via CLIP regardless of --embedder (which then only governs "
                         "text/captions) — must match whether `chunk`/`store` also used --multimodal.")
    p.set_defaults(func=cmd_embed)

    p = sub.add_parser("store", help="Stage 4: upsert vectors into the vector store")
    p.add_argument("--store", choices=["chroma", "qdrant"], default="chroma")
    p.add_argument("--multimodal", action="store_true",
                    help="Route image vectors into their own collection instead of sharing the text one — "
                         "must match whether `embed` also used --multimodal.")
    p.set_defaults(func=cmd_store)

    p = sub.add_parser("retrieve", help="Stage 5: retrieve chunks for a question -> 04_retrieved.json")
    p.add_argument("question")
    p.add_argument("--embedder", choices=["hf", "ollama", "clip"], default="hf",
                    help="MUST match the embedder used in `embed`/`store`")
    p.add_argument("--store", choices=["chroma", "qdrant"], default="chroma")
    p.add_argument("--retrieval", choices=["vector", "hybrid", "router", "multi_query"], default="vector")
    p.add_argument("--rerank", action="store_true")
    p.add_argument("--parent-child", dest="parent_child", action="store_true")
    p.add_argument("--multimodal", action="store_true",
                    help="Also retrieve from the CLIP image index (must match `store --multimodal`) — results "
                         "are tagged metadata['retrieval_branch'] so `generate --multimodal` can split them back "
                         "apart. Added AFTER --rerank/--parent-child, before saving the checkpoint.")
    p.add_argument("--generator", choices=["ollama", "hf", "vlm", "vllm-server"], default="ollama",
                    help="Only used when --retrieval multi_query")
    p.add_argument("--vlm-backend", dest="vlm_backend", choices=["ollama", "hf"], default="ollama")
    p.add_argument("--vlm-model", dest="vlm_model", default=None)
    p.add_argument("--cache", action="store_true")
    p.add_argument("--top-k", dest="top_k", type=int, default=5)
    p.set_defaults(func=cmd_retrieve)

    p = sub.add_parser("generate", help="Stage 6: generate an answer from 04_retrieved.json -> 05_answer.json")
    p.add_argument("--generator", choices=["ollama", "hf", "vlm", "vllm-server"], default="ollama")
    p.add_argument("--vlm-backend", dest="vlm_backend", choices=["ollama", "hf"], default="ollama")
    p.add_argument("--vlm-model", dest="vlm_model", default=None)
    p.add_argument("--compress", action="store_true")
    p.add_argument("--multimodal", action="store_true",
                    help="Answer with generation/dual_modality_generator.py: independent text/image drafts, "
                         "each dropped if not viable, synthesized if both are. Use with --generator ollama or "
                         "hf, not vlm. Expects 04_retrieved.json to already contain image-branch results "
                         "(i.e. `retrieve` was also run with --multimodal).")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("status", help="Show which checkpoint files currently exist")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("clean", help="Delete all checkpoint files (leaves the vector store untouched)")
    p.set_defaults(func=cmd_clean)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
