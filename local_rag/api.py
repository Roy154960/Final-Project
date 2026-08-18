"""
REST API - ingestion + query endpoints over the existing local RAG pipeline.

This does not reimplement anything: it wires the same modules pipeline.py's
CLI uses (ingestion/, chunking/, embeddings/, vectorstore/, retrieval/,
generation/, safety/) behind HTTP endpoints, so "ingest a document through
your API" and "ask a question, get a grounded answer" (the two required
endpoints) work the same way the CLI does.

Endpoints:
    POST /ingest         upload a document -> parse -> chunk -> embed -> store
                          (multimodal=true also CLIP-embeds+captions images
                          into their own index — see pipeline.ingest_images_multimodal;
                          PDF tables are extracted as structured markdown chunks
                          automatically, always on — see ingestion/table_extraction.py;
                          page_vlm controls Strategy 3 whole-page VLM description,
                          the VLM call itself only runs when multimodal=true —
                          see pipeline.describe_complex_pages)
    POST /query          ask a question -> retrieve -> (rerank) -> generate
                          (multimodal=true also retrieves+answers from the
                          image index — see generation/dual_modality_generator.py)
    POST /query/compare  run one question through multiple retrieval
                         strategies side-by-side (retrieval only, no
                         generation) — the "compare methods" step for
                         retrieval, applied live over your real corpus
    GET  /documents      list what's currently indexed
    GET  /health          readiness + current config

Run:
    uvicorn api:app --reload --port 8001

Config via environment variables (all optional, free/local defaults):
    RAG_EMBEDDER   = hf | ollama | clip      (default: hf)
    RAG_STORE      = chroma | qdrant         (default: chroma)
    RAG_GENERATOR  = ollama | hf | vlm       (default: ollama, needs `ollama serve`
                                               + `ollama pull llama3.2`; switch to
                                               `hf` if you don't want to run Ollama.
                                               NOTE: multimodal=true on /query is
                                               incompatible with RAG_GENERATOR=vlm —
                                               see _get_dual_generator()'s docstring.)

Examples:
    curl -X POST "http://localhost:8001/ingest" -F "file=@data/raw/my_doc.pdf"

    curl -X POST "http://localhost:8001/ingest?multimodal=true" -F "file=@data/raw/my_doc.pdf"

    curl -X POST "http://localhost:8001/ingest?multimodal=true&page_vlm=always" \\
        -F "file=@data/raw/chart_heavy_report.pdf"

    curl -X POST "http://localhost:8001/query" -H "Content-Type: application/json" \\
        -d '{"question": "What does the document say about X?", "rerank": true}'

    curl -X POST "http://localhost:8001/query" -H "Content-Type: application/json" \\
        -d '{"question": "Which figure illustrates glazing?", "multimodal": true}'
"""

import os
import shutil
import time
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import pipeline as pipeline_mod
from pipeline import (
    get_embedder, get_store, get_generator, get_image_store,
    ingest_images_multimodal, retrieve_multimodal, describe_complex_pages,
)
from config import RAW_DOCS_DIR
from utils.logging_config import get_logger

from ingestion.loader import ingest_path
from ingestion.schema import Chunk
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

logger = get_logger("local_rag.api")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBEDDER_NAME = os.environ.get("RAG_EMBEDDER", "hf")
STORE_NAME = os.environ.get("RAG_STORE", "chroma")
GENERATOR_NAME = os.environ.get("RAG_GENERATOR", "ollama")

app = FastAPI(
    title="Local RAG API",
    description="Local, free, fully-offline retrieval-augmented generation over your own documents.",
    version="1.0",
)

# Embedder + vector store are loaded once at startup (cheap-ish, and every
# request needs them anyway). The generator (an LLM) and the reranker (a
# cross-encoder) are both loaded lazily on first use, so the API comes up
# instantly even before Ollama is running or a model has been downloaded.
# dual_generators caches DualModalityGenerator instances (which each wrap a
# loaded VLM) keyed by (vlm_backend, vlm_model), so different /query
# requests using the same VLM config don't reload it every time.
# clip_embedder / image_store / caption_vlms exist for the same reason —
# pipeline.ingest_images_multimodal() and pipeline.retrieve_multimodal()
# both load their own CLIP embedder and reconnect to the image store on
# every call by default (fine for pipeline.py's CLI, where each stage is a
# short-lived process), which would mean api.py — a long-running server —
# reloading the CLIP model from scratch on every single /ingest or /query
# call. Loaded once here instead and passed in explicitly.
_state = {"embedder": None, "store": None, "generator": None, "reranker": None, "dual_generators": {},
          "clip_embedder": None, "image_store": None, "caption_vlms": {}}


@app.on_event("startup")
def _startup():
    print(f"[api:startup] loading embedder={EMBEDDER_NAME!r}, store={STORE_NAME!r}...")
    _state["embedder"] = get_embedder(EMBEDDER_NAME)
    _state["store"] = get_store(STORE_NAME, _state["embedder"].dimensions)
    print(f"[api:startup] ready — {_state['store'].count()} vector(s) already indexed, "
          f"generator will load lazily on first /query (configured: {GENERATOR_NAME!r})\n")
    logger.info("api started", extra={"embedder": EMBEDDER_NAME, "store": STORE_NAME,
                                       "generator_configured": GENERATOR_NAME})


def _get_generator():
    if _state["generator"] is None:
        print(f"[api:lazy-load] loading generator={GENERATOR_NAME!r} (first use)...")
        _state["generator"] = get_generator(GENERATOR_NAME)
        print(f"[api:lazy-load] generator ready: {_state['generator'].name}")
    return _state["generator"]


def _get_dual_generator(vlm_backend: str, vlm_model: Optional[str]):
    """Lazily builds (and caches) a DualModalityGenerator for --multimodal-style
    /query requests — see generation/dual_modality_generator.py. Wraps the same
    cached text generator _get_generator() returns, so RAG_GENERATOR=vlm is
    incompatible here for the same reason it is in pipeline.py's cmd_ask."""
    if GENERATOR_NAME == "vlm":
        raise HTTPException(
            status_code=400,
            detail="multimodal=true already handles the image branch itself (see "
                   "generation/dual_modality_generator.py) — set RAG_GENERATOR to ollama or hf, "
                   "not vlm, so the text branch has a plain text generator to draft from.",
        )
    key = (vlm_backend, vlm_model)
    if key not in _state["dual_generators"]:
        print(f"[api:lazy-load] loading dual generator (vlm_backend={vlm_backend!r}, "
              f"vlm_model={vlm_model!r}) (first use)...")
        from generation.dual_modality_generator import DualModalityGenerator
        _state["dual_generators"][key] = DualModalityGenerator(
            _get_generator(), vlm_backend=vlm_backend, vlm_model=vlm_model
        )
        print(f"[api:lazy-load] dual generator ready: {_state['dual_generators'][key].name}")
    return _state["dual_generators"][key]


def _get_reranker():
    # Loaded once and reused — instantiating a fresh CrossEncoder per request
    # would reload the model from disk on every single call.
    if _state["reranker"] is None:
        print("[api:lazy-load] loading cross-encoder reranker (first use)...")
        from retrieval.reranker import Reranker
        _state["reranker"] = Reranker()
        print("[api:lazy-load] reranker ready")
    return _state["reranker"]


def _get_clip_embedder():
    if _state["clip_embedder"] is None:
        print("[api:lazy-load] loading CLIP embedder (first multimodal use)...")
        _state["clip_embedder"] = get_embedder("clip")
        print(f"[api:lazy-load] CLIP embedder ready: {_state['clip_embedder'].name}")
    return _state["clip_embedder"]


def _get_image_store():
    if _state["image_store"] is None:
        _state["image_store"] = get_image_store(STORE_NAME, _get_clip_embedder().dimensions)
        print(f"[api:lazy-load] image store ready: {_state['image_store'].name}")
    return _state["image_store"]


def _get_caption_vlm(vlm_backend: str, vlm_model: Optional[str]):
    key = (vlm_backend, vlm_model)
    if key not in _state["caption_vlms"]:
        print(f"[api:lazy-load] loading captioning VLM (vlm_backend={vlm_backend!r}, "
              f"vlm_model={vlm_model!r}) (first use)...")
        from ingestion.image_captioning import load_vlm
        _state["caption_vlms"][key] = load_vlm(vlm_backend, vlm_model)
        print("[api:lazy-load] captioning VLM ready")
    return _state["caption_vlms"][key]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class IngestResponse(BaseModel):
    filename: str
    n_raw_documents: int
    n_chunks: int
    n_text_chunks: int
    n_image_chunks: int
    n_flagged_injection: int
    n_pii_redacted: int
    n_captions_indexed: int = 0  # only >0 when multimodal=true — see /ingest
    n_table_chunks: int = 0  # tables extracted via pdfplumber, indexed alongside page text — always on
    n_pages_flagged_for_description: int = 0  # Strategy 3 — see page_vlm param on /ingest
    n_page_descriptions_indexed: int = 0  # only >0 when multimodal=true AND pages were flagged


class BatchIngestFileResult(BaseModel):
    filename: str
    ok: bool
    result: Optional[IngestResponse] = None  # set when ok=True
    error: Optional[str] = None              # set when ok=False -- always a plain, safe-to-display message


class BatchIngestResponse(BaseModel):
    n_files: int
    n_succeeded: int
    n_failed: int
    files: List[BatchIngestFileResult]


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    retrieval: str = "vector"        # vector | hybrid | router | multi_query
    rerank: bool = False
    parent_child: bool = False
    compress: bool = False
    multimodal: bool = False         # also retrieve + answer from the CLIP image index — see /ingest's multimodal
    vlm_backend: str = "ollama"      # only used when multimodal=true
    vlm_model: Optional[str] = None  # only used when multimodal=true


class SourceChunk(BaseModel):
    id: str
    text: str
    score: float
    filename: Optional[str] = None
    page: Optional[int] = None
    image_path: Optional[str] = None  # see this field's own comment at the query() route below for when this is set
    source_type: Optional[str] = None  # "image_caption" = a TEXT-store hit that's really an image's dual-indexed
                                        # caption+nearby-text (see pipeline.build_caption_chunks) -- distinguishes
                                        # this from a genuine prose chunk without needing to infer it from image_path
                                        # alone. None for plain text chunks and for retrieval_branch="image" CLIP hits.


class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieval_strategy_used: str
    sources: list[SourceChunk]


class DocumentsResponse(BaseModel):
    n_files: int
    n_chunks: int
    files: dict[str, int]


class HealthResponse(BaseModel):
    status: str
    embedder: Optional[str]
    store: str
    n_indexed_chunks: int
    generator_configured: str
    generator_loaded: bool


class CompareRequest(BaseModel):
    question: str
    top_k: int = 5
    strategies: list[str] = ["vector", "hybrid", "router"]  # multi_query opt-in: costs an LLM call
    rerank: bool = False


class StrategyResult(BaseModel):
    strategy: str
    latency_ms: Optional[float] = None
    top_score: Optional[float] = None
    n_results: int = 0
    sources: list[SourceChunk] = []
    error: Optional[str] = None


class CompareResponse(BaseModel):
    question: str
    results: list[StrategyResult]


_VALID_STRATEGIES = {"vector", "hybrid", "router", "multi_query"}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health():
    store = _state["store"]
    return HealthResponse(
        status="ok",
        embedder=_state["embedder"].name if _state["embedder"] else None,
        store=STORE_NAME,
        n_indexed_chunks=store.count() if store else 0,
        generator_configured=GENERATOR_NAME,
        generator_loaded=_state["generator"] is not None,
    )


@app.get("/documents", response_model=DocumentsResponse)
def list_documents():
    """List distinct source files currently indexed, with chunk counts each."""
    store = _state["store"]
    records = store.get_all()
    by_file: dict[str, int] = {}
    for r in records:
        fname = r.get("metadata", {}).get("filename", "unknown")
        by_file[fname] = by_file.get(fname, 0) + 1
    return DocumentsResponse(n_files=len(by_file), n_chunks=len(records), files=by_file)


@app.post("/query/compare", response_model=CompareResponse)
def compare_retrieval_strategies(req: CompareRequest):
    """
    Run the same question through multiple retrieval strategies side by side
    — retrieval only, no generation — so you can see which one actually
    performs best on YOUR indexed corpus. This is the same "compare multiple
    methods, don't just pick one" ethos as chunking/benchmark_chunkers.py and
    embeddings/benchmark_embedders.py, applied live over HTTP against your
    real corpus instead of an offline toy one. A strategy that errors (e.g.
    multi_query with no Ollama server running) is reported per-strategy
    rather than failing the whole comparison.
    """
    embedder = _state["embedder"]
    store = _state["store"]

    if store.count() == 0:
        raise HTTPException(status_code=409, detail="Index is empty — POST a document to /ingest first.")

    unknown = set(req.strategies) - _VALID_STRATEGIES
    if unknown:
        raise HTTPException(status_code=400,
                             detail=f"Unknown strategies: {sorted(unknown)} (valid: {sorted(_VALID_STRATEGIES)})")

    print(f"\n[api:compare] question={req.question!r} strategies={req.strategies} rerank={req.rerank}")

    results_out = []
    for strategy in req.strategies:
        start = time.perf_counter()
        try:
            if strategy == "vector":
                from retrieval.vector_retriever import vector_retrieve
                results = vector_retrieve(req.question, embedder, store, top_k=req.top_k)

            elif strategy == "hybrid":
                from retrieval.hybrid_retriever import hybrid_retrieve
                results = hybrid_retrieve(req.question, embedder, store, top_k=req.top_k)

            elif strategy == "router":
                from retrieval.query_router import rule_based_route, route_and_retrieve
                decision_preview = rule_based_route(req.question)
                corpus = store.get_all() if decision_preview.route == "keyword_hybrid" else None
                results, _decision = route_and_retrieve(req.question, embedder, store,
                                                         corpus_for_hybrid=corpus, top_k=req.top_k)

            else:  # multi_query
                from retrieval.multi_query import multi_query_retrieve
                results = multi_query_retrieve(req.question, embedder, store, _get_generator(),
                                                top_k_final=req.top_k)

            if req.rerank:
                results = _get_reranker().rerank(req.question, results, top_k=req.top_k)

            elapsed_ms = (time.perf_counter() - start) * 1000
            print(f"[api:compare]   {strategy:12s} -> {len(results)} result(s) in {elapsed_ms:.0f}ms")
            sources = [
                SourceChunk(
                    id=str(r.get("id", "")),
                    text=r["text"][:300],
                    score=float(r.get("rerank_score", r.get("score", 0.0))),
                    filename=r.get("metadata", {}).get("filename"),
                    page=r.get("metadata", {}).get("page"),
                )
                for r in results
            ]
            results_out.append(StrategyResult(
                strategy=strategy,
                latency_ms=round(elapsed_ms, 1),
                top_score=sources[0].score if sources else None,
                n_results=len(sources),
                sources=sources,
            ))
        except Exception as e:
            print(f"[api:compare]   {strategy:12s} -> FAILED: {e}")
            logger.warning("compare strategy failed", extra={"strategy": strategy, "error": str(e)})
            results_out.append(StrategyResult(strategy=strategy, error=str(e)))

    print("[api:compare] done\n")
    return CompareResponse(question=req.question, results=results_out)


_CHUNK_METHODS = {
    "fixed_size": chunk_fixed_size,
    "recursive": chunk_recursive,
    "sentence_based": chunk_sentence_based,
    "structure_aware": chunk_markdown_by_heading,
    # "semantic" handled separately below since it needs an embed_fn.
}


def _validate_ingest_params(chunking: str, page_vlm: str) -> None:
    """Shared validation for both /ingest and /ingest/batch -- raises the
    same HTTPException either route already raised before this refactor,
    just runs ONCE per request now instead of once per file in a batch
    (the params apply uniformly across every file in a batch, so there's
    no reason to re-validate them per file)."""
    if chunking != "auto" and chunking != "semantic" and chunking not in _CHUNK_METHODS:
        raise HTTPException(status_code=400,
                             detail=f"Unknown chunking method: {chunking!r} "
                                    f"(valid: auto, semantic, {sorted(_CHUNK_METHODS)})")
    if page_vlm not in ("auto", "always", "never"):
        raise HTTPException(status_code=400,
                             detail=f"Unknown page_vlm: {page_vlm!r} (valid: auto, always, never)")


def _ingest_one_file(
    file: UploadFile,
    *,
    redact_pii: bool,
    scan_injection: bool,
    parent_child: bool,
    chunking: str,
    multimodal: bool,
    vlm_backend: str,
    vlm_model: Optional[str],
    page_vlm: str,
) -> IngestResponse:
    """
    The actual parse -> chunk -> embed -> store pipeline for ONE file --
    every line of this is the original /ingest route's own body,
    unchanged, just extracted so /ingest/batch (below) can call it once
    per file in a loop without duplicating ~150 lines of logic a second
    time. /ingest itself (below) is now a thin wrapper around this same
    function, so a single-file upload behaves EXACTLY as it always has --
    same validation, same staging path, same response shape.

    Raises HTTPException on a per-file problem (bad parse, no extractable
    content) exactly as before -- /ingest lets that propagate straight to
    the caller (unchanged behavior); /ingest/batch (below) catches it
    per-file instead, so one bad PDF in a batch doesn't abort the rest.
    """
    embedder = _state["embedder"]
    store = _state["store"]

    print(f"[api:ingest] received {file.filename!r} ({file.content_type}), "
          f"multimodal={multimodal}, chunking={chunking!r}, page_vlm={page_vlm!r}")

    dest_path = RAW_DOCS_DIR / f"{uuid.uuid4().hex}_{file.filename}"
    try:
        with dest_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        file.file.close()
    print(f"[api:ingest] staged to disk -> {dest_path}")

    try:
        raw_docs = ingest_path(str(dest_path), describe_pages=page_vlm)
    except ValueError as e:
        dest_path.unlink(missing_ok=True)
        print(f"[api:ingest] parsing FAILED: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    if not raw_docs:
        print("[api:ingest] parsing produced no extractable content — aborting")
        raise HTTPException(status_code=422, detail="No extractable content found in this file "
                                                      "(check server logs — OCR may have failed).")

    print(f"[api:ingest] parsed -> {len(raw_docs)} raw document(s) "
          f"({sum(1 for d in raw_docs if d.modality != 'image')} text/pdf, "
          f"{sum(1 for d in raw_docs if d.modality == 'image')} image)")

    # The file was saved to disk under a UUID-prefixed name (to avoid collisions
    # between uploads), and ingestion sets metadata["filename"] from that saved
    # path — restore the original upload name here so source attribution in
    # /query responses and generated answers shows what the user actually
    # uploaded, not an internal disk name.
    for doc in raw_docs:
        doc.metadata["filename"] = file.filename

    parents_by_id = pipeline_mod._load_parents_store() if parent_child else {}
    all_chunks = []
    for doc in raw_docs:
        if doc.modality == "image":
            all_chunks.append(doc)  # CLIP embeds the raw image directly, no chunking needed
        elif doc.modality == "pdf_table":
            all_chunks.extend(chunk_pdf_table_as_unit(doc))  # never split a table mid-row
        elif parent_child:
            children, doc_parents = build_parent_child_chunks(doc)
            all_chunks.extend(children)
            parents_by_id.update({pid: {"text": p.text, "metadata": p.metadata} for pid, p in doc_parents.items()})
        elif chunking == "auto":
            # Previous hardcoded default: page-as-unit for PDFs, recursive otherwise.
            if doc.modality == "pdf_text":
                all_chunks.extend(chunk_pdf_page_as_unit(doc))
            else:
                all_chunks.extend(chunk_recursive(doc))
        elif chunking == "semantic":
            all_chunks.extend(chunk_semantic(doc, embed_fn=embedder.embed_texts))
        else:
            all_chunks.extend(_CHUNK_METHODS[chunking](doc))

    if parent_child:
        pipeline_mod._save_parents_store(parents_by_id)

    text_chunks = [c for c in all_chunks if getattr(c, "modality", "text") == "text"]
    image_chunks = [c for c in all_chunks if getattr(c, "modality", "text") == "image"]
    n_table_chunks = sum(1 for c in text_chunks if c.metadata.get("source_type") == "table")
    print(f"[api:ingest] chunked -> {len(text_chunks)} text chunk(s), {len(image_chunks)} image chunk(s) "
          f"(method={'parent_child' if parent_child else chunking})")

    n_captions_indexed = 0

    n_flagged = 0
    if scan_injection and text_chunks:
        flagged = flag_suspicious_chunks(text_chunks)
        n_flagged = len(flagged)
        print(f"[api:ingest] injection scan -> {n_flagged} chunk(s) flagged")
        if flagged:
            logger.warning("chunks flagged for possible prompt injection",
                            extra={"uploaded_filename": file.filename, "n_flagged": n_flagged})

    n_redacted = 0
    if redact_pii and text_chunks:
        text_chunks = redact_chunks(text_chunks)
        n_redacted = sum(1 for c in text_chunks if c.metadata.get("pii_redacted"))
        print(f"[api:ingest] PII redaction -> {n_redacted} chunk(s) had something redacted")

    if text_chunks:
        print(f"[api:ingest] embedding {len(text_chunks)} text chunk(s) via {embedder.name}...")
        vectors = embedder.embed_texts([c.text for c in text_chunks])
        ids = [c.chunk_id for c in text_chunks]
        store.upsert(ids=ids, vectors=vectors, texts=[c.text for c in text_chunks],
                     metadatas=[c.metadata for c in text_chunks])
        print(f"[api:ingest] stored -> text collection now has {store.count()} total vector(s)")

    if image_chunks:
        if multimodal:
            print(f"[api:ingest] multimodal: CLIP-embedding {len(image_chunks)} image(s) into their own "
                  f"collection + VLM-captioning each one (backend={vlm_backend})...")
            stats = ingest_images_multimodal(image_chunks, embedder, store, STORE_NAME,
                                              vlm_backend=vlm_backend, vlm_model=vlm_model,
                                              clip_embedder=_get_clip_embedder(), image_store=_get_image_store(),
                                              vlm=_get_caption_vlm(vlm_backend, vlm_model))
            n_captions_indexed = stats["n_captions_indexed"]
            print(f"[api:ingest] multimodal done -> {stats['n_images']} image(s) stored, "
                  f"{n_captions_indexed} caption(s) dual-indexed into the text store")
        elif embedder.supports_images():
            print(f"[api:ingest] embedding {len(image_chunks)} image(s) via {embedder.name}...")
            vectors = embedder.embed_images([c.image_path for c in image_chunks])
            store.upsert(
                ids=[c.doc_id for c in image_chunks],
                vectors=vectors,
                texts=["" for _ in image_chunks],
                metadatas=[{**c.metadata, "image_path": c.image_path} for c in image_chunks],
            )
            print(f"[api:ingest] stored -> collection now has {store.count()} total vector(s)")
        else:
            print(f"[api:ingest] {len(image_chunks)} image(s) found but {embedder.name} doesn't support "
                  f"them — skipped (pass multimodal=true, or set RAG_EMBEDDER=clip)")
            logger.warning("images found but current embedder doesn't support them "
                            "(pass multimodal=true, or set RAG_EMBEDDER=clip)",
                            extra={"n_images": len(image_chunks), "embedder": embedder.name})

    n_pages_flagged = 0
    n_descriptions_indexed = 0
    if multimodal:
        # Strategy 3 — independent of image_chunks: a page can be flagged
        # as visually complex purely from vector drawings, with zero
        # embedded raster images. ingest_path() already ran the cheap
        # describe_pages=page_vlm heuristic, so pages_needing_description
        # is only ever the pre-filtered, already-flagged subset.
        pages_needing_description = [d for d in raw_docs if d.modality == "pdf_text"
                                      and d.metadata.get("page_image_path")]
        if pages_needing_description:
            print(f"[api:ingest] multimodal: VLM-describing {len(pages_needing_description)} "
                  f"visually-complex page(s) (backend={vlm_backend})...")
            page_stats = describe_complex_pages(pages_needing_description, embedder, store,
                                                 vlm_backend=vlm_backend, vlm_model=vlm_model,
                                                 vlm=_get_caption_vlm(vlm_backend, vlm_model))
            n_pages_flagged = page_stats["n_pages_flagged"]
            n_descriptions_indexed = page_stats["n_descriptions_indexed"]
            print(f"[api:ingest] page description done -> {n_descriptions_indexed} description(s) "
                  f"dual-indexed into the text store")

    logger.info("ingested via API", extra={"uploaded_filename": file.filename, "n_chunks": len(all_chunks)})
    print(f"[api:ingest] done -> {file.filename!r}: {len(all_chunks)} chunk(s) total\n")

    return IngestResponse(
        filename=file.filename,
        n_raw_documents=len(raw_docs),
        n_chunks=len(all_chunks),
        n_text_chunks=len(text_chunks),
        n_image_chunks=len(image_chunks),
        n_flagged_injection=n_flagged,
        n_pii_redacted=n_redacted,
        n_table_chunks=n_table_chunks,
        n_pages_flagged_for_description=n_pages_flagged,
        n_page_descriptions_indexed=n_descriptions_indexed,
        n_captions_indexed=n_captions_indexed,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest_document(
    file: UploadFile = File(...),
    redact_pii: bool = Query(False, description="Scrub emails/phones/SSNs/credit cards/IPs before storage"),
    scan_injection: bool = Query(False, description="Flag chunks that look like prompt-injection attempts"),
    parent_child: bool = Query(False, description="Store small child chunks; generate from larger parents later"),
    chunking: str = Query("auto", description="auto | fixed_size | recursive | sentence_based | semantic | "
                                                "structure_aware. Ignored if parent_child=true. 'semantic' reuses "
                                                "this request's embedder to find topic breaks."),
    multimodal: bool = Query(False, description="Images go through CLIP into their own collection (regardless "
                                                  "of RAG_EMBEDDER), each gets a VLM caption dual-indexed into "
                                                  "the TEXT store — see generation/dual_modality_generator.py."),
    vlm_backend: str = Query("ollama", description="Only used when multimodal=true (captions each image)"),
    vlm_model: Optional[str] = Query(None, description="Only used when multimodal=true "
                                                         "(defaults to llava/moondream2 per backend)"),
    page_vlm: str = Query("auto", description="Strategy 3 (whole-page VLM description, PDFs only). "
                                                "'auto' (default): a cheap, local, no-model-call heuristic flags "
                                                "only pages that look chart/diagram-heavy (vector-drawing-dense, "
                                                "low native text) — see ingestion/ingest_pdf.py. 'always' flags "
                                                "every page (slow — one VLM call per page). 'never' disables the "
                                                "heuristic and flagging entirely. The VLM call itself only runs "
                                                "when multimodal=true is also passed, and only for flagged pages, "
                                                "reusing the same VLM instance as image captioning."),
):
    """
    Upload any supported document (.txt, .md, .pdf, .png/.jpg/...) and ingest
    it end-to-end: parse -> chunk -> embed -> store.

    Scanned / image-only PDF pages are handled automatically by
    ingestion/ingest_pdf.py's OCR fallback (ocr_on_empty_pages=True by
    default) — no flag needed here, and a page is never silently returned
    as empty text: if there's no native text layer, OCR runs on that page.

    For MULTIPLE files in one request, see POST /ingest/batch below --
    this route stays single-file-only so an existing integration built
    against it keeps working exactly as it always has.
    """
    _validate_ingest_params(chunking, page_vlm)
    return _ingest_one_file(
        file, redact_pii=redact_pii, scan_injection=scan_injection, parent_child=parent_child,
        chunking=chunking, multimodal=multimodal, vlm_backend=vlm_backend, vlm_model=vlm_model,
        page_vlm=page_vlm,
    )


@app.post("/ingest/batch", response_model=BatchIngestResponse)
def ingest_documents_batch(
    files: List[UploadFile] = File(..., description="Two or more files -- e.g. several PDFs -- in one "
                                                       "multipart request, field name 'files' repeated once "
                                                       "per file."),
    redact_pii: bool = Query(False, description="Scrub emails/phones/SSNs/credit cards/IPs before storage"),
    scan_injection: bool = Query(False, description="Flag chunks that look like prompt-injection attempts"),
    parent_child: bool = Query(False, description="Store small child chunks; generate from larger parents later"),
    chunking: str = Query("auto", description="auto | fixed_size | recursive | sentence_based | semantic | "
                                                "structure_aware. Ignored if parent_child=true."),
    multimodal: bool = Query(False, description="Images go through CLIP into their own collection, each gets "
                                                  "a VLM caption dual-indexed into the TEXT store."),
    vlm_backend: str = Query("ollama", description="Only used when multimodal=true"),
    vlm_model: Optional[str] = Query(None, description="Only used when multimodal=true"),
    page_vlm: str = Query("auto", description="Strategy 3 (whole-page VLM description, PDFs only) -- "
                                                "auto | always | never, see /ingest's own description"),
):
    """
    Same pipeline as POST /ingest (parse -> chunk -> embed -> store),
    run once per file for MULTIPLE files in a single request -- e.g.
    ingesting a whole folder of PDFs in one call instead of one HTTP
    round trip per file. All query params above apply UNIFORMLY to
    every file in the batch (there's no per-file override) -- send
    separate requests if different files genuinely need different
    settings.

    ONE BAD FILE NEVER ABORTS THE REST OF THE BATCH: each file is
    ingested independently, and a parse failure / no-extractable-content
    error / any other unexpected exception for ONE file is caught and
    recorded as that file's own `error` entry, while every other file in
    the same request still gets a normal shot at ingesting successfully.
    Check each entry's own `ok` field rather than assuming the whole
    batch either fully succeeded or fully failed -- a 200 response here
    can still contain individual failures; see `n_failed` for a quick
    count without walking the whole `files` list.

    Example (curl, three PDFs at once):
        curl -X POST "http://localhost:8000/ingest/batch" \\
            -F "files=@doc1.pdf" -F "files=@doc2.pdf" -F "files=@doc3.pdf"

    For a real browse-and-select-multiple-files experience instead of curl
    or Swagger UI's own file widget (List[UploadFile] renders unreliably
    there -- see GET /ingest/batch/ui's own docstring), open
    GET /ingest/batch/ui in a browser instead.
    """
    _validate_ingest_params(chunking, page_vlm)

    results: List[BatchIngestFileResult] = []
    for f in files:
        try:
            one = _ingest_one_file(
                f, redact_pii=redact_pii, scan_injection=scan_injection, parent_child=parent_child,
                chunking=chunking, multimodal=multimodal, vlm_backend=vlm_backend, vlm_model=vlm_model,
                page_vlm=page_vlm,
            )
            results.append(BatchIngestFileResult(filename=f.filename, ok=True, result=one))
        except HTTPException as e:
            print(f"[api:ingest_batch] {f.filename!r} FAILED: {e.detail}")
            results.append(BatchIngestFileResult(filename=f.filename, ok=False, error=str(e.detail)))
        except Exception as e:  # noqa: BLE001 -- one file's own unexpected failure must never abort the rest of the batch
            print(f"[api:ingest_batch] {f.filename!r} FAILED unexpectedly: {type(e).__name__}: {e}")
            logger.exception("unexpected failure ingesting one file in a batch",
                              extra={"uploaded_filename": f.filename})
            results.append(BatchIngestFileResult(
                filename=f.filename, ok=False,
                error=f"Unexpected error during ingestion ({type(e).__name__}) -- check server logs.",
            ))

    n_succeeded = sum(1 for r in results if r.ok)
    print(f"[api:ingest_batch] done -> {n_succeeded}/{len(files)} file(s) ingested successfully\n")

    return BatchIngestResponse(
        n_files=len(files),
        n_succeeded=n_succeeded,
        n_failed=len(files) - n_succeeded,
        files=results,
    )


@app.get("/ingest/batch/ui", response_class=HTMLResponse, include_in_schema=False)
def ingest_batch_ui():
    """
    A real "browse and select multiple PDFs" page for POST /ingest/batch
    above -- open this URL directly in a browser
    (http://localhost:8000/ingest/batch/ui, adjust host/port to match how
    you're running this service).

    WHY THIS EXISTS instead of just using Swagger UI's own "Try it out"
    button on /ingest/batch: `files: List[UploadFile] = File(...)` is the
    functionally correct, fully-working way to accept multiple files in
    FastAPI (curl with several -F "files=@..." flags works fine, and so
    does a real HTML form -- both proven by this project's own /ingest/batch
    docstring example and by this very page), but Swagger UI's own
    auto-generated multi-file widget is unreliably broken on top of it --
    a long-standing, well-documented Swagger UI limitation, not a bug in
    this project's own code: OpenAPI's "array of binary files" shape has
    never had clean, consistent multi-file-picker support across Swagger
    UI versions (see e.g. fastapi/fastapi discussions #8306 and #9497,
    and swagger-api/swagger-spec issue #254 from the underlying spec
    limitation itself) -- in various FastAPI/Swagger UI version
    combinations it's rendered as a single add-one-string-at-a-time
    control instead of a real "choose files" browse dialog, or dropped
    the file picker for a plain text box entirely. A plain HTML
    `<input type="file" multiple>`, by contrast, has reliably supported
    a real OS-native "select several files at once" dialog in every
    browser for over a decade -- this page uses exactly that, with a
    plain `fetch()` POST of the resulting FormData straight to
    POST /ingest/batch (the same route curl or any other client would
    hit), so there's no second code path to keep in sync -- this page is
    a thin client of the real endpoint, not a reimplementation of it.

    include_in_schema=False -- deliberately not listed in
    /docs or /openapi.json: this is a hand-built page for people, not an
    API route other code is meant to call.
    """
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Batch PDF Ingest</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }
  h1 { font-size: 1.4rem; }
  fieldset { border: 1px solid #ccc; border-radius: 6px; margin: 1rem 0; padding: 0.75rem 1rem; }
  legend { padding: 0 0.4rem; color: #555; font-size: 0.9rem; }
  label { display: block; margin: 0.5rem 0 0.2rem; font-size: 0.9rem; }
  input[type=file] { display: block; margin: 0.4rem 0; }
  input[type=text], select { padding: 0.3rem; font-size: 0.9rem; }
  button { margin-top: 1rem; padding: 0.6rem 1.2rem; font-size: 1rem; cursor: pointer;
           background: #2563eb; color: white; border: none; border-radius: 6px; }
  button:disabled { background: #93b4ec; cursor: default; }
  #status { margin-top: 1rem; font-size: 0.9rem; color: #555; }
  table { border-collapse: collapse; width: 100%; margin-top: 1rem; font-size: 0.9rem; }
  th, td { border: 1px solid #ddd; padding: 0.4rem 0.6rem; text-align: left; vertical-align: top; }
  th { background: #f5f5f5; }
  .ok { color: #16813d; font-weight: 600; }
  .fail { color: #b91c1c; font-weight: 600; }
  details { margin-top: 0.5rem; }
  .file-list { font-size: 0.85rem; color: #555; margin-top: 0.3rem; }
</style>
</head>
<body>
<h1>Batch PDF Ingest</h1>
<p>Pick several PDFs at once (the OS file dialog's own multi-select -- Ctrl/Cmd-click or Shift-click), then submit. Posts straight to <code>POST /ingest/batch</code>.</p>

<form id="batchForm">
  <input type="file" id="files" name="files" multiple accept="application/pdf,.pdf" required>
  <div class="file-list" id="fileList"></div>

  <fieldset>
    <legend>Options</legend>
    <label><input type="checkbox" id="multimodal"> multimodal (VLM-caption embedded images, dual-index into text store)</label>

    <details>
      <summary>Advanced</summary>
      <label for="chunking">chunking</label>
      <select id="chunking">
        <option value="auto" selected>auto</option>
        <option value="fixed_size">fixed_size</option>
        <option value="recursive">recursive</option>
        <option value="sentence_based">sentence_based</option>
        <option value="semantic">semantic</option>
        <option value="structure_aware">structure_aware</option>
      </select>

      <label for="page_vlm">page_vlm (Strategy 3, PDFs only)</label>
      <select id="page_vlm">
        <option value="auto" selected>auto</option>
        <option value="always">always</option>
        <option value="never">never</option>
      </select>

      <label for="vlm_backend">vlm_backend (only used if multimodal is checked)</label>
      <select id="vlm_backend">
        <option value="ollama" selected>ollama</option>
        <option value="groq">groq</option>
        <option value="hf">hf</option>
      </select>

      <label for="vlm_model">vlm_model (blank = backend default)</label>
      <input type="text" id="vlm_model" placeholder="">

      <label><input type="checkbox" id="redact_pii"> redact_pii</label>
      <label><input type="checkbox" id="scan_injection"> scan_injection</label>
      <label><input type="checkbox" id="parent_child"> parent_child</label>
    </details>
  </fieldset>

  <button type="submit" id="submitBtn">Ingest</button>
</form>

<div id="status"></div>
<div id="results"></div>

<script>
const filesInput = document.getElementById('files');
const fileList = document.getElementById('fileList');
filesInput.addEventListener('change', () => {
  const names = Array.from(filesInput.files).map(f => f.name);
  fileList.textContent = names.length ? `${names.length} file(s): ${names.join(', ')}` : '';
});

document.getElementById('batchForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const files = filesInput.files;
  if (!files.length) return;

  const submitBtn = document.getElementById('submitBtn');
  const status = document.getElementById('status');
  const results = document.getElementById('results');
  submitBtn.disabled = true;
  results.innerHTML = '';
  status.textContent = `Uploading and ingesting ${files.length} file(s)... this can take a while for large PDFs or multimodal=true.`;

  const params = new URLSearchParams({
    redact_pii: document.getElementById('redact_pii').checked,
    scan_injection: document.getElementById('scan_injection').checked,
    parent_child: document.getElementById('parent_child').checked,
    chunking: document.getElementById('chunking').value,
    multimodal: document.getElementById('multimodal').checked,
    vlm_backend: document.getElementById('vlm_backend').value,
    page_vlm: document.getElementById('page_vlm').value,
  });
  const vlmModel = document.getElementById('vlm_model').value.trim();
  if (vlmModel) params.set('vlm_model', vlmModel);

  const formData = new FormData();
  for (const f of files) formData.append('files', f);

  try {
    const resp = await fetch(`/ingest/batch?${params.toString()}`, { method: 'POST', body: formData });
    const data = await resp.json();
    if (!resp.ok) {
      status.textContent = `Request failed (HTTP ${resp.status}): ${data.detail || JSON.stringify(data)}`;
      submitBtn.disabled = false;
      return;
    }
    status.textContent = `Done: ${data.n_succeeded}/${data.n_files} succeeded, ${data.n_failed} failed.`;

    let html = '<table><tr><th>File</th><th>Status</th><th>Details</th></tr>';
    for (const r of data.files) {
      if (r.ok) {
        const res = r.result;
        html += `<tr><td>${r.filename}</td><td class="ok">OK</td>` +
          `<td>${res.n_chunks} chunk(s) (${res.n_text_chunks} text, ${res.n_image_chunks} image` +
          (res.n_captions_indexed ? `, ${res.n_captions_indexed} caption(s) dual-indexed` : '') +
          `)</td></tr>`;
      } else {
        html += `<tr><td>${r.filename}</td><td class="fail">FAILED</td><td>${r.error}</td></tr>`;
      }
    }
    html += '</table>';
    results.innerHTML = html;
  } catch (err) {
    status.textContent = `Request failed: ${err}`;
  } finally {
    submitBtn.disabled = false;
  }
});
</script>
</body>
</html>"""


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Ask a question against whatever is currently indexed; get a grounded,
    source-attributed answer back."""
    embedder = _state["embedder"]
    store = _state["store"]

    print(f"\n[api:query] question={req.question!r} retrieval={req.retrieval!r} "
          f"top_k={req.top_k} rerank={req.rerank} multimodal={req.multimodal}")

    if store.count() == 0 and not req.multimodal:
        print("[api:query] index is empty — aborting")
        raise HTTPException(status_code=409, detail="Index is empty — POST a document to /ingest first.")

    generator = _get_generator()

    if req.retrieval == "vector":
        from retrieval.vector_retriever import vector_retrieve
        results = vector_retrieve(req.question, embedder, store, top_k=req.top_k)
        strategy_used = "vector"

    elif req.retrieval == "hybrid":
        from retrieval.hybrid_retriever import hybrid_retrieve
        results = hybrid_retrieve(req.question, embedder, store, top_k=req.top_k)
        strategy_used = "hybrid"

    elif req.retrieval == "router":
        from retrieval.query_router import rule_based_route, route_and_retrieve
        decision_preview = rule_based_route(req.question)
        corpus = store.get_all() if decision_preview.route == "keyword_hybrid" else None
        results, decision = route_and_retrieve(req.question, embedder, store,
                                                corpus_for_hybrid=corpus, top_k=req.top_k)
        strategy_used = f"router->{decision.route}"

    elif req.retrieval == "multi_query":
        from retrieval.multi_query import multi_query_retrieve
        results = multi_query_retrieve(req.question, embedder, store, generator, top_k_final=req.top_k)
        strategy_used = "multi_query"

    else:
        print(f"[api:query] unknown retrieval strategy {req.retrieval!r} — aborting")
        raise HTTPException(status_code=400, detail=f"Unknown retrieval strategy: {req.retrieval!r} "
                                                      f"(use vector | hybrid | router | multi_query)")

    print(f"[api:query] retrieve ({strategy_used}) -> {len(results)} result(s)")

    if req.rerank:
        results = _get_reranker().rerank(req.question, results, top_k=req.top_k)
        print(f"[api:query] rerank -> {len(results)} result(s) after cross-encoder scoring")

    if req.parent_child:
        parents_by_id_raw = pipeline_mod._load_parents_store()
        parents_lookup = {
            pid: Chunk.new(doc_id="", text=p["text"], **p["metadata"])
            for pid, p in parents_by_id_raw.items()
        }
        results = resolve_to_parents(results, parents_lookup)
        print(f"[api:query] parent_child -> resolved to {len(results)} parent chunk(s)")

    if req.compress:
        from retrieval.contextual_compression import compress_retrieved_chunks
        results = compress_retrieved_chunks(req.question, results, generator)
        print(f"[api:query] compress -> {len(results)} chunk(s) remain after contextual compression")

    if req.multimodal:
        # Added AFTER rerank/parent_child/compress on purpose — see
        # pipeline.retrieve_multimodal()'s docstring.
        n_before = len(results)
        results = retrieve_multimodal(req.question, STORE_NAME, results, top_k=req.top_k,
                                       clip_embedder=_get_clip_embedder(), image_store=_get_image_store())
        strategy_used = f"{strategy_used}+images"
        generator = _get_dual_generator(req.vlm_backend, req.vlm_model)
        print(f"[api:query] multimodal image branch -> +{len(results) - n_before} image result(s), "
              f"generator=dual({generator.name})")

    if not results:
        print("[api:query] no results retrieved — aborting")
        raise HTTPException(status_code=404, detail="No relevant chunks were retrieved for this question.")

    print(f"[api:query] generating answer with {generator.name}...")
    answer = generator.generate(req.question, results)
    print(f"[api:query] done -> answer: {answer[:120]!r}{'...' if len(answer) > 120 else ''}\n")

    sources = [
        SourceChunk(
            id=str(r.get("id", "")),
            text=r["text"][:500],
            score=float(r.get("rerank_score", r.get("score", 0.0))),
            filename=r.get("metadata", {}).get("filename"),
            page=r.get("metadata", {}).get("page"),
            # CONFIRMED bug this fixes, not a hypothetical one: this used to
            # only reveal image_path when retrieval_branch == "image" (a
            # multimodal=true CLIP-branch hit) -- but pipeline.
            # build_caption_chunks() ALSO stores a real image_path in the
            # metadata of every dual-indexed image-caption chunk it writes
            # into the TEXT store (see that function's own docstring), and
            # those chunks are retrievable by a completely ordinary text
            # query, multimodal=true or not. The old condition silently
            # nulled out image_path for exactly that case -- the one way
            # to confirm "a plain text query found this image via its
            # surrounding page text" -- making that impossible to verify
            # from this response alone even though the underlying hit was
            # real. Now shows image_path whenever the chunk's own metadata
            # actually has one, regardless of which branch retrieved it.
            image_path=r.get("metadata", {}).get("image_path"),
            source_type=r.get("metadata", {}).get("source_type"),
        )
        for r in results
    ]

    return QueryResponse(
        question=req.question,
        answer=answer,
        retrieval_strategy_used=strategy_used,
        sources=sources,
    )


if __name__ == "__main__":
    # CONFIRMED ROOT CAUSE of "py api.py just prints a DeprecationWarning
    # and a ResourceTracker traceback, then exits": this file defines
    # `app` and its routes but never actually starts a server -- the
    # module docstring's own "Run:" line (`uvicorn api:app --reload
    # --port 8001`) is the ONLY way this was runnable before this block
    # existed. Running `python api.py` directly just imports the module
    # top to bottom, reaches end-of-file with nothing left to execute,
    # and the interpreter exits normally -- `_startup()` above (the
    # actual embedder/store loading) never runs, since nothing ever
    # drove FastAPI's ASGI lifespan to call it. The DeprecationWarning
    # fires at decoration time (import), before that; the
    # "Exception ignored in: <function ResourceTracker.__del__ ...>
    # AttributeError: '_thread.RLock' object has no attribute
    # '_recursion_count'" right after it is just normal Python 3.12
    # process-exit GC noise from the `multiprocess` package (a
    # transitive dependency pulled in around here, likely via
    # sentence-transformers/datasets) hitting a known, harmless,
    # non-fatal multiprocess/Python-3.12 incompatibility in that
    # package's own __del__ cleanup -- it looks alarming but it is a
    # symptom of the process ending, not the cause. Nothing about it is
    # specific to this project.
    #
    # This block makes `python api.py` do what the docstring's uvicorn
    # command does, so both ways of running this file actually start a
    # server. --reload is intentionally NOT set here (it requires
    # uvicorn's own import-string/multiprocess-based reloader, which
    # needs `uvicorn api:app --reload` from the command line, not
    # uvicorn.run() from inside the module being reloaded) -- use the
    # docstring's uvicorn command instead of `python api.py` during
    # active development if you want autoreload on save.
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
