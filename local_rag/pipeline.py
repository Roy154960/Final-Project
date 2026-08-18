"""
End-to-end pipeline: Ingest -> Chunk -> Embed -> Store -> Retrieve -> Generate

Every model used is free and local. This script wires the individual
per-step modules together — including the production-hardening features
(incremental indexing, dedup, PII redaction, injection scanning, parent-
child chunking, embedding cache) and the advanced retrieval techniques
(query routing, multi-query, contextual compression) — behind CLI flags,
so you can compose exactly the configuration you want to test.

Examples:
    # Basic ingest
    python pipeline.py ingest --source data/raw --embedder hf --store chroma

    # Production-hardened ingest: only re-embed changed files, dedup,
    # redact PII, flag injection attempts, cache embeddings
    python pipeline.py ingest --source data/raw --embedder hf --store chroma \
        --incremental --dedup --redact-pii --scan-injection --cache

    # Parent-child chunking (small chunks retrieved, larger parents generated from)
    python pipeline.py ingest --source data/raw --embedder hf --store chroma --parent-child

    # Advanced ask: route the query, expand it, rerank, then compress context
    python pipeline.py ask "What does the report say about Q3 revenue?" \
        --embedder hf --store chroma --retrieval router --rerank --compress --generator ollama

    # Wipe the corpus for a fresh start (prompts for confirmation)
    python pipeline.py clear --all
"""

import argparse
import json
from pathlib import Path
from typing import Optional

from ingestion.loader import ingest_path
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
from embeddings.base import BaseEmbedder
from vectorstore.base import BaseVectorStore
from ingestion.schema import RawDocument, Chunk
from utils.logging_config import get_logger
from utils.tracing import trace_span
from config import DATA_DIR

logger = get_logger("local_rag.pipeline")

PARENTS_STORE_PATH = DATA_DIR / "parents_store.json"


def get_embedder(name: str, use_cache: bool = False) -> BaseEmbedder:
    if name == "hf":
        from embeddings.hf_embedder import HFEmbedder
        embedder = HFEmbedder()
    elif name == "ollama":
        from embeddings.ollama_embedder import OllamaEmbedder
        embedder = OllamaEmbedder()
    elif name == "clip":
        from embeddings.clip_embedder import ClipEmbedder
        embedder = ClipEmbedder()
    else:
        raise ValueError(f"Unknown embedder: {name}")

    if use_cache:
        from embeddings.cache import CachedEmbedder
        embedder = CachedEmbedder(embedder)
    return embedder


def get_store(name: str, dimensions: int, collection_name: Optional[str] = None) -> BaseVectorStore:
    kwargs = {"collection_name": collection_name} if collection_name else {}
    if name == "chroma":
        from vectorstore.chroma_store import ChromaStore
        return ChromaStore(**kwargs)
    if name == "qdrant":
        from vectorstore.qdrant_store import QdrantStore
        return QdrantStore(dimensions=dimensions, **kwargs)
    raise ValueError(f"Unknown store: {name}")


def get_image_store(store_name: str, dimensions: int) -> BaseVectorStore:
    """Separate vector-store collection dedicated to CLIP image embeddings,
    kept apart from get_store()'s default text collection — see
    config.CHROMA_IMAGE_COLLECTION / QDRANT_IMAGE_COLLECTION and
    generation/dual_modality_generator.py for why."""
    from config import CHROMA_IMAGE_COLLECTION, QDRANT_IMAGE_COLLECTION
    collection = CHROMA_IMAGE_COLLECTION if store_name == "chroma" else QDRANT_IMAGE_COLLECTION
    return get_store(store_name, dimensions, collection_name=collection)


def build_caption_chunks(image_docs: list[RawDocument], captions: list[str]) -> list[Chunk]:
    """
    Pairs each image RawDocument with its VLM caption (see
    ingestion/image_captioning.py) and returns Chunk objects ready to embed
    via the TEXT embedder and dual-index into the TEXT store.

    The chunk's EMBEDDED text is nearby_text (a small snippet of whatever
    page text sits closest to the image -- ingestion/ingest_pdf.py's own
    _nearby_page_text already stashed that on each image RawDocument's own
    metadata["nearby_text"] at extraction time, empty/absent for images
    ingested via ingest_image.py's standalone-file path, which has no
    "page" for anything to be near) FOLLOWED BY the caption. A short VLM
    caption alone ("a bowl of fruit") often lacks the vocabulary a person
    actually searches with ("still life," "chiaroscuro," "underpainting")
    that the surrounding prose usually has -- appending the caption here
    means this image can be found by either.

    ORDER IS LOAD-BEARING, not stylistic -- CONFIRMED live-run failure,
    not a hypothetical one: this used to be `f"{cap} {nearby_text}"`
    (caption first). A real query matched an image's nearby_text
    EXACTLY, verbatim ("Political Structure: A Power Game"), and the
    image still didn't appear anywhere in the top 5 results. Root cause:
    this project's default text embedder
    (sentence-transformers/all-MiniLM-L6-v2, see embeddings/hf_embedder.py
    -- no max_seq_length override anywhere, so it uses the model's own
    default) silently truncates any input past 256 tokens. This
    project's own VLM captions routinely run several hundred words in a
    structured, multi-section format (see ingestion/image_captioning.py's
    own prompt -- "Subject Matter:", "Composition & Layout:", "Style:",
    "Colors:", ...), comfortably exceeding that limit on their own,
    BEFORE nearby_text was ever appended. With caption-first ordering,
    nearby_text -- almost always much shorter, and often the MORE
    discriminative, exact-phrase-searchable part of the two -- was being
    silently cut off by the embedder before it was ever seen at all, not
    just down-weighted by dilution. Putting nearby_text first means any
    truncation now eats into the tail of the (typically longer, more
    redundant) caption instead, so the short, precise, and often
    exact-match-searched part always survives being embedded, no matter
    how long the caption itself runs.

    The DISPLAYED caption -- chunk.metadata["caption"], what
    generate_answer's own citation and format_markdown_image actually
    show the person -- stays the VLM's caption alone, unchanged, even
    though chunk.text (what actually gets embedded and searched) may be
    longer. Set explicitly after Chunk.new() below rather than left to
    whatever **doc.metadata happened to carry, so this stays true even if
    doc.metadata's own "caption" key (there isn't one today, but nothing
    stops one being added later) ever drifted from `cap`.

    This is the ONLY place that builds these chunks — api.py and stages.py
    both call this instead of rebuilding it, on purpose: Chunk.new()'s
    `modality=`/`image_path=` are named parameters (they become dataclass
    fields), NOT metadata dict entries, so passing them through **{...}
    silently drops them from what actually reaches store.upsert(metadatas=...)
    — a real bug caught while first writing this. Setting .metadata
    explicitly after construction, in one shared function, means that fix
    can't quietly get lost in a second, slightly different reimplementation.
    """
    chunks: list[Chunk] = []
    for doc, cap in zip(image_docs, captions):
        if not cap:
            continue
        nearby_text = (doc.metadata.get("nearby_text") or "").strip()
        embed_text = f"{nearby_text} {cap}".strip() if nearby_text else cap
        chunk = Chunk.new(doc_id=doc.doc_id, text=embed_text, **doc.metadata)
        chunk.metadata["source_type"] = "image_caption"
        chunk.metadata["image_path"] = doc.image_path
        chunk.metadata["caption"] = cap
        chunks.append(chunk)
    return chunks


def ingest_images_multimodal(image_docs: list[RawDocument], text_embedder: BaseEmbedder,
                              text_store: BaseVectorStore, store_name: str,
                              vlm_backend: str = "ollama", vlm_model: Optional[str] = None,
                              clip_embedder: Optional[BaseEmbedder] = None,
                              image_store: Optional[BaseVectorStore] = None, vlm=None) -> dict[str, int]:
    """
    Shared by pipeline.py's cmd_ingest and api.py's /ingest — given a batch
    of image RawDocuments, in one call: (a) CLIP-embeds and stores them into
    their own collection (get_image_store), and (b) VLM-captions each one
    and dual-indexes the caption into the TEXT store via text_embedder /
    text_store. See generation/dual_modality_generator.py for how the two
    come back together at query time.

    clip_embedder / image_store / vlm: pass already-loaded instances to skip
    loading CLIP and reconnecting the image store from scratch — matters a
    lot for a long-running caller like api.py, which would otherwise reload
    the CLIP model (a real, multi-second, in-process torch/open_clip load)
    on every single /ingest call, unlike pipeline.py's CLI, where each stage
    is a short-lived process that only ever calls this once anyway. Left as
    None here (unchanged default), the function loads its own exactly as
    before — pipeline.py's cmd_ingest doesn't pass these and isn't affected.

    stages.py does NOT call this directly, since its embed/store steps are
    split across separate checkpointed processes — it calls the smaller
    pieces this is built from (get_embedder("clip"), get_image_store(),
    ingestion.image_captioning.caption_images(), build_caption_chunks())
    individually at the right stage boundary instead. Both paths share
    build_caption_chunks() either way.
    """
    from ingestion.image_captioning import caption_images

    if not image_docs:
        return {"n_images": 0, "n_captions_indexed": 0}

    if clip_embedder is None:
        clip_embedder = get_embedder("clip")
    if image_store is None:
        image_store = get_image_store(store_name, clip_embedder.dimensions)

    captions = caption_images(image_docs, vlm_backend=vlm_backend, vlm_model=vlm_model, vlm=vlm)

    image_vectors = clip_embedder.embed_images([d.image_path for d in image_docs])
    image_store.upsert(
        ids=[d.doc_id for d in image_docs],
        vectors=image_vectors,
        texts=["" for _ in image_docs],
        metadatas=[{**d.metadata, "image_path": d.image_path, "caption": cap}
                   for d, cap in zip(image_docs, captions)],
    )

    caption_chunks = build_caption_chunks(image_docs, captions)
    if caption_chunks:
        caption_vectors = text_embedder.embed_texts([c.text for c in caption_chunks])
        text_store.upsert(ids=[c.chunk_id for c in caption_chunks], vectors=caption_vectors,
                           texts=[c.text for c in caption_chunks],
                           metadatas=[c.metadata for c in caption_chunks])

    return {"n_images": len(image_docs), "n_captions_indexed": len(caption_chunks)}


def build_page_description_chunks(page_docs: list[RawDocument], descriptions: list[str]) -> list[Chunk]:
    """
    Pairs each VLM-flagged page RawDocument (metadata["page_image_path"]
    set by ingestion/ingest_pdf.py) with its VLM-written whole-page
    description (see ingestion/page_description.py) and returns Chunk
    objects ready to embed via the TEXT embedder and index into the TEXT
    store. This is ADDITIVE alongside that page's own native/OCR text
    chunk (from chunk_pdf_page_as_unit) — never a replacement for it,
    same pattern as build_caption_chunks() above and
    chunk_pdf_table_as_unit() for tables.
    """
    chunks: list[Chunk] = []
    for doc, desc in zip(page_docs, descriptions):
        if not desc:
            continue
        chunk = Chunk.new(doc_id=doc.doc_id, text=desc, **doc.metadata)
        chunk.metadata["source_type"] = "page_visual_description"
        chunks.append(chunk)
    return chunks


def describe_complex_pages(page_docs: list[RawDocument], text_embedder: BaseEmbedder,
                            text_store: BaseVectorStore, vlm_backend: str = "ollama",
                            vlm_model: Optional[str] = None, vlm=None) -> dict[str, int]:
    """
    Shared by pipeline.py's cmd_ingest and api.py's /ingest — the
    downstream (expensive) half of Strategy 3, VLM-describing whole pages.
    See ingestion/page_description.py's module docstring for the full
    cost-control story; in short: ingestion/ingest_pdf.py's cheap,
    no-model-call describe_pages="auto" heuristic has ALREADY decided
    which pages are worth this, flagging them with
    metadata["page_image_path"]. This function filters page_docs down to
    just that flagged subset, so callers can pass the full raw_docs list
    (like ingest_images_multimodal() is passed the full image_docs list)
    without needing to pre-filter themselves.

    Only ever called when --multimodal is passed at the call site (same
    gate as ingest_images_multimodal()) — a user who never opts into
    multimodal ingestion pays zero VLM cost for this, exactly like images.

    vlm: pass an already-loaded VLM instance (e.g. the same one used for
    image captioning) to skip a second model load — see
    ingest_images_multimodal()'s docstring for the same reasoning.
    """
    from ingestion.page_description import describe_pages as _describe_pages

    flagged = [d for d in page_docs if d.modality == "pdf_text" and d.metadata.get("page_image_path")]
    if not flagged:
        return {"n_pages_flagged": 0, "n_descriptions_indexed": 0}

    descriptions = _describe_pages(flagged, vlm_backend=vlm_backend, vlm_model=vlm_model, vlm=vlm)

    description_chunks = build_page_description_chunks(flagged, descriptions)
    if description_chunks:
        vectors = text_embedder.embed_texts([c.text for c in description_chunks])
        text_store.upsert(ids=[c.chunk_id for c in description_chunks], vectors=vectors,
                           texts=[c.text for c in description_chunks],
                           metadatas=[c.metadata for c in description_chunks])

    return {"n_pages_flagged": len(flagged), "n_descriptions_indexed": len(description_chunks)}


def retrieve_multimodal(question: str, store_name: str, text_results: list[dict], top_k: int = 5,
                         clip_embedder: Optional[BaseEmbedder] = None,
                         image_store: Optional[BaseVectorStore] = None) -> list[dict]:
    """
    Shared by pipeline.py's cmd_ask, api.py's /query, and stages.py's
    cmd_retrieve: runs the image-branch retrieval (CLIP text-encoder query
    against the dedicated image store) and tags both branches'
    metadata["retrieval_branch"] so generation/dual_modality_generator.py can
    split them back apart. text_results should already have gone through
    whichever text-only postprocessing (rerank/parent_child/compress) you
    want — those steps are skipped for the image branch on purpose (a
    cross-encoder reranker scoring an image's empty text against the
    question would likely rank it out of top_k).

    clip_embedder / image_store: same reasoning as ingest_images_multimodal()
    above — pass already-loaded instances to avoid reloading CLIP on every
    single /query call.
    """
    from retrieval.image_retriever import retrieve_images_clip

    if clip_embedder is None:
        clip_embedder = get_embedder("clip")
    if image_store is None:
        image_store = get_image_store(store_name, clip_embedder.dimensions)
    image_results = retrieve_images_clip(question, clip_embedder, image_store, top_k=top_k)

    for r in image_results:
        r.setdefault("metadata", {})["retrieval_branch"] = "image"
    for r in text_results:
        r.setdefault("metadata", {})["retrieval_branch"] = "text"
    return text_results + image_results


def get_generator(name: str, vlm_backend: str = "ollama", vlm_model: Optional[str] = None):
    if name == "ollama":
        from generation.ollama_generator import OllamaGenerator
        return OllamaGenerator()
    if name == "hf":
        from generation.hf_generator import HFGenerator
        return HFGenerator()
    if name == "vlm":
        from generation.multimodal_generator import MultimodalGenerator
        default_model = "llava" if vlm_backend == "ollama" else "moondream2"
        return MultimodalGenerator(vlm_backend=vlm_backend, vlm_model=vlm_model or default_model)
    if name == "vllm-server":
        from serving.vllm_openai_server import VLLMServerGenerator
        return VLLMServerGenerator()
    raise ValueError(f"Unknown generator: {name}")


def _load_parents_store() -> dict:
    if PARENTS_STORE_PATH.exists():
        return json.loads(PARENTS_STORE_PATH.read_text())
    return {}


def _save_parents_store(parents: dict) -> None:
    PARENTS_STORE_PATH.write_text(json.dumps(parents, indent=2))


def cmd_ingest(args):
    from ingestion.incremental_indexer import diff_against_manifest, update_manifest_entry, remove_manifest_entry
    from ingestion.deduplication import deduplicate
    from safety.pii_redaction import redact_chunks
    from safety.prompt_injection import flag_suspicious_chunks

    embedder = get_embedder(args.embedder, use_cache=args.cache)
    store = get_store(args.store, embedder.dimensions)

    with trace_span("ingest", source=args.source):
        new_files, changed_files, deleted_paths, unchanged_files = diff_against_manifest(args.source)
        logger.info("diff against manifest computed", extra={
            "new": len(new_files), "changed": len(changed_files),
            "deleted": len(deleted_paths), "unchanged": len(unchanged_files),
        })

        # Deleted files' stale vectors are cleaned up regardless of
        # --incremental -- a file that's gone from --source should never
        # leave orphaned vectors behind no matter which mode is running.
        for deleted_path in deleted_paths:
            stale_chunk_ids = remove_manifest_entry(deleted_path)
            if stale_chunk_ids:
                store.delete(stale_chunk_ids)
            logger.info("removed vectors for deleted file",
                        extra={"path": deleted_path, "n_removed": len(stale_chunk_ids)})

        if args.incremental:
            # Efficiency mode: skip files whose bytes are byte-identical
            # to their last-recorded manifest hash -- see
            # diff_against_manifest's own docstring. Only new_files and
            # changed_files (by file hash) get reprocessed.
            files_to_process = new_files + changed_files
        else:
            # Default (--incremental NOT passed): reprocess EVERY current
            # file, every run -- CONFIRMED real gap this closes. A file's
            # bytes on disk not having changed does NOT mean its
            # EXTRACTED text would come out the same on a re-run: a
            # --force-ocr re-ingest, or any change to extraction/chunking
            # code/config since the last run, produces different text for
            # the SAME source file, which (since chunk_id is derived from
            # content -- see ingestion/schema.py's RawDocument.new/
            # Chunk.new) means different chunk_ids than whatever's
            # already stored under the OLD ones. Those old ids were never
            # being cleaned up outside the --incremental "changed_files"
            # case, so old and new vectors for the same source silently
            # coexisted forever in default mode -- confirmed: re-ingesting
            # a document did NOT reliably overwrite its old embedding.
            # Reprocessing every file here, not just new_files/
            # changed_files, is what actually fixes that, at the cost of
            # this mode always doing full work (unchanged since before --
            # --incremental already existed as the "skip what's
            # byte-identical" opt-in for large corpora that don't need
            # this).
            files_to_process = new_files + changed_files + unchanged_files

        # Stale-chunk cleanup for every file about to be reprocessed --
        # deliberately NOT gated behind --incremental (see the comment
        # above): this is what actually makes re-ingesting a document
        # overwrite its old embedding instead of silently accumulating a
        # second, stale copy alongside it.
        for f in files_to_process:
            stale_chunk_ids = remove_manifest_entry(str(f))
            if stale_chunk_ids:
                store.delete(stale_chunk_ids)
                logger.info("removed stale vectors before re-ingesting",
                            extra={"path": str(f), "n_removed": len(stale_chunk_ids)})

        raw_docs = []
        for f in files_to_process:
            raw_docs.extend(ingest_path(str(f), describe_pages=args.page_vlm, force_ocr=args.force_ocr))

    logger.info("ingested raw documents", extra={"count": len(raw_docs)})

    # Method dispatch for the flat (non-parent-child) chunkers. "semantic" is
    # handled separately below since it needs an embed_fn, unlike the others.
    _chunk_methods = {
        "fixed_size": chunk_fixed_size,
        "recursive": chunk_recursive,
        "sentence_based": chunk_sentence_based,
        "structure_aware": chunk_markdown_by_heading,
    }

    parents_by_id = _load_parents_store() if args.parent_child else {}
    all_chunks = []
    for doc in raw_docs:
        if doc.modality == "image":
            all_chunks.append(doc)  # CLIP embeds the raw image directly, no chunking needed
        elif doc.modality == "pdf_table":
            all_chunks.extend(chunk_pdf_table_as_unit(doc))  # never split a table mid-row
        elif args.parent_child:
            children, doc_parents = build_parent_child_chunks(doc)
            all_chunks.extend(children)
            parents_by_id.update({pid: {"text": p.text, "metadata": p.metadata}
                                   for pid, p in doc_parents.items()})
        elif args.chunking == "auto":
            # Original default behavior: page-as-unit for PDFs (keeps OCR'd
            # pages intact), recursive splitting for everything else.
            if doc.modality == "pdf_text":
                all_chunks.extend(chunk_pdf_page_as_unit(doc))
            else:
                all_chunks.extend(chunk_recursive(doc))
        elif args.chunking == "semantic":
            all_chunks.extend(chunk_semantic(doc, embed_fn=embedder.embed_texts))
        else:
            all_chunks.extend(_chunk_methods[args.chunking](doc))

    if args.parent_child:
        _save_parents_store(parents_by_id)

    logger.info("chunked documents", extra={"n_chunks": len(all_chunks)})

    text_chunks = [c for c in all_chunks if getattr(c, "modality", "text") == "text"]
    image_chunks = [c for c in all_chunks if getattr(c, "modality", "text") == "image"]

    if args.scan_injection and text_chunks:
        flagged = flag_suspicious_chunks(text_chunks)
        if flagged:
            logger.warning("chunks flagged for possible prompt injection",
                            extra={"n_flagged": len(flagged), "examples": flagged[:5]})

    if args.redact_pii and text_chunks:
        text_chunks = redact_chunks(text_chunks)
        n_redacted = sum(1 for c in text_chunks if c.metadata.get("pii_redacted"))
        if n_redacted:
            logger.info("redacted PII from chunks", extra={"n_chunks_redacted": n_redacted})

    if args.dedup and text_chunks:
        before = len(text_chunks)
        embed_fn = embedder.embed_texts if args.dedup_near else None
        text_chunks = deduplicate(text_chunks, embed_fn=embed_fn)
        logger.info("deduplicated chunks", extra={"before": before, "after": len(text_chunks)})

    if text_chunks:
        with trace_span("embed_and_store_text", n_chunks=len(text_chunks)):
            vectors = embedder.embed_texts([c.text for c in text_chunks])
            ids = [getattr(c, "chunk_id", getattr(c, "doc_id", "")) for c in text_chunks]
            store.upsert(ids=ids, vectors=vectors, texts=[c.text for c in text_chunks],
                         metadatas=[c.metadata for c in text_chunks])

            # Manifest writing is UNCONDITIONAL now, not gated behind
            # --incremental -- see this function's own comment above
            # ("Default (--incremental NOT passed): reprocess EVERY
            # current file...") for why. A manifest entry only ever
            # helps the NEXT run find and delete THIS run's chunk_ids
            # before writing fresh ones for the same source path, and
            # that needs to work whether or not --incremental happens to
            # be passed on either run -- a plain, default-mode ingest
            # that never wrote a manifest entry at all is exactly what
            # let stale vectors accumulate silently before this fix.
            #
            # BUG FIX (still applies, unrelated to the above): this used
            # to group by metadata["filename"] (just the basename, e.g.
            # "note.txt") and then re-search raw_docs for a filename
            # match to recover the real path. Two files with the same
            # basename in different subdirectories collide on that key —
            # chunk ids from both get merged together and
            # update_manifest_entry() only ever gets called for whichever
            # file `next(...)` happened to find first, so the OTHER file
            # never gets a manifest entry written at all. On every
            # subsequent run, diff_against_manifest() finds no entry for
            # it, treats it as "new" again, and it gets reprocessed —
            # forever, no matter how many times you run it.
            #
            # Fix: doc_id is unique per RawDocument and every chunker
            # preserves it back to the originating doc (confirmed across
            # fixed_size/recursive/sentence_based/semantic/structure_aware/
            # parent_child), so build the doc_id -> source_path mapping
            # directly from raw_docs instead of re-deriving it by name.
            doc_id_to_source_path = {d.doc_id: d.source_path for d in raw_docs}
            ids_by_source_path: dict = {}
            for c, cid in zip(text_chunks, ids):
                source_path = doc_id_to_source_path.get(c.doc_id)
                if source_path:
                    ids_by_source_path.setdefault(source_path, []).append(cid)
            for source_path, chunk_ids in ids_by_source_path.items():
                update_manifest_entry(source_path, chunk_ids)

        logger.info("stored text chunks", extra={"count": len(text_chunks)})

    if image_chunks and args.multimodal:
        # Dual-index path: images always go through CLIP into their own
        # collection (regardless of which --embedder was chosen for text),
        # and each image's VLM caption is ALSO embedded via the TEXT
        # embedder and dual-indexed into the TEXT store — see
        # ingest_images_multimodal() above and
        # generation/dual_modality_generator.py for how these two indexes
        # come back together at query time.
        with trace_span("multimodal_ingest_images", n_images=len(image_chunks)):
            stats = ingest_images_multimodal(image_chunks, embedder, store, args.store,
                                              vlm_backend=args.vlm_backend, vlm_model=args.vlm_model)
        logger.info("stored images (CLIP) + dual-indexed captions (text store)", extra=stats)
    elif image_chunks and embedder.supports_images():
        with trace_span("embed_and_store_images", n_images=len(image_chunks)):
            image_paths = [c.image_path for c in image_chunks]
            vectors = embedder.embed_images(image_paths)
            store.upsert(
                ids=[getattr(c, "doc_id", "") for c in image_chunks],
                vectors=vectors,
                texts=["" for _ in image_chunks],
                metadatas=[{**c.metadata, "image_path": c.image_path} for c in image_chunks],
            )
        logger.info("stored image chunks", extra={"count": len(image_chunks)})
    elif image_chunks:
        logger.warning("images found but embedder doesn't support them (use --multimodal, or --embedder clip)",
                        extra={"n_images": len(image_chunks), "embedder": args.embedder})

    if args.multimodal:
        # Strategy 3 (whole-page VLM description) — independent of whether
        # this PDF had any embedded raster images at all: a page can be
        # chart-heavy purely via vector drawings. ingest_pdf.py already did
        # the cheap heuristic filtering (describe_pages=args.page_vlm), so
        # this only spends a VLM call on pages actually flagged.
        pages_needing_description = [d for d in raw_docs if d.modality == "pdf_text"
                                      and d.metadata.get("page_image_path")]
        if pages_needing_description:
            with trace_span("multimodal_describe_pages", n_pages=len(pages_needing_description)):
                page_stats = describe_complex_pages(pages_needing_description, embedder, store,
                                                     vlm_backend=args.vlm_backend, vlm_model=args.vlm_model)
            logger.info("described visually-complex pages (VLM) + indexed descriptions (text store)",
                        extra=page_stats)


def cmd_ask(args):
    embedder = get_embedder(args.embedder, use_cache=args.cache)
    store = get_store(args.store, embedder.dimensions)
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

    with trace_span("retrieve", question=args.question, strategy=args.retrieval):
        if args.retrieval == "vector":
            from retrieval.vector_retriever import vector_retrieve
            results = vector_retrieve(args.question, embedder, store, top_k=args.top_k)

        elif args.retrieval == "router":
            from retrieval.query_router import rule_based_route, route_and_retrieve
            # Only pull the full corpus if the router might actually need it
            # (keyword_hybrid route) — avoids an unnecessary get_all() on
            # every semantic/metadata-filter question.
            decision_preview = rule_based_route(args.question)
            corpus = store.get_all() if decision_preview.route == "keyword_hybrid" else None
            results, decision = route_and_retrieve(args.question, embedder, store,
                                                    corpus_for_hybrid=corpus, top_k=args.top_k)
            logger.info("query routed", extra={"route": decision.route, "reason": decision.reason})

        elif args.retrieval == "multi_query":
            from retrieval.multi_query import multi_query_retrieve
            results = multi_query_retrieve(args.question, embedder, store, generator,
                                            top_k_final=args.top_k)

        elif args.retrieval == "hybrid":
            from retrieval.hybrid_retriever import hybrid_retrieve
            results = hybrid_retrieve(args.question, embedder, store, top_k=args.top_k)

        else:
            raise NotImplementedError(f"Retrieval strategy '{args.retrieval}' is not recognized.")

    if args.rerank:
        from retrieval.reranker import Reranker
        with trace_span("rerank", n_candidates=len(results)):
            results = Reranker().rerank(args.question, results, top_k=args.top_k)

    if args.parent_child:
        parents_by_id_raw = _load_parents_store()
        from ingestion.schema import Chunk
        parents_lookup = {
            pid: Chunk.new(doc_id="", text=p["text"], **p["metadata"])
            for pid, p in parents_by_id_raw.items()
        }
        results = resolve_to_parents(results, parents_lookup)

    if args.compress:
        from retrieval.contextual_compression import compress_retrieved_chunks
        with trace_span("compress", n_chunks=len(results)):
            results = compress_retrieved_chunks(args.question, results, generator)

    if args.multimodal:
        # Added AFTER rerank/parent_child/compress on purpose — see
        # retrieve_multimodal()'s docstring above.
        with trace_span("retrieve_images", question=args.question):
            results = retrieve_multimodal(args.question, args.store, results, top_k=args.top_k)

    print(f"\nRetrieved {len(results)} chunk(s):")
    for r in results:
        preview = r["text"][:100] or f"[image: {r.get('metadata', {}).get('image_path', '?')}]"
        print(f"  [{r.get('score', 0):.3f}] {preview}")

    with trace_span("generate", model=generator.name):
        answer = generator.generate(args.question, results)
    print(f"\n--- Answer ({generator.name}) ---\n{answer}")


def _clear_vector_collection(store_name: str, which: str, embedder_choice: str) -> None:
    """
    Deletes then immediately recreates the named ChromaDB/Qdrant
    collection -- leaves a genuinely empty, immediately-usable
    collection behind, not just one whose count() happens to read 0,
    and not a half-deleted, broken one either.

    `which`: "text" (CHROMA_COLLECTION/QDRANT_COLLECTION -- everything
    retrieval_qa and friends search) or "images" (CHROMA_IMAGE_COLLECTION/
    QDRANT_IMAGE_COLLECTION -- the separate CLIP-embedding collection;
    see pipeline.py's own get_image_store docstring for why it's kept
    apart from the text one). Recreating "text" does NOT remove the
    image-caption chunks build_caption_chunks() dual-indexes into the
    TEXT store alongside real prose (see that function's own docstring)
    -- those live in the SAME collection as everything else and are
    cleared right along with it; there's no way to keep "only the
    prose" half of a text-collection clear.

    `embedder_choice`: only actually used for `--store qdrant`, where
    recreating a collection needs a real vector dimensionality up
    front -- Qdrant collections are dimension-fixed at creation time,
    unlike Chroma's (see ChromaStore.__init__: no dimensions argument
    at all, Chroma infers this per-upsert). Loading a real embedder
    just to read its own .dimensions attribute is the SAME thing
    cmd_ingest/cmd_ask already do (see get_store's own call sites
    above) -- reused here rather than hardcoding a dimension number
    that could silently bake in the WRONG size and break every future
    upsert against a freshly "cleared" Qdrant collection. Chroma's own
    path below never loads an embedder at all -- ignored entirely,
    since Chroma doesn't need it.
    """
    from config import (
        CHROMA_COLLECTION, CHROMA_IMAGE_COLLECTION,
        QDRANT_COLLECTION, QDRANT_IMAGE_COLLECTION,
    )

    if store_name == "chroma":
        collection_name = CHROMA_COLLECTION if which == "text" else CHROMA_IMAGE_COLLECTION
        dimensions = 0  # unused by ChromaStore -- see this function's own docstring
    else:
        collection_name = QDRANT_COLLECTION if which == "text" else QDRANT_IMAGE_COLLECTION
        embedder = get_embedder("clip" if which == "images" else embedder_choice)
        dimensions = embedder.dimensions

    store = get_store(store_name, dimensions, collection_name=collection_name)
    before = store.count()
    store._client.delete_collection(collection_name)
    # Recreate through the SAME constructor path (get_store -> ChromaStore/
    # QdrantStore.__init__) every other part of this project already uses
    # to create a collection, rather than re-implementing the creation
    # params (distance metric, vector size) by hand a second time here.
    get_store(store_name, dimensions, collection_name=collection_name)
    print(f"  Cleared {store_name} '{which}' collection ({collection_name}): {before} item(s) removed.")


def _clear_directory(path: Path, label: str) -> None:
    """
    Deletes every FILE inside `path`, recursively, then removes any
    subdirectories left empty by that -- but never the top-level `path`
    itself, so anything elsewhere in this project that assumes
    RAW_DOCS_DIR/CACHE_DIR/PERSONAL_UPLOADS_DIR exists as a directory
    (see config.py's own startup `for d in (...): d.mkdir(...)` loop)
    keeps working immediately afterward without needing that loop to
    run again first.
    """
    path = Path(path)
    if not path.exists():
        print(f"  {label}: nothing to clear ({path} doesn't exist).")
        return
    n_files = 0
    for item in path.rglob("*"):
        if item.is_file():
            item.unlink()
            n_files += 1
    # Deepest-first so a parent directory is only removed after its own
    # now-empty children already have been.
    for item in sorted((p for p in path.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)):
        try:
            item.rmdir()
        except OSError:
            pass  # not actually empty (a hidden/system file survived unlink) -- leave it, not worth failing the whole clear over
    print(f"  Cleared {label} ({path}): {n_files} file(s) removed.")


_CLEAR_TARGET_DESCRIPTIONS = {
    "text": "the main TEXT collection -- every retrieval_qa/multi_hop/painting_lookup chunk, "
            "INCLUDING dual-indexed image captions (see build_caption_chunks)",
    "images": "the separate IMAGE collection -- CLIP embeddings image_qa searches",
    "raw_files": "data/raw -- staged source files/images from past ingestion runs",
    "cache": "the embedding cache -- safe to clear, just means the next ingest recomputes "
             "embeddings instead of reusing cached ones",
    "personal_uploads": "data/personal_uploads -- files people have attached directly into "
                         "chat conversations (personal_docs/personal_rag), separate from the "
                         "main corpus",
}


def cmd_clear(args):
    """
    Empties the local RAG -- deletes and recreates the requested vector
    collection(s), and/or deletes the contents of the requested local
    data directories, so a fresh `ingest` run starts from genuinely
    nothing rather than layering on top of whatever was there before.

    Nothing is cleared with NO flags at all and no --all -- an empty
    selection prints what's available and exits, rather than silently
    doing nothing (easy to misread as "it worked") or guessing a
    destructive default. See _CLEAR_TARGET_DESCRIPTIONS above for
    exactly what each flag/--all actually touches.
    """
    if args.all:
        targets = ["text", "images", "raw_files"]
    else:
        targets = [
            name for name, flag in (
                ("text", args.text), ("images", args.images),
                ("raw_files", args.raw_files), ("cache", args.cache),
                ("personal_uploads", args.personal_uploads),
            ) if flag
        ]

    if not targets:
        print("Nothing selected -- pass --all, or one or more of:")
        for name, desc in _CLEAR_TARGET_DESCRIPTIONS.items():
            print(f"  --{name.replace('_', '-')}: {desc}")
        print("\nRun `python pipeline.py clear --help` for the full flag list. Nothing was deleted.")
        return

    print(f"About to clear ({args.store}):")
    for t in targets:
        print(f"  - {_CLEAR_TARGET_DESCRIPTIONS[t]}")

    if not args.yes:
        confirm = input("\nThis is IRREVERSIBLE. Type 'yes' to continue: ").strip().lower()
        if confirm != "yes":
            print("Cancelled -- nothing was deleted.")
            return

    print()
    if "text" in targets:
        _clear_vector_collection(args.store, "text", args.embedder)
    if "images" in targets:
        _clear_vector_collection(args.store, "images", args.embedder)
    if "raw_files" in targets:
        from config import RAW_DOCS_DIR
        _clear_directory(RAW_DOCS_DIR, "raw source files")
    if "cache" in targets:
        from embeddings.cache import CACHE_DIR
        _clear_directory(CACHE_DIR, "embedding cache")
    if "personal_uploads" in targets:
        from config import PERSONAL_UPLOADS_DIR
        _clear_directory(PERSONAL_UPLOADS_DIR, "personal uploads")

    print("\nDone. Run `python pipeline.py ingest ...` to rebuild from a clean slate.")


def main():
    parser = argparse.ArgumentParser(description="Local multimodal RAG pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest, chunk, embed, and store a folder of documents")
    ingest_parser.add_argument("--source", required=True, help="Folder of mixed docs (txt/md/pdf/images)")
    ingest_parser.add_argument("--embedder", choices=["hf", "ollama", "clip"], default="hf")
    ingest_parser.add_argument("--store", choices=["chroma", "qdrant"], default="chroma")
    ingest_parser.add_argument("--incremental", action="store_true", help="Only (re-)ingest new/changed files")
    ingest_parser.add_argument("--chunking", choices=["auto", "fixed_size", "recursive", "sentence_based",
                                                       "semantic", "structure_aware"], default="auto",
                                help="auto = page-as-unit for PDFs / recursive otherwise (previous hardcoded "
                                     "behavior). Pick one explicitly to actually use fixed_size/sentence_based/"
                                     "semantic/structure_aware — see chunking/benchmark_chunkers.py to compare "
                                     "them on your own corpus first. Ignored if --parent-child is set.")
    ingest_parser.add_argument("--parent-child", dest="parent_child", action="store_true",
                                help="Store small child chunks, generate from larger parents")
    ingest_parser.add_argument("--dedup", action="store_true", help="Remove exact-duplicate chunks")
    ingest_parser.add_argument("--dedup-near", dest="dedup_near", action="store_true",
                                help="Also remove near-duplicate chunks (costs an embedding pass)")
    ingest_parser.add_argument("--redact-pii", dest="redact_pii", action="store_true")
    ingest_parser.add_argument("--scan-injection", dest="scan_injection", action="store_true")
    ingest_parser.add_argument("--cache", action="store_true", help="Cache embeddings on disk")
    ingest_parser.add_argument("--multimodal", action="store_true",
                                help="Images always go through CLIP into their own collection (regardless of "
                                     "--embedder), each gets a VLM caption dual-indexed into the TEXT store — "
                                     "see generation/dual_modality_generator.py. Overrides the plain "
                                     "'--embedder clip embeds everything' behavior for images.")
    ingest_parser.add_argument("--vlm-backend", dest="vlm_backend", choices=["ollama", "hf"], default="ollama",
                                help="Only used when --multimodal (captions each image)")
    ingest_parser.add_argument("--page-vlm", dest="page_vlm", choices=["auto", "always", "never"], default="auto",
                                help="Strategy 3 (whole-page VLM description). 'auto' (default): a cheap, local, "
                                     "no-model-call heuristic flags only pages that look chart/diagram-heavy "
                                     "(vector-drawing-dense, low native text); 'always' flags every page (slow — "
                                     "one VLM call per page); 'never' disables the heuristic and flagging "
                                     "entirely. The VLM call itself only runs when --multimodal is also passed.")
    ingest_parser.add_argument("--vlm-model", dest="vlm_model", default=None,
                                help="Only used when --multimodal (defaults to llava/moondream2 per backend)")
    ingest_parser.add_argument("--force-ocr", dest="force_ocr", action="store_true",
                                help="Skip trusting each PDF page's native text layer entirely and OCR every "
                                     "page instead, even where get_text() returns plenty of text. For a "
                                     "specific, confirmed problem: some PDFs (observed with a Noor-Book.com "
                                     "Arabic PDF, but not specific to that one source) have a real native "
                                     "text layer whose Arabic glyphs are stored in visual/reversed order "
                                     "rather than correct logical reading order -- get_text() faithfully "
                                     "returns that broken order, since the PDF's own content stream has it "
                                     "backwards, not this extraction code. OCR reads the rendered page "
                                     "image instead, sidestepping the broken order entirely. Point --source "
                                     "at just the one affected file/folder, not your whole corpus -- this is "
                                     "much slower and unnecessary on pages that already extract correctly.")
    ingest_parser.set_defaults(func=cmd_ingest)

    ask_parser = subparsers.add_parser("ask", help="Ask a question against the stored index")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--embedder", choices=["hf", "ollama", "clip"], default="hf")
    ask_parser.add_argument("--store", choices=["chroma", "qdrant"], default="chroma")
    ask_parser.add_argument("--retrieval", choices=["vector", "router", "multi_query", "hybrid"], default="vector")
    ask_parser.add_argument("--rerank", action="store_true")
    ask_parser.add_argument("--parent-child", dest="parent_child", action="store_true",
                             help="Resolve retrieved children back to their parent chunks before generation")
    ask_parser.add_argument("--compress", action="store_true", help="Contextually compress chunks before generation")
    ask_parser.add_argument("--multimodal", action="store_true",
                             help="Also retrieve from the CLIP image index (see ingest --multimodal) and answer "
                                  "with generation/dual_modality_generator.py: independent text/image drafts, "
                                  "each dropped if not viable, synthesized if both are. Use with --generator "
                                  "ollama or hf, not vlm.")
    ask_parser.add_argument("--generator", choices=["ollama", "hf", "vlm", "vllm-server"], default="ollama")
    ask_parser.add_argument("--vlm-backend", dest="vlm_backend", choices=["ollama", "hf"], default="ollama",
                             help="Only used when --generator vlm")
    ask_parser.add_argument("--vlm-model", dest="vlm_model", default=None,
                             help="Only used when --generator vlm (defaults to llava/moondream2 per backend)")
    ask_parser.add_argument("--cache", action="store_true", help="Cache embeddings on disk")
    ask_parser.add_argument("--top-k", dest="top_k", type=int, default=5)
    ask_parser.set_defaults(func=cmd_ask)

    clear_parser = subparsers.add_parser(
        "clear", help="Empty the local RAG -- delete and recreate vector collection(s) and/or "
                      "clear local data directories, for a genuinely fresh start"
    )
    clear_parser.add_argument("--store", choices=["chroma", "qdrant"], default="chroma")
    clear_parser.add_argument("--embedder", choices=["hf", "ollama", "clip"], default="hf",
                               help="Only used with --store qdrant, to determine the recreated "
                                    "text collection's own vector size -- see "
                                    "_clear_vector_collection's own docstring. Ignored for "
                                    "--store chroma (not needed there).")
    clear_parser.add_argument("--all", action="store_true",
                               help="Clear text + images + raw_files (the core corpus) -- NOT "
                                    "cache or personal_uploads, see those flags below if you "
                                    "want those too")
    clear_parser.add_argument("--text", action="store_true", help="Clear the main text collection")
    clear_parser.add_argument("--images", action="store_true", help="Clear the CLIP image collection")
    clear_parser.add_argument("--raw-files", dest="raw_files", action="store_true",
                               help="Delete data/raw's staged source files")
    clear_parser.add_argument("--cache", action="store_true", help="Delete the embedding cache")
    clear_parser.add_argument("--personal-uploads", dest="personal_uploads", action="store_true",
                               help="Delete data/personal_uploads -- per-conversation chat "
                                    "attachments, separate from the main corpus")
    clear_parser.add_argument("-y", "--yes", action="store_true",
                               help="Skip the interactive confirmation prompt")
    clear_parser.set_defaults(func=cmd_clear)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
