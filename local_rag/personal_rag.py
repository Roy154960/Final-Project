"""
Personal (per-conversation) RAG -- images, PDFs, and plain text files a
person attaches directly in a chat thread, kept separate from the main corpus
(config.CHROMA_COLLECTION) and from each other.

Everything lives in ONE Chroma collection, config.PERSONAL_RAG_COLLECTION
("temp"). "Separated simply by an ID" (per the feature request this module
exists for) means every chunk this module writes carries the owning
thread_id as plain Chroma metadata (THREAD_ID_METADATA_FIELD below), and
every read or delete filters on that field via Chroma's own `where` --
there is no per-thread collection, no per-thread directory, nothing
heavier than one metadata field. Deleting a thread deletes every chunk
whose metadata thread_id matches (see delete_thread_data), so nothing
uploaded into a conversation outlives that conversation.

Two different processes touch this collection, same as the main corpus
collection already does:
  - agents/api.py's upload endpoint (POST /chat/{thread_id}/upload) calls
    ingest_upload() directly, in-process -- same "HTTP layer calls
    straight into local_rag/" pattern local_rag/api.py's own /ingest
    endpoint already uses for the main corpus.
  - The MCP server subprocess (mcp_server/server.py, spawned once by
    agents/mcp_client.py and kept alive for the life of the agents/api.py
    process) calls search_personal() via its search_personal_documents
    tool, from inside the personal_docs specialist (agents/specialists.py).

That cross-process split matters for one concrete, CONFIRMED reason: a
chromadb PersistentClient/Collection handle opened once and reused across
many calls does not reliably see rows written by a *different* process's
client afterward -- confirmed directly (two plain Python processes against
the same on-disk path: a collection handle opened before a write, then
queried after the write, raised "Error creating hnsw segment reader:
Nothing found on disk" instead of returning the new row; a *fresh*
PersistentClient opened right before each call saw it correctly every
time). mcp_server/server.py's own docstring already documents an adjacent,
looser version of this same staleness for the main corpus's BM25 snapshot
("restart the server after re-ingesting") -- for THIS collection, staleness
isn't an acceptable tradeoff (the whole point is "ask about the file you
just uploaded, in the same conversation, without restarting anything"), so
every function below opens its own fresh ChromaStore rather than caching
one at module scope. A PersistentClient open is cheap (no model weights),
so paying it per call is the right trade here -- unlike HFEmbedder below,
which stays a real, cached, load-once-per-process singleton, since nothing
about embedding a piece of text depends on another process's writes.

Run directly to smoke-test:
    python -m personal_rag data/raw/sample.pdf smoke-thread-id
"""

import base64
import mimetypes
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

from config import PERSONAL_RAG_COLLECTION, PERSONAL_UPLOADS_DIR, PERSONAL_RAG_SINGLE_IMAGE_VLM_BACKEND
from embeddings.hf_embedder import HFEmbedder
from vectorstore.chroma_store import ChromaStore
from ingestion.loader import ingest_path
from ingestion.schema import Chunk, RawDocument
from ingestion.image_captioning import caption_images, load_vlm
from chunking.recursive import chunk_recursive

# Every chunk this module ever writes carries this metadata field -- the
# one thing that separates one thread's uploads from another's within the
# single shared "temp" collection. Every search()/delete_thread_data() call
# filters on it via Chroma's own `where`.
THREAD_ID_METADATA_FIELD = "thread_id"

# What POST /chat/{thread_id}/upload (agents/api.py) accepts. Narrower
# than ingestion/loader.py's own TEXT_EXTS + IMAGE_EXTS + ".pdf" -- this
# feature is specifically "images, PDFs, and plain text files" (per the
# requests it was built for), not a general-purpose document uploader.
# ".txt" is included (per an explicit follow-up request: personal-doc
# uploads should cover plain-text notes/snippets, not just PDFs and
# images); ".md" deliberately is NOT -- ingestion/loader.py's TEXT_EXTS
# treats both identically (see that module's own `if p.suffix.lower() in
# TEXT_EXTS` branch, which reads either one as plain text with no
# markdown-aware parsing), so admitting ".md" here would be free to add
# the same way, but nobody's asked for it yet -- add it to this tuple if
# that changes. Anything else (.docx, .csv, ...) is still rejected with a
# clear reason rather than silently accepted.
SUPPORTED_UPLOAD_EXTS = (".pdf", ".txt", ".png", ".jpg", ".jpeg", ".webp", ".bmp")

# How many images get handed to the VLM per caption_images() call. Mirrors
# the shape of the main corpus's own --multimodal ingest path
# (local_rag/api.py's /ingest -> pipeline.py's ingest_images_multimodal ->
# caption_images(image_docs, ...)), which loads its VLM ONCE
# (_get_caption_vlm, cached for the life of the api.py process) and
# captions every image handed to it in that one call, rather than
# reloading a fresh VLM client per image the way this module's captioning
# loop used to (one caption_images([doc], ...) call per image, each
# starting from vlm=None -- confirmed live: a personal PDF upload with 37
# extracted images produced 37 separate "[caption] captioning 1 image(s)
# via ollama..." log lines instead of a handful of batched ones). Capped
# at 10 rather than "all of them in one call" so one enormous PDF upload
# (hundreds of embedded images) can't build one unbounded list of VLM
# calls before the first caption is even returned -- 10 is a small,
# responsive batch size, not a hard architectural limit.
PERSONAL_CAPTION_BATCH_SIZE = 10

# Cap on how large an uploaded image's base64-encoded payload is allowed
# to be before it gets embedded directly into a chat message as a data:
# URI (see _image_to_data_uri below). Same value and same reasoning as
# mcp_server/image_tools.py's MAX_IMAGE_BYTES_FOR_B64 -- deliberately kept
# in sync by eye (not imported) since this module talks to the pipeline
# directly and has no dependency on mcp_server/ at all (see this module's
# own docstring on why ingest is a direct call, never through MCP).
MAX_IMAGE_BYTES_FOR_B64 = 5 * 1024 * 1024

_embedder: Optional[HFEmbedder] = None


def _get_embedder() -> HFEmbedder:
    """
    Lazy, cached, ONE per process -- loading sentence-transformers weights
    is the expensive part (same reasoning HFEmbedder's every other caller
    in this project already follows), and unlike the ChromaStore handles
    below, nothing about which embedding vector a given piece of text
    produces depends on another process's writes, so caching this one is
    safe. Deliberately the SAME model class (and, since no model_name is
    passed, the SAME default model -- "sentence-transformers/all-MiniLM-
    L6-v2") the main corpus's mcp_server/server.py already loads, so a
    personal-doc chunk and a corpus chunk are directly comparable in the
    same embedding space if this collection is ever queried alongside the
    main one.
    """
    global _embedder
    if _embedder is None:
        _embedder = HFEmbedder()
    return _embedder


def _fresh_store() -> ChromaStore:
    """
    A brand-new ChromaStore (fresh chromadb PersistentClient, freshly
    fetched-or-created collection handle) -- see this module's own
    docstring for the confirmed cross-process staleness this sidesteps.
    Same CHROMA_PERSIST_DIR as the main corpus (ChromaStore's own default),
    different collection name (PERSONAL_RAG_COLLECTION, "temp") -- one
    Chroma database directory on disk, two independent collections inside
    it, exactly like CHROMA_COLLECTION and CHROMA_IMAGE_COLLECTION already
    coexist there for the main multimodal pipeline.
    """
    return ChromaStore(collection_name=PERSONAL_RAG_COLLECTION)


def _caption_image_docs(image_docs: list[RawDocument]) -> list[str]:
    """
    Caption every image RawDocument in ONE batched pass instead of one
    caption_images() call per image -- see PERSONAL_CAPTION_BATCH_SIZE's
    own docstring for the confirmed live-run problem this replaces (a VLM
    client reload per image on a many-image PDF upload). Loads the VLM
    exactly once here and reuses it across every batch of
    PERSONAL_CAPTION_BATCH_SIZE images, the same "load once, caption many"
    shape local_rag/api.py's _get_caption_vlm() + ingest_images_multimodal()
    already give the main corpus's --multimodal ingest path.

    Backend choice: a SINGLE stand-alone image (len(image_docs) == 1 --
    i.e. this upload IS one image, not one of several figures pulled out
    of a PDF) uses config.PERSONAL_RAG_SINGLE_IMAGE_VLM_BACKEND ("groq"
    by default, see that constant's own docstring for why -- latency and
    quality matter more for "explain the one picture I just sent" than
    for bulk ingestion). Anything else -- zero images, or several at once
    from a multi-image PDF -- always uses the local "ollama" VLM
    regardless of that setting, specifically so a many-figure PDF upload
    can never burn through a free hosted API's rate limit.

    Falls back to the local "ollama" VLM automatically if Groq can't be
    used at all (GroqVLM's own call raises if no key is set -- see
    config.py's own docstring) OR fails at caption time (a network
    error, a rate limit, a transient 5xx) -- never lets a hosted-API
    hiccup turn into "this image never got captioned." This is
    genuinely a second, outer layer of the SAME fallback
    vlm/fallback_vlm.py's FallbackVLM already does internally on every
    call -- kept here anyway rather than special-cased away, since one
    extra no-op try/except costs nothing. Logged either way so it's
    obvious from the server's own output which backend actually
    answered.

    Returns one caption string per input doc, same order, "" for any
    image the VLM failed to caption (caption_images() itself never raises
    -- see its own docstring) -- ingest_upload() below turns an empty
    caption into a plain placeholder rather than dropping the image
    entirely.
    """
    if not image_docs:
        return []

    is_single_image = len(image_docs) == 1
    backend = PERSONAL_RAG_SINGLE_IMAGE_VLM_BACKEND if is_single_image else "ollama"

    vlm = None
    if backend == "groq":
        try:
            vlm = load_vlm(backend)
            print(f"[personal_rag] captioning this single image via the online "
                  f"VLM ({vlm.name})", file=sys.stderr)
        except Exception as e:
            print(f"[personal_rag] online VLM unavailable ({e}) -- falling "
                  "back to the local ollama VLM for this image", file=sys.stderr)
            backend = "ollama"

    captions: list[str] = []
    for start in range(0, len(image_docs), PERSONAL_CAPTION_BATCH_SIZE):
        batch = image_docs[start : start + PERSONAL_CAPTION_BATCH_SIZE]
        try:
            captions.extend(caption_images(batch, vlm_backend=backend, vlm=vlm))
        except Exception as e:
            if backend != "ollama":
                # One more safety net: a failure *inside* caption_images()
                # itself (rather than at load_vlm() above) for the online
                # backend -- e.g. every image in this batch individually
                # raised -- degrades to a fresh local attempt rather than
                # leaving this batch entirely uncaptioned.
                print(f"[personal_rag] online VLM captioning failed ({e}) -- "
                      "retrying this batch locally via ollama", file=sys.stderr)
                captions.extend(caption_images(batch, vlm_backend="ollama", vlm=None))
            else:
                raise

    # caption_images() itself never raises per image -- a single failed
    # image (network blip, rate limit, the VLM's own safety filter) comes
    # back as "" rather than an exception (see that function's own
    # docstring). For the single-image case specifically, "" is worth one
    # local retry rather than accepting a blank caption outright: this is
    # the exact "explain this image" moment the online backend was chosen
    # FOR, so it's worth the extra few seconds of a local fallback before
    # giving up on a real caption entirely.
    if is_single_image and backend == "groq" and captions and not captions[0]:
        print("[personal_rag] online VLM returned no caption for this image -- "
              "retrying locally via ollama", file=sys.stderr)
        captions = caption_images(image_docs, vlm_backend="ollama", vlm=None)

    return captions


def _persist_personal_image(thread_id: str, doc: RawDocument) -> Optional[str]:
    """
    Copy an uploaded image's bytes from wherever ingest_path() staged them
    (a temp path under RAW_DOCS_DIR that agents/api.py's upload endpoint
    always deletes once ingestion returns -- see that endpoint's own
    comment) into PERSONAL_UPLOADS_DIR/<thread_id>/, a location that
    outlives the staging copy. This is what makes it possible to actually
    show the person's own uploaded image later -- in the immediate
    upload-confirmation message (agents/api.py) and again on any later
    "what is this image" follow-up (agents/specialists.py's
    personal_docs_node / image_qa_node reading the persisted path back out
    of this chunk's own metadata) -- instead of only ever having a text
    caption with nothing behind it once the staged temp file is gone.

    Named by doc.doc_id (content-derived, see ingestion/schema.py's
    _stable_id) plus the original suffix, so re-uploading the
    byte-identical image into the same thread overwrites the same file
    rather than accumulating duplicates.

    Never raises: a copy failure (disk full, permissions, the staged file
    already gone) degrades to None, and ingest_upload() below still stores
    the caption as text -- a missing image to *show* later is strictly
    less bad than losing the upload's searchable text entirely over a
    filesystem hiccup.
    """
    if not doc.image_path:
        return None
    try:
        src = Path(doc.image_path)
        suffix = src.suffix or ".png"
        dest_dir = PERSONAL_UPLOADS_DIR / thread_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{doc.doc_id}{suffix}"
        shutil.copyfile(src, dest)
        return str(dest)
    except OSError as e:
        print(f"[personal_rag] could not persist uploaded image {doc.image_path!r}: {e!r}",
              file=sys.stderr)
        return None


def _image_to_data_uri(image_path: Optional[str], max_bytes: int = MAX_IMAGE_BYTES_FOR_B64) -> Optional[str]:
    """
    Read a persisted personal-upload image back off disk and encode it as
    a `data:<mime>;base64,...` URI, ready to drop straight into a markdown
    `![caption](data-uri)` block -- the same embedding shape
    mcp_server/image_tools.py's retrieve_images_embedded already uses for
    the main corpus, applied here to a personal upload's own file instead.
    Never raises: a missing/unreadable/oversized file all degrade to None
    (logged, not thrown), the same "one bad image never breaks the whole
    response" contract image_tools.py's own _encode_image_base64 follows.
    """
    if not image_path:
        return None
    path = Path(image_path)
    try:
        if not path.is_file() or path.stat().st_size > max_bytes:
            return None
        raw_bytes = path.read_bytes()
    except OSError:
        return None
    mime_type, _ = mimetypes.guess_type(str(path))
    if not mime_type:
        mime_type = "application/octet-stream"
    return f"data:{mime_type};base64,{base64.b64encode(raw_bytes).decode('ascii')}"


def ingest_upload(thread_id: str, file_path: str, filename: str) -> dict:
    """
    Ingest one uploaded image, PDF, or plain text file into thread_id's
    slice of the "temp" collection. Called directly (in-process) by
    agents/api.py's POST /chat/{thread_id}/upload -- never through the
    MCP server, since this is a deterministic action the HTTP layer
    performs on the person's explicit upload, not a decision a
    specialist's LLM makes (same "ingest is a direct pipeline call,
    retrieval goes through MCP" split local_rag/api.py's own /ingest
    endpoint already draws for the main corpus).

    PDFs and .txt files: both extracted via ingestion.loader.ingest_path
    (describe_pages="never" -- the whole-page VLM description strategy is
    a main-corpus, --multimodal-only feature; a quick personal upload
    doesn't pay for an extra VLM call per visually-complex page), then
    chunked with the same chunk_recursive() the base pipeline defaults
    to. ingest_path() already reads a .txt file as plain text via its own
    TEXT_EXTS branch -- nothing personal_rag-specific is needed to
    support it beyond admitting the extension in SUPPORTED_UPLOAD_EXTS
    and the `modality` line below; a .txt RawDocument's own `.modality`
    comes back "text", so it lands in `other_docs` alongside PDF text and
    is chunked exactly the same way, never mistaken for an image.

    Images: captioned via a VLM, batched PERSONAL_CAPTION_BATCH_SIZE at a
    time (see _caption_image_docs) rather than CLIP-embedded -- this
    collection is text-only, see this module's own docstring. Each
    image's bytes are ALSO persisted (see _persist_personal_image) and
    the resulting path stashed on the chunk's own metadata
    ("image_path", plus "original_modality": "image") so a later reader
    -- the immediate upload-confirmation message this function's caller
    builds, or a follow-up question's personal_docs_node/image_qa_node --
    can show the actual picture the person sent, captioned, instead of
    only ever having text to work with. This is deliberately NOT the same
    thing as the main corpus's --multimodal CLIP indexing: there is no
    cross-modal "find an image that resembles this text" search here,
    only "here is the specific file this thread uploaded."

    Every resulting chunk's id is prefixed with thread_id (not just
    tagged via metadata) before upsert, so two different threads
    uploading the byte-identical file -- which would otherwise produce
    the SAME content-derived chunk_id (see ingestion/schema.py's
    _stable_id) -- can never collide and silently overwrite each other's
    thread_id metadata in the shared collection.

    Returns {"filename", "n_chunks", "modality", "captioned_images"} --
    n_chunks == 0 is a valid, non-error outcome (e.g. a PDF with no
    extractable text, or an image the VLM couldn't caption and had
    nothing else to fall back to); callers should show that plainly
    rather than treating it as a failure. "captioned_images" is a list of
    {"filename", "caption", "image_path", "data_uri"} dicts, one per
    image actually captioned (empty for a PDF upload, or an image upload
    where captioning produced nothing) -- data_uri is None when the file
    couldn't be persisted/re-read or exceeded MAX_IMAGE_BYTES_FOR_B64, the
    same "degrade, don't break the whole upload" contract
    _image_to_data_uri already documents.

    Raises ValueError for an unsupported extension (see
    SUPPORTED_UPLOAD_EXTS) -- the one thing this function does treat as
    the caller's mistake rather than degrading silently, since accepting
    it would mean upserting nothing at all with no way to tell the person
    why.
    """
    ext = Path(file_path).suffix.lower()
    if ext not in SUPPORTED_UPLOAD_EXTS:
        raise ValueError(
            f"Unsupported upload type {ext!r} -- personal RAG uploads accept "
            f"PDF, plain text, or image files only ({', '.join(SUPPORTED_UPLOAD_EXTS)})."
        )

    raw_docs = ingest_path(file_path, describe_pages="never")

    chunks: list[Chunk] = []
    captioned_images: list[dict] = []
    # Three-way, not the old binary "pdf, else image" -- a .txt upload is
    # neither. CONFIRMED bug this replaces: before ".txt" was a supported
    # extension at all, "else image" was harmless dead code (every other
    # accepted extension really was an image); the moment ".txt" became
    # accepted, that same line would have mislabeled every text upload's
    # own `modality` field as "image" in the stats dict this function
    # returns -- even though the actual ingestion logic below (which
    # branches on each RawDocument's own `.modality`, not on `ext`) would
    # still have chunked it correctly as text. Only the REPORTED modality
    # would have been wrong, but a caller (agents/api.py's upload
    # confirmation message) trusts this field to describe what was
    # actually uploaded.
    if ext == ".pdf":
        modality = "pdf"
    elif ext == ".txt":
        modality = "text"
    else:
        modality = "image"

    image_docs = [d for d in raw_docs if d.modality == "image"]
    other_docs = [d for d in raw_docs if d.modality != "image"]

    if image_docs:
        captions = _caption_image_docs(image_docs)
        for doc, caption in zip(image_docs, captions):
            persisted_path = _persist_personal_image(thread_id, doc)
            display_name = doc.metadata.get("filename", filename)
            text = caption or f"(image: {display_name} -- no caption available)"

            # IMPORTANT: build the chunk via Chunk.new(**doc.metadata) ONLY
            # -- image_path is deliberately NOT passed as part of that
            # **unpack, then set on chunk.metadata explicitly right after
            # construction instead. Chunk.new()'s own `image_path=` is a
            # NAMED parameter (it becomes the dataclass's OWN .image_path
            # field, never chunk.metadata), so passing it through
            # **{...} silently binds it there instead of into metadata --
            # exactly the bug pipeline.py's build_caption_chunks() already
            # documents having caught and fixed once for the main
            # corpus's own image-caption chunks ("a real bug caught while
            # first writing this"). A first version of this function
            # repeated that exact mistake here: the persisted image path
            # never reached store.upsert()'s metadatas (which is built
            # from chunk.metadata alone, see below), so a chunk's own
            # picture could never be found again on a later search --
            # every "what is this image" follow-up degraded to
            # caption-only text with nothing to actually render.
            chunk = Chunk.new(
                doc_id=doc.doc_id,
                text=text,
                modality="text",
                **{**doc.metadata, "filename": display_name},
            )
            chunk.metadata["original_modality"] = "image"
            # Monotonic-enough per-process ordering (not wall-clock
            # precision that matters) so a thread that uploaded more than
            # one image can tell which one came LAST -- see
            # latest_uploaded_image() below, which is what
            # agents/specialists.py falls back to when a semantic search
            # over this thread's uploads doesn't confidently surface the
            # one just sent (e.g. two visually/texturally similar
            # uploads in the same thread ranking close together).
            chunk.metadata["uploaded_at"] = time.time()
            if persisted_path:
                # Only set when persistence actually succeeded -- Chroma's
                # metadata values must be str/int/float/bool, so an
                # `"image_path": None` entry (the alternative on a failed
                # copy) would fail the upsert below rather than just
                # leaving this one chunk without a displayable image.
                chunk.metadata["image_path"] = persisted_path
            chunks.append(chunk)

            if caption:
                captioned_images.append({
                    "filename": display_name,
                    "caption": caption,
                    "image_path": persisted_path,
                    "data_uri": _image_to_data_uri(persisted_path),
                })

    for doc in other_docs:
        if not doc.content.strip():
            continue
        chunks.extend(chunk_recursive(doc))

    if not chunks:
        return {"filename": filename, "n_chunks": 0, "modality": modality, "captioned_images": []}

    embedder = _get_embedder()
    vectors = embedder.embed_texts([c.text for c in chunks])

    ids = [f"{thread_id}:{c.chunk_id}" for c in chunks]
    metadatas = [
        {**c.metadata, THREAD_ID_METADATA_FIELD: thread_id, "filename": filename,
         "source": "personal_upload"}
        for c in chunks
    ]

    store = _fresh_store()
    store.upsert(ids=ids, vectors=vectors, texts=[c.text for c in chunks], metadatas=metadatas)

    return {
        "filename": filename,
        "n_chunks": len(chunks),
        "modality": modality,
        "captioned_images": captioned_images,
    }


# Chroma's query() always returns the top_k NEAREST neighbors, full stop
# -- there's no notion of "not similar enough," so a thread with only a
# couple of unrelated uploads will still get back its single closest
# match even when that match has nothing meaningfully to do with the
# question asked. Confirmed as a real, not hypothetical, problem: a
# thread that had uploaded a treatise PDF plus one unrelated product
# photo got asked "explain miniature painting" and had the PRODUCT
# PHOTO's caption served up as the only context, because it was the
# single closest thing in a very small collection -- not because the
# treatise's own text wasn't there, but because nothing filtered out a
# match that was simply the best of a bad set. generate_answer() then
# correctly says "not enough information" per its own prompt (see
# generation/prompts.py's RAG_SYSTEM_PROMPT), but does so while
# discussing an unrelated image, which reads as confusing/wrong even
# though each individual step behaved as designed.
#
# 0.2 is a conservative floor for all-MiniLM-L6-v2 cosine similarity
# (scores below this are essentially unrelated text for this model, in
# practice) -- deliberately loose rather than tight, since dropping a
# borderline-but-real match is a worse failure than occasionally letting
# through one that's a little thin. Tune by eye against your own uploads
# if it ever filters out something that should have matched.
MIN_PERSONAL_RAG_RELEVANCE_SCORE = 0.2


def search_personal(thread_id: str, query: str, k: int = 5) -> list[dict]:
    """
    Vector-search thread_id's own slice of the "temp" collection --
    called from mcp_server/server.py's search_personal_documents tool,
    which the personal_docs specialist (agents/specialists.py) binds
    directly, the same "explicit tool call, not a react-agent loop"
    shape multi_hop_node uses.

    Plain vector search, not hybrid/reranked like the main corpus's
    retrieve() -- a single conversation's uploads are a handful of files
    at most, not a multi-thousand-chunk corpus, so BM25 fusion and
    cross-encoder reranking (both meant to sharpen precision over a much
    larger candidate pool) aren't worth the extra model calls here.
    Filtered by MIN_PERSONAL_RAG_RELEVANCE_SCORE afterward, though (see
    that constant's own docstring) -- a small candidate pool is exactly
    where "always return the k nearest, however unrelated" bites hardest.

    Returns the same {"text", "score", "metadata"} shape retrieve() does,
    so generate_answer() (already generic over any chunks list, see
    mcp_server/server.py) can consume this directly with zero changes.
    Returns [] if this thread has never had anything uploaded, if nothing
    in what WAS uploaded is relevant, or if everything that came back
    scored below MIN_PERSONAL_RAG_RELEVANCE_SCORE -- callers should treat
    all three as "no personal documents to ground an answer in," not an
    error, the exact same convention retrieve() already documents for an
    empty corpus.
    """
    store = _fresh_store()
    if store.count() == 0:
        return []
    vector = _get_embedder().embed_texts([query])[0]
    results = store.query(vector, top_k=k, where={THREAD_ID_METADATA_FIELD: thread_id})
    return [r for r in results if r.get("score", 0.0) >= MIN_PERSONAL_RAG_RELEVANCE_SCORE]


def latest_uploaded_image(thread_id: str) -> Optional[dict]:
    """
    The single most recently uploaded IMAGE in this thread, or None if
    this thread has never uploaded one -- a direct metadata lookup
    (ChromaStore.get_where), NOT a semantic search, since "which image
    did this thread send most recently" has nothing to do with how
    closely today's question happens to embed near that image's own
    caption text.

    Called from mcp_server/server.py's latest_personal_image tool, which
    agents/specialists.py's image_qa_node / personal_docs_node fall back
    to when search_personal()'s own similarity ranking doesn't
    confidently surface an image-origin chunk for the current question
    (see agents/specialists.py's _best_personal_image_result for exactly
    when and why). Confirmed reason this fallback matters, not a
    hypothetical: a thread that uploads two visually/texturally similar
    images (e.g. the same reference picture sent twice) can have a
    generic follow-up like "what is this?" rank the OLDER upload higher
    than the one that was actually just sent -- pure semantic search has
    no notion of "just now" at all, only "textually similar." This
    function is the deterministic, recency-based answer to that specific
    question, used ONLY as a fallback -- see the caller for why it isn't
    the primary lookup (a thread with several genuinely DIFFERENT
    uploaded images still needs semantic search to pick the right one
    when the question names something specific).

    Returns the same {"text", "score", "metadata"} shape search_personal
    returns, so callers built against that shape (generate_answer,
    agents/specialists.py's _format_personal_image_chunk) need no
    special-casing -- "score" is a constant 1.0 here (there is no
    similarity being measured; this row was chosen by recency, not rank).
    """
    store = _fresh_store()
    if store.count() == 0:
        return None
    candidates = store.get_where({
        THREAD_ID_METADATA_FIELD: thread_id,
        "original_modality": "image",
    })
    if not candidates:
        return None
    best = max(candidates, key=lambda c: (c.get("metadata") or {}).get("uploaded_at", 0))
    return {"text": best["text"], "score": 1.0, "metadata": best.get("metadata") or {}}


def delete_thread_data(thread_id: str) -> None:
    """
    Forget everything uploaded into thread_id -- called by agents/api.py's
    DELETE /chat/{thread_id} alongside the checkpointer's own
    adelete_thread, so a deleted conversation's uploads don't silently
    keep living in the shared "temp" collection (and, since ids are
    thread_id-prefixed, could never even be found again by anything else
    once the thread_id itself is forgotten). Never raises on a thread_id
    that never uploaded anything -- ChromaStore.delete_where's own
    `if where:` guard aside, an empty match is simply a no-op delete, not
    an error, the same "deleting nothing is not a failure" convention the
    checkpointer's own adelete_thread already follows for a thread_id
    with no checkpoints.
    """
    store = _fresh_store()
    store.delete_where({THREAD_ID_METADATA_FIELD: thread_id})

    # Mirror the same cleanup on the persisted-image side (see
    # PERSONAL_UPLOADS_DIR's own docstring in config.py) -- a deleted
    # thread's uploaded pictures shouldn't keep sitting on disk forever
    # just because the Chroma rows that pointed at them are gone. Never
    # raises: a thread that never uploaded an image simply has no such
    # directory, which is a no-op removal, not an error -- same
    # convention delete_where's own `if where:` guard already follows.
    image_dir = PERSONAL_UPLOADS_DIR / thread_id
    if image_dir.exists():
        try:
            shutil.rmtree(image_dir)
        except OSError as e:
            print(f"[personal_rag] could not remove {image_dir}: {e!r}", file=sys.stderr)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample.pdf"
    tid = sys.argv[2] if len(sys.argv) > 2 else "smoke-thread"
    stats = ingest_upload(tid, path, Path(path).name)
    print(f"ingested -> {stats}")
    results = search_personal(tid, "What is this document about?", k=3)
    print(f"search -> {len(results)} result(s)")
    for r in results:
        print(f"  score={r['score']:.3f} text={r['text'][:80]!r}")
    delete_thread_data(tid)
    print("deleted thread data")
