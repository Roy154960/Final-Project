"""
Image retrieval + auto-caption backing for the new `retrieve_images` MCP
tool (see server.py for the tool wrapper).

Deliberately reuses existing pipeline infrastructure rather than
reimplementing it:
  - embeddings/clip_embedder.py + vectorstore's CHROMA_IMAGE_COLLECTION
    for genuine cross-modal (text query -> image) retrieval, exactly the
    store pipeline.py's own --multimodal ingest path already builds --
    see retrieval/image_retriever.py's module docstring for why CLIP
    similarity, not the text embedder, is what makes this retrieval
    "real" rather than keyword-matching a filename.
  - ingestion/image_captioning.py's VLM captioning, for the "auto
    generated caption" the user-facing spec asks for. Ingest time
    already captions every image once (see pipeline.py's --multimodal
    path) and stores it as metadata["caption"] alongside the image's
    CLIP vector -- this module uses that stored caption as the primary
    source, and only calls the VLM live (one extra model call) as a
    fallback for the rare case a retrieved image has no stored caption
    (e.g. ingested before --multimodal captioning was enabled). This
    mirrors corpus_meta's "prefer a cheap, already-computed snapshot
    over an expensive live call" design choice in agents/specialists.py.

Like server.py's own component block, image/CLIP loading happens lazily
and is cached at module scope rather than per-call -- CLIP and the VLM
both load real model weights, which should happen at most once per
server process, not once per tool invocation.
"""

import base64
import mimetypes
import sys
from pathlib import Path
from typing import Optional

# Safety cap on how large a single image's base64-encoded payload is
# allowed to be before retrieve_images_with_data() (below) will embed it.
# Base64 runs ~33% larger than the raw bytes, and this response travels
# over the same MCP stdio/JSON channel server.py's own comment already
# warns is easy to corrupt with unexpected output -- a multi-ingested
# corpus can contain full-page scans well into the tens of MB, and
# embedding several of those in one k=3-5 tool response would bloat the
# payload a client (or an LLM context window reading the tool result)
# has to handle. 5 MB is a generous starting point for typical
# treatise figures/diagrams; tune per-corpus if needed.
MAX_IMAGE_BYTES_FOR_B64 = 5 * 1024 * 1024

_embedder = None  # ClipEmbedder, lazy-loaded (torch + open_clip are optional deps)
_image_store = None  # ChromaStore over CHROMA_IMAGE_COLLECTION
_vlm = None  # OllamaVLM, lazy-loaded only if a live caption fallback is ever needed
_load_attempted = False


def _log(msg: str) -> None:
    print(f"[image_tools] {msg}", file=sys.stderr)


def _escape_markdown_caption(caption: Optional[str]) -> Optional[str]:
    """
    Neutralize a caption before it's interpolated into hand-built
    `![caption](url)` markdown syntax (format_markdown_image,
    format_markdown_image_embedded below).

    Captions are free-form VLM output -- config.IMAGE_CAPTION_PROMPT
    asks for "subject matter, style, composition, colors, and any
    visible text" with no constraint on punctuation -- so an unescaped
    "]" in a caption prematurely closes the image's `![...]` alt-text
    span, and everything after it (the whole `(url)` destination) falls
    through as plain paragraph text instead of being parsed as an image.
    For a short image_path that's a barely-noticeable glitch; for a
    data: URI carrying a multi-hundred-KB base64 payload
    (format_markdown_image_embedded), the same bug spills the entire
    raw base64 string into the chat as visible text -- this is the fix
    for exactly that failure mode.

    Escapes backslash and square brackets (the characters that can
    break the alt-text span) and collapses newlines/repeated whitespace
    to single spaces (a caption spanning multiple lines would break the
    single-line syntax these functions build regardless of escaping).
    Deliberately leaves other punctuation -- parentheses, asterisks,
    backticks, etc. -- untouched: those only matter inside the
    URL/destination part of markdown link syntax, and every destination
    these functions build (image_path, data_uri) is program-constructed,
    never caption text, so they're not at risk here. Returns the input
    unchanged if it's empty/None, so callers' own fallback-text logic
    (`caption or "(no caption available)"`) still works untouched.
    """
    if not caption:
        return caption
    collapsed = " ".join(caption.split())
    return "".join(f"\\{ch}" if ch in ("\\", "[", "]") else ch for ch in collapsed)


def _ensure_loaded() -> bool:
    """
    Lazily build the CLIP embedder + image vector store on first use, not
    at import time -- unlike server.py's text-side components (always
    needed), the image path is optional: a project that never ran
    `pipeline.py --multimodal` has no CHROMA_IMAGE_COLLECTION data at
    all, and importing torch/open_clip just to immediately find an empty
    store would be wasted startup cost on every server run, multimodal or
    not.

    Returns True if both the embedder and a non-empty image store are
    available, False otherwise (missing optional deps, or nothing
    ingested into the image collection yet) -- callers treat False as
    "return an empty result," never as an error, the same as
    server.py's `_retriever is None` check for the text-only case.

    Cached via `_load_attempted` so a missing dependency (or an empty
    store) is logged once per server process, not once per tool call.
    """
    global _embedder, _image_store, _load_attempted

    if _load_attempted:
        return _embedder is not None and _image_store is not None and _image_store.count() > 0

    _load_attempted = True
    try:
        from config import CHROMA_IMAGE_COLLECTION
        from embeddings.clip_embedder import ClipEmbedder
        from vectorstore.chroma_store import ChromaStore
    except ImportError as e:
        _log(f"multimodal deps unavailable ({e}) -- retrieve_images will return []")
        return False

    try:
        _embedder = ClipEmbedder()
        _image_store = ChromaStore(collection_name=CHROMA_IMAGE_COLLECTION)
    except Exception as e:  # noqa: BLE001 -- e.g. CLIP weights failed to download/load
        _log(f"failed to initialize CLIP/image store ({e}) -- retrieve_images will return []")
        _embedder = None
        _image_store = None
        return False

    if _image_store.count() == 0:
        _log(
            "image collection is empty -- no images have been ingested with "
            "`pipeline.py --multimodal` yet; retrieve_images will return []"
        )
        return False

    return True


def _load_vlm():
    """
    Lazily load a VLM instance for the live-caption fallback path only --
    most calls never touch this, since ingest-time captions are the
    primary source (see this module's docstring). Cached at module scope
    once loaded, same reasoning as `_ensure_loaded`.

    Groq-first, automatic local-Ollama-fallback (vlm/fallback_vlm.py's
    FallbackVLM) -- same "online first, local backup" wrapper used
    everywhere else in this project a VLM gets called live rather than
    at ingest time (see personal_rag.py's own single-image captioning
    path). A missing GROQ_API_KEY or a Groq-side hiccup degrades
    silently to the exact same local OllamaVLM this function returned
    before Groq was added -- never a reason for this tool's own "every
    returned image has a caption" contract to fail.
    """
    global _vlm
    if _vlm is None:
        from vlm.fallback_vlm import FallbackVLM

        _vlm = FallbackVLM()
    return _vlm


def retrieve_images_with_captions(query: str, k: int = 3) -> list[dict]:
    """
    The function behind the `retrieve_images` MCP tool.

    Embeds `query` with CLIP's text encoder and does genuine cross-modal
    similarity search against the image store (see
    retrieval/image_retriever.py's module docstring for why this is real
    visual retrieval, not filename keyword matching). For each hit,
    returns its already-stored ingest-time caption if present; if a hit
    somehow has no stored caption (metadata["caption"] missing or
    empty), captions it live with a VLM as a fallback, so this tool's
    contract -- "every returned image has a caption" -- always holds
    rather than depending on ingest-time coverage.

    Returns a list of dicts, best match first:
      {"image_path": str, "caption": str, "score": float, "metadata": dict}
    Returns [] if the image store isn't available or is empty (see
    `_ensure_loaded`) -- never raises. The calling specialist (image_qa)
    is expected to say so plainly to the user, the same "say plainly,
    don't guess" rule this project already applies to an empty text
    retrieve() result.
    """
    if not _ensure_loaded():
        return []

    query_vec = _embedder.embed_texts([query])[0]
    hits = _image_store.query(query_vec, top_k=k)

    results = []
    for hit in hits:
        metadata = hit.get("metadata", {}) or {}
        image_path = metadata.get("image_path", "")
        caption = (metadata.get("caption") or "").strip()

        if not caption and image_path and Path(image_path).exists():
            try:
                caption = _load_vlm().describe_image(image_path).strip()
            except Exception as e:  # noqa: BLE001 -- a failed live caption shouldn't drop the image
                _log(f"live captioning failed for {image_path}: {e}")
                caption = "(no caption available)"
        elif not caption:
            caption = "(no caption available)"

        results.append(
            {
                "image_path": image_path,
                "caption": caption,
                "score": hit.get("score", 0.0),
                "metadata": metadata,
            }
        )

    return results


def retrieve_similar_images_with_captions(image_path: str, k: int = 3, exclude_path: Optional[str] = None) -> list[dict]:
    """
    The function behind the `find_similar_images` MCP tool.

    Genuine image-to-image retrieval: embeds the image AT `image_path`
    with CLIP's own IMAGE encoder (not its text encoder -- see
    ClipEmbedder.embed_images, embeddings/clip_embedder.py) and queries
    the SAME corpus image store retrieve_images_with_captions above
    queries, just with a different kind of query vector. CLIP puts text
    and image embeddings in one shared space (see this module's own
    top docstring), so this is the direct visual-similarity sibling of
    that function's cross-modal text-to-image search -- "find me corpus
    images that visually resemble THIS picture," not "find me corpus
    images matching this description."

    `image_path` is expected to be a path THIS SERVER PROCESS can read
    directly off local disk -- for the `find_similar_images` tool's own
    use (a person's personal upload), that's local_rag/personal_rag.py's
    own on-disk storage, which this same process already reads via the
    `personal_rag` import server.py uses for search_personal_documents/
    latest_personal_image. Never a URL, never bytes -- if a caller only
    has bytes (e.g. an image that was never persisted to disk), it needs
    to write them to a temp file first.

    `exclude_path`, when given, drops any hit whose OWN image_path
    exactly matches it before results are returned -- the query image
    itself, if it happens to already be IN the corpus store (a genuine
    possibility for e.g. a well-known painting the person also has a
    personal copy of), would otherwise always rank as its own top-1
    "most similar" result, which tells the person nothing they don't
    already know. Comparing raw paths is deliberately simple rather than
    a content hash: the query image is never itself a corpus-store
    member in the common case (a person's own photo/study reference), so
    this only ever matters for the rarer coincidental-match case, where
    an exact path match is already a strong enough signal.

    Returns the exact same shape retrieve_images_with_captions does:
      [{"image_path": str, "caption": str, "score": float, "metadata": dict}, ...]
    best match first. Returns [] if the image store isn't available or
    is empty (see `_ensure_loaded`), or if `image_path` itself can't be
    read/embedded (a corrupted file, an unsupported format -- see
    ClipEmbedder.embed_images' own zero-vector degrade, which this
    function treats as "no usable query, no results" rather than
    surfacing a zero-vector's meaningless nearest-neighbors) -- never
    raises, same "say plainly, don't guess" contract every other
    function in this module already follows.
    """
    if not _ensure_loaded():
        return []
    if not image_path or not Path(image_path).is_file():
        _log(f"find_similar_images: query image not readable at {image_path!r}")
        return []

    query_vecs = _embedder.embed_images([image_path])
    if query_vecs is None or len(query_vecs) == 0 or not query_vecs[0].any():
        # embed_images() degrades an unreadable/corrupted image to a zero
        # vector rather than raising (see its own docstring) -- a zero
        # vector's "nearest neighbors" are meaningless (every image is
        # equally, arbitrarily "close" to it), so treat this the same as
        # "couldn't read the image" rather than returning junk results.
        _log(f"find_similar_images: CLIP could not embed {image_path!r} -- returning []")
        return []

    hits = _image_store.query(query_vecs[0], top_k=k + (1 if exclude_path else 0))

    results = []
    for hit in hits:
        metadata = hit.get("metadata", {}) or {}
        hit_path = metadata.get("image_path", "")
        if exclude_path and hit_path and Path(hit_path) == Path(exclude_path):
            continue
        caption = (metadata.get("caption") or "").strip()

        if not caption and hit_path and Path(hit_path).exists():
            try:
                caption = _load_vlm().describe_image(hit_path).strip()
            except Exception as e:  # noqa: BLE001 -- a failed live caption shouldn't drop the image
                _log(f"live captioning failed for {hit_path}: {e}")
                caption = "(no caption available)"
        elif not caption:
            caption = "(no caption available)"

        results.append(
            {
                "image_path": hit_path,
                "caption": caption,
                "score": hit.get("score", 0.0),
                "metadata": metadata,
            }
        )
        if len(results) >= k:
            break

    return results


def format_markdown_image(item: dict) -> str:
    """
    Render one retrieve_images() result as a markdown image embed with
    its caption underneath -- the "images embedded with a small
    auto-generated caption" format the image_qa specialist assembles its
    final answer from. Kept as its own function (not inlined in
    specialists.py) so the exact markdown shape only needs to change in
    one place.

    `image_path` is a local filesystem path, not a URL -- rendering it
    as a markdown image only actually displays inline in a client that
    can resolve local paths (e.g. a local markdown viewer, or a future
    web UI serving `data/` as static files). In a chat client that can't,
    the path text itself is still useful as a pointer to the file. This
    is a known limitation worth stating directly rather than glossing
    over: serving images over HTTP (e.g. a small static file route
    alongside the MCP server) is a reasonable Part-2 extension, not
    implemented here.
    """
    caption = _escape_markdown_caption(item.get("caption")) or "(no caption available)"
    path = item.get("image_path") or "(no path)"
    return f"![{caption}]({path})\n*{caption}*"


# ----------------------------------------------------------------------
# Base64-embedded retrieval -- new, additive code.
#
# format_markdown_image()'s own docstring above already names the gap:
# `image_path` is a path on the SERVER's local disk, so a remote caller
# (an MCP client on a different machine/process, a browser, a chat UI
# with no filesystem access of its own) can never actually load it --
# "serving images over HTTP... is a reasonable Part-2 extension, not
# implemented here." Everything below is the base64 alternative to that
# HTTP-serving extension: instead of a path the caller has to separately
# fetch, the image's actual bytes ride along IN the tool response.
#
# Deliberately kept as brand-new functions (retrieve_images_with_data,
# format_markdown_image_embedded) rather than changes to
# retrieve_images_with_captions() / format_markdown_image() above --
# every existing caller of those two (the `retrieve_images` MCP tool,
# agents/specialists.py's image_qa_node, this module's own smoke tests)
# keeps working completely unchanged, byte-for-byte, whether or not it
# ever adopts the new base64 path.
# ----------------------------------------------------------------------


def _encode_image_base64(image_path: str, max_bytes: int = MAX_IMAGE_BYTES_FOR_B64) -> Optional[dict]:
    """
    Read one image file off disk and base64-encode it, entirely locally
    -- no network, no external service. Mirrors this module's existing
    "never raise, degrade to an explainable empty/None result" contract
    (see _ensure_loaded, retrieve_images_with_captions): a missing file,
    an unreadable file, or one over `max_bytes` all return None rather
    than raising, so one bad image can't take down a whole
    retrieve_images_with_data() call the way one bad caption already
    can't (see caption_images()'s own per-image try/except in
    ingestion/image_captioning.py -- same philosophy, applied here to
    reading bytes instead of generating text).

    Returns {"base64": str, "mime_type": str, "size_bytes": int} on
    success. `mime_type` is guessed from the file extension (stdlib
    `mimetypes`, no extra dependency) and falls back to
    "application/octet-stream" for an extension it doesn't recognize --
    still a valid data: URI, just without a specific image MIME type a
    renderer could use to pick a decoder, which only matters if the file
    genuinely isn't a normal image format to begin with.
    """
    if not image_path:
        return None

    path = Path(image_path)
    if not path.exists() or not path.is_file():
        _log(f"cannot base64-encode {image_path!r} -- file not found on disk")
        return None

    try:
        size_bytes = path.stat().st_size
    except OSError as e:  # noqa: BLE001 -- e.g. a permissions error mid-check
        _log(f"cannot stat {image_path!r}: {e}")
        return None

    if size_bytes > max_bytes:
        _log(
            f"skipping base64 encode of {image_path!r} -- {size_bytes} bytes exceeds "
            f"the {max_bytes}-byte cap (see MAX_IMAGE_BYTES_FOR_B64)"
        )
        return None

    try:
        raw_bytes = path.read_bytes()
    except OSError as e:  # noqa: BLE001 -- e.g. a permissions error or race with deletion
        _log(f"failed to read {image_path!r} for base64 encoding: {e}")
        return None

    mime_type, _ = mimetypes.guess_type(str(path))
    if not mime_type:
        mime_type = "application/octet-stream"

    return {
        "base64": base64.b64encode(raw_bytes).decode("ascii"),
        "mime_type": mime_type,
        "size_bytes": size_bytes,
    }


def retrieve_images_with_data(query: str, k: int = 3, max_bytes: int = MAX_IMAGE_BYTES_FOR_B64) -> list[dict]:
    """
    The function behind the `retrieve_images_embedded` MCP tool.

    Runs the exact same retrieval as retrieve_images_with_captions()
    above -- reused by calling it directly, not duplicated -- so the
    CLIP query, top-k search, and stored/live-fallback captioning logic
    only exist in one place. This function's only added job is reading
    each hit's image file off disk and base64-encoding it (see
    _encode_image_base64 above) before returning.

    Returns a list of dicts, best match first, each with everything
    retrieve_images_with_captions() already returns --
    "image_path", "caption", "score", "metadata" -- PLUS:
      - "image_base64": the raw base64-encoded image bytes (str), or
            None if the file was missing, unreadable, or over
            `max_bytes`
      - "mime_type": e.g. "image/png", or None when image_base64 is None
      - "data_uri": a ready-to-embed "data:<mime_type>;base64,<data>"
            string -- drop it straight into markdown
            (`![caption](data_uri)`) or an HTML/React `<img src=...>` --
            or None when image_base64 is None
      - "encoding_note": present ONLY when image_base64 is None,
            explaining why (missing file / unreadable / over the size
            cap) in plain text -- absent entirely when encoding
            succeeded, so callers can check `"encoding_note" in item` or
            just `item["image_base64"] is None`

    A file that can't be embedded still comes back in the list with
    image_base64=None rather than being dropped -- the path, caption,
    score, and metadata are still legitimate, useful information even
    without the bytes, the same "degrade a field, don't drop the
    result" policy the caption fallback above already follows. Returns
    [] under the exact same conditions retrieve_images_with_captions()
    does (image stack unavailable, nothing ingested with --multimodal)
    -- never raises.
    """
    results = retrieve_images_with_captions(query, k=k)

    for item in results:
        encoded = _encode_image_base64(item.get("image_path", ""), max_bytes=max_bytes)
        if encoded is None:
            item["image_base64"] = None
            item["mime_type"] = None
            item["data_uri"] = None
            item["encoding_note"] = (
                "image bytes could not be embedded (file missing, unreadable, or over "
                f"the {max_bytes}-byte cap) -- image_path/caption/score/metadata above "
                "are still valid"
            )
        else:
            item["image_base64"] = encoded["base64"]
            item["mime_type"] = encoded["mime_type"]
            item["data_uri"] = f"data:{encoded['mime_type']};base64,{encoded['base64']}"

    return results


def retrieve_similar_images_with_data(
    image_path: str, k: int = 3, exclude_path: Optional[str] = None, max_bytes: int = MAX_IMAGE_BYTES_FOR_B64
) -> list[dict]:
    """
    The function behind the `find_similar_images` MCP tool's embedded
    variant -- sibling to retrieve_images_with_data() above, for
    retrieve_similar_images_with_captions() the same way that function
    is a sibling to retrieve_images_with_captions(). Runs the image-to-
    image search, then reuses the exact same _encode_image_base64
    encoding this module already has for the text-query path, so both
    retrieval modes produce byte-for-byte the same result shape
    ("image_base64"/"mime_type"/"data_uri"/"encoding_note") and can be
    rendered with the same format_markdown_image_embedded() either way.
    """
    results = retrieve_similar_images_with_captions(image_path, k=k, exclude_path=exclude_path)

    for item in results:
        encoded = _encode_image_base64(item.get("image_path", ""), max_bytes=max_bytes)
        if encoded is None:
            item["image_base64"] = None
            item["mime_type"] = None
            item["data_uri"] = None
            item["encoding_note"] = (
                "image bytes could not be embedded (file missing, unreadable, or over "
                f"the {max_bytes}-byte cap) -- image_path/caption/score/metadata above "
                "are still valid"
            )
        else:
            item["image_base64"] = encoded["base64"]
            item["mime_type"] = encoded["mime_type"]
            item["data_uri"] = f"data:{encoded['mime_type']};base64,{encoded['base64']}"

    return results


def format_markdown_image_embedded(item: dict) -> str:
    """
    Sibling to format_markdown_image() above, for
    retrieve_images_with_data() results: embeds the image as a `data:`
    URI instead of a bare filesystem path, so it renders inline in ANY
    markdown client with no static file server, no HTTP round-trip, and
    no filesystem access needed on the reader's side -- the exact gap
    format_markdown_image()'s own docstring names as not implemented.

    Falls back to format_markdown_image()'s own (unchanged) path-based
    rendering when this item has no "data_uri" -- e.g. the size cap or a
    missing file (see retrieve_images_with_data) -- so a partially-
    embeddable result set still renders something useful for every item
    instead of a broken image tag for the ones that couldn't be encoded.
    """
    data_uri = item.get("data_uri")
    if not data_uri:
        return format_markdown_image(item)
    caption = _escape_markdown_caption(item.get("caption")) or "(no caption available)"
    return f"![{caption}]({data_uri})\n*{caption}*"


if __name__ == "__main__":
    print("This module is meant to be imported by mcp_server/server.py.")
    print("Quick manual check (needs a --multimodal-ingested corpus):")
    for r in retrieve_images_with_captions("a painter's palette", k=2):
        print(format_markdown_image(r))
        print()
