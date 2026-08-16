"""
Shared data structures that flow through every stage of the pipeline:
Ingest -> Chunk -> Embed -> Store -> Retrieve -> Generate
"""

import hashlib
from dataclasses import dataclass, field
from typing import Literal, Optional
from pathlib import Path


def _stable_id(*parts: str) -> str:
    """
    Deterministic id derived from content, not a random uuid4 — so
    re-ingesting the identical file/chunk produces the SAME id every time,
    and store.upsert() (which is ID-keyed) naturally overwrites in place
    instead of inserting a duplicate. Previously every RawDocument/Chunk got
    a fresh uuid4() regardless of content, so re-POSTing the same file to
    POST /ingest (or re-running plain `pipeline.py ingest` without
    --incremental) silently duplicated it in the vector store on every call.

    Content, not source_path, is the anchor: api.py stages every upload
    under a fresh uuid4-prefixed temp filename specifically to avoid
    collisions between concurrent uploads
    (`RAW_DOCS_DIR / f"{uuid.uuid4().hex}_{file.filename}"`), which means
    source_path is DIFFERENT on every single upload even for the
    byte-identical file — anchoring identity to it would silently defeat
    this fix for exactly the API-upload case it's meant to fix. Content is
    the one thing guaranteed stable across re-ingestion regardless of which
    interface (CLI or API) staged it or under what temp path.
    """
    digest_input = "\x1f".join(parts)  # unit separator avoids "a"+"bc" == "ab"+"c" collisions
    return hashlib.sha256(digest_input.encode("utf-8", errors="ignore")).hexdigest()


@dataclass
class RawDocument:
    """Output of the Ingest step, before chunking."""
    doc_id: str
    source_path: str
    modality: Literal["text", "pdf_text", "pdf_table", "image"]
    content: str            # extracted/raw text, or image caption placeholder
    image_path: Optional[str] = None  # set if modality == "image" (or embedded PDF image)
    metadata: dict = field(default_factory=dict)

    @staticmethod
    def new(source_path: str, modality: str, content: str, image_path: str = None, **meta):
        # Text/PDF-page docs: content (the extracted text) is the identity —
        # different pages of the same PDF naturally have different content,
        # so this also distinguishes per-page RawDocuments from one PDF
        # without needing to special-case "page" here. "page" is folded in
        # too anyway, as a cheap extra guard against two genuinely-identical
        # (e.g. blank) pages colliding.
        #
        # Images: content is always "" (see ingest_image.py / ingest_pdf.py —
        # filled in later by CLIP/captioning), so the actual image BYTES are
        # hashed instead — image_path's string value can't be trusted either,
        # for the same randomized-staging-path reason as source_path above.
        if image_path and not content:
            try:
                identity = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
            except OSError:
                # Not on disk yet / unreadable for some reason — fall back to
                # the path string rather than crashing document construction.
                identity = image_path
        else:
            identity = content
        doc_id = _stable_id(identity, str(meta.get("page", "")))
        return RawDocument(
            doc_id=doc_id,
            source_path=source_path,
            modality=modality,
            content=content,
            image_path=image_path,
            metadata=meta,
        )


@dataclass
class Chunk:
    """Output of the Chunk step, input to Embed."""
    chunk_id: str
    doc_id: str
    text: str
    modality: Literal["text", "image"]
    image_path: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @staticmethod
    def new(doc_id: str, text: str, modality: str = "text", image_path: str = None, **meta):
        # doc_id is itself content-derived now (see RawDocument.new() above),
        # so hashing it together with this chunk's own text gives a chunk_id
        # that's stable across re-ingestion of the identical source, while
        # still unique per distinct chunk within that source (chunks from
        # the same doc almost always differ in text; two byte-identical
        # chunks in the same doc collapsing into one store entry is the same
        # tradeoff ingestion/deduplication.py's exact-hash mode already
        # makes on purpose).
        chunk_id = _stable_id(doc_id, text)
        return Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=text,
            modality=modality,
            image_path=image_path,
            metadata=meta,
        )
