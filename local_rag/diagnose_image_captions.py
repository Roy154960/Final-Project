"""
local_rag/diagnose_image_captions.py

Answers ONE question directly, bypassing semantic search entirely:
does the TEXT store actually contain any dual-indexed image-caption
chunks (metadata["source_type"] == "image_caption", written by
pipeline.build_caption_chunks() -- see that function's own docstring)
at all?

Why this exists: a real /query test (multimodal=false, a question
whose top 10 vector-similarity hits should plausibly include an image
if any dual-indexed image-caption chunk were relevant) came back with
EVERY source showing "source_type": null -- meaning none of the top 10
matches were image-caption chunks. That result alone is ambiguous
between two very different explanations:
  (a) image-caption chunks exist, but none of them were similar enough
      to THIS PARTICULAR question to make the top 10 -- a genuine
      retrieval-relevance question, or
  (b) image-caption chunks don't exist in this corpus AT ALL -- in
      which case no query, however well chosen, could ever surface
      one, and the real issue is upstream at ingestion time, not
      retrieval.
A semantic query can never distinguish these two on its own -- "zero
hits" and "zero candidates" look identical from a vector search. This
script sidesteps the ambiguity entirely with a metadata-only filter
(ChromaDB's own collection.get(where=...), no embedding involved) that
answers "how many exist, period" directly.

ALSO checks the separate image (CLIP) collection's own count for
context: this project's own /ingest route has TWO different paths that
both result in a "real image in the corpus" (see api.py's
_ingest_one_file):
  - multimodal=true  -> ingest_images_multimodal() -> CLIP-embeds the
    image into the image collection AND VLM-captions it into the text
    collection (build_caption_chunks()) -- THIS is the dual-indexed
    path this whole feature depends on.
  - multimodal=false, embedder.supports_images()==true (e.g. embedder=clip
    as the MAIN text embedder) -> embeds the image via CLIP into the
    image collection ONLY -- no caption, no text-store entry, no
    nearby-text embedding AT ALL.
A corpus built the second way has real, findable images via CLIP visual
search (multimodal=true /query calls), but literally nothing this
script is checking for exists -- not a retrieval bug, a "these specific
images were never ingested with multimodal=true" fact. Comparing the
image-collection count against the image-caption-chunk count is what
tells the two apart.

Run:
    python -m diagnose_image_captions
    python -m diagnose_image_captions --store qdrant
"""

import argparse
import sys

from config import CHROMA_COLLECTION, CHROMA_IMAGE_COLLECTION


def _diagnose_chroma() -> None:
    from vectorstore.chroma_store import ChromaStore

    text_store = ChromaStore(collection_name=CHROMA_COLLECTION)
    image_store = ChromaStore(collection_name=CHROMA_IMAGE_COLLECTION)

    text_total = text_store.count()
    image_total = image_store.count()

    # Metadata-only filter -- collection.get(), NOT collection.query() --
    # no embedding computed or compared at all, so this can never miss a
    # match for relevance reasons; it either exists or it doesn't.
    result = text_store._collection.get(
        where={"source_type": "image_caption"},
        include=["documents", "metadatas"],
        limit=10_000,
    )
    n_caption_chunks = len(result["ids"])

    print(f"Text collection ({CHROMA_COLLECTION}): {text_total} chunk(s) total")
    print(f"Image collection ({CHROMA_IMAGE_COLLECTION}): {image_total} image(s) total (CLIP-embedded)")
    print(f"Dual-indexed image-caption chunks in the TEXT collection: {n_caption_chunks}")
    print()

    if n_caption_chunks == 0 and image_total > 0:
        print(
            "DIAGNOSIS: images exist in the corpus (the image collection is non-empty), "
            "but NONE of them were ever dual-indexed into the text store as an "
            "image_caption chunk. This is NOT a retrieval bug -- there is nothing for "
            "any query, however well chosen, to find here. These images were ingested "
            "via the plain CLIP-only path (multimodal=false at /ingest time, with an "
            "image-capable embedder), not the multimodal=true path that actually calls "
            "build_caption_chunks(). To get this feature working, the affected file(s) "
            "need to be RE-INGESTED with multimodal=true (POST /ingest or "
            "/ingest/batch with ?multimodal=true, or `python pipeline.py ingest "
            "--multimodal ...`) -- there's no way to backfill captions onto an already-"
            "stored image without re-running ingestion; the caption text simply was "
            "never generated or written anywhere the first time."
        )
    elif n_caption_chunks == 0 and image_total == 0:
        print(
            "DIAGNOSIS: no images exist ANYWHERE in this corpus -- neither the image "
            "collection nor any image-caption text chunk. Nothing has been ingested "
            "with images at all yet (or --all/--images was cleared since). Ingest a "
            "PDF with embedded images using multimodal=true to populate both."
        )
    else:
        print(
            f"DIAGNOSIS: {n_caption_chunks} image-caption chunk(s) genuinely exist in "
            "the text store. If your /query test still didn't surface one, check the "
            "'usable' samples printed below FIRST, not just any of the 75 -- some "
            "fraction of a corpus's image-caption chunks can legitimately have an "
            "empty nearby_text (a genuinely blank/image-only scanned page has nothing "
            "real nearby to find -- see ingestion/ingest_pdf.py's own "
            "_nearby_page_text fallback chain), and testing against one of those was "
            "never going to work regardless of query phrasing, since there's no real "
            "surrounding-text signal on that specific page to retrieve by. A query "
            "close to one of the USABLE samples below is the fair test."
        )

    if n_caption_chunks:
        # Sort so chunks with a REAL nearby_text come first -- printing
        # the first 10 by raw insertion order was a real problem on a
        # live run: this corpus's image_caption chunks happen to start
        # with a long run of blank/black scanned pages from one book
        # (VLM caption: "This image is completely black... no visible
        # subject matter"), all correctly carrying an EMPTY nearby_text
        # -- see _nearby_page_text's own fallback chain in
        # ingestion/ingest_pdf.py: its last line is
        # `return page_text[:max_chars]`, so a genuinely blank page's
        # own already-empty extracted text correctly degrades to "",
        # not a bug. Printing 10 of those in a row told the person
        # nothing useful about whether nearby-text retrieval WORKS --
        # there was nothing on those specific pages to test with in the
        # first place. Sorting real-nearby_text chunks first means the
        # samples below are actually usable as test cases.
        def _has_real_nearby_text(meta: dict) -> bool:
            return len((meta.get("nearby_text") or "").strip()) > 20

        indices = list(range(n_caption_chunks))
        indices.sort(key=lambda i: not _has_real_nearby_text(result["metadatas"][i]))

        n_with_nearby_text = sum(1 for i in indices if _has_real_nearby_text(result["metadatas"][i]))
        print(
            f"\nOf those {n_caption_chunks}, {n_with_nearby_text} have a REAL (20+ char) "
            f"nearby_text and {n_caption_chunks - n_with_nearby_text} don't (either a "
            f"genuinely blank/image-only page with nothing nearby to find, or a "
            f"standalone upload with no page to be near at all -- see this project's "
            f"own nearby_text comment in pipeline.build_caption_chunks)."
        )
        print(f"Showing the {min(10, n_caption_chunks)} MOST USABLE examples first "
              f"(real nearby_text sorted ahead of empty ones) -- these are the ones "
              f"actually worth building a /query test case around:")
        for rank, i in enumerate(indices[:10]):
            meta = result["metadatas"][i]
            doc = result["documents"][i]
            print(f"\n  [{rank + 1}] {meta.get('filename', '?')}, page {meta.get('page', '?')}")
            print(f"      image_path: {meta.get('image_path', '?')}")
            print(f"      caption:    {(meta.get('caption') or '')[:150]!r}")
            print(f"      nearby_text:{(meta.get('nearby_text') or '(none)')[:150]!r}")
            print(f"      embedded text (caption+nearby, what was actually searched): {doc[:150]!r}")


def _diagnose_qdrant() -> None:
    from config import QDRANT_COLLECTION, QDRANT_IMAGE_COLLECTION
    from vectorstore.qdrant_store import QdrantStore

    # dimensions is irrelevant here -- both collections already exist
    # (or don't); this script never writes anything, so a wrong
    # placeholder dimension is harmless, unlike pipeline.py's own
    # cmd_clear, which actually recreates collections and does need the
    # real value -- see that function's own comment.
    text_store = QdrantStore(collection_name=QDRANT_COLLECTION, dimensions=1)
    image_store = QdrantStore(collection_name=QDRANT_IMAGE_COLLECTION, dimensions=1)

    text_total = text_store.count()
    image_total = image_store.count()

    matches, n_caption_chunks = [], 0
    offset = None
    while True:
        points, offset = text_store._client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter={"must": [{"key": "source_type", "match": {"value": "image_caption"}}]},
            with_payload=True,
            limit=100,
            offset=offset,
        )
        matches.extend(points)
        n_caption_chunks += len(points)
        if offset is None:
            break

    print(f"Text collection ({QDRANT_COLLECTION}): {text_total} chunk(s) total")
    print(f"Image collection ({QDRANT_IMAGE_COLLECTION}): {image_total} image(s) total (CLIP-embedded)")
    print(f"Dual-indexed image-caption chunks in the TEXT collection: {n_caption_chunks}")
    print()

    if n_caption_chunks == 0 and image_total > 0:
        print(
            "DIAGNOSIS: images exist in the corpus, but none were dual-indexed as an "
            "image_caption chunk -- same conclusion and same fix as the Chroma path's "
            "own message; see this script's --store chroma output for the full "
            "explanation, or re-run with --store chroma if that's actually what you're using."
        )
    elif n_caption_chunks:
        print(f"\nFirst {min(10, len(matches))} image-caption chunk(s) found:")
        for i, p in enumerate(matches[:10]):
            meta = p.payload or {}
            print(f"\n  [{i + 1}] {meta.get('filename', '?')}, page {meta.get('page', '?')}")
            print(f"      image_path: {meta.get('image_path', '?')}")
            print(f"      caption:    {(meta.get('caption') or '')[:150]!r}")
            print(f"      nearby_text:{(meta.get('nearby_text') or '(none)')[:150]!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", choices=["chroma", "qdrant"], default="chroma")
    args = parser.parse_args()

    print(f"[diagnose_image_captions] checking the '{args.store}' store...\n", file=sys.stderr)
    if args.store == "chroma":
        _diagnose_chroma()
    else:
        _diagnose_qdrant()


if __name__ == "__main__":
    main()
