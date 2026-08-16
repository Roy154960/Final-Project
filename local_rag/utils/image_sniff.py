"""
Tiny, dependency-free image-format sniffer — reads just the first ~16
bytes of a file and matches them against known magic numbers.

Why this exists: ingestion/ingest_pdf.py saves each PDF-embedded image
using whatever `ext` PyMuPDF's extract_image() reports for it (e.g.
"jpeg", "png", "jpx") — that ext comes from PyMuPDF's own detection of
the image's filter chain in the PDF, and isn't independently re-verified
against the actual saved bytes. Scanned/archival PDFs (Internet Archive
book exports, in particular) very commonly embed images with a
JPXDecode (JPEG2000) filter rather than a plain JPEG/DCTDecode one — it
compresses scanned pages noticeably better than JPEG at the same visual
quality, which is exactly why archival scanning pipelines lean on it.
That's the ordinary reason you'll see some extracted images end up with
a ".jpx" extension: nothing went wrong, that's genuinely what was
embedded in the source PDF.

Separately — and this is the actual bug case — PyMuPDF's ext detection
doesn't perfectly identify every filter-chain/colorspace combination, and
some archival PDFs have partially malformed or truncated image streams to
begin with. Either way, the practical symptom is the same: a file saved
with a normal-looking extension like ".jpeg" whose actual bytes are
something else entirely (often JPEG2000) or aren't a complete/valid image
at all — which is exactly what makes Pillow's Image.open() raise "cannot
identify image file" later, in embeddings/clip_embedder.py, with no
indication of *why*.

This module lets both ends of that gap report the SAME diagnosis instead
of the CLIP embed step just showing a bare, unexplained PIL error:
ingest_pdf.py can flag the mismatch immediately at extraction time, and
clip_embedder.py can explain precisely why a given image failed to open
when it does.
"""

from pathlib import Path
from typing import Optional

# (magic bytes, human-readable format name). Checked in order — most
# specific/common signatures first. Not exhaustive; anything not listed
# here just sniffs as None ("unrecognized"), which is still useful
# information (usually means truncated/corrupted rather than mislabeled).
_SIGNATURES: list[tuple[bytes, str]] = [
    (b"\x89PNG\r\n\x1a\n", "PNG"),
    (b"\xff\xd8\xff", "JPEG"),
    (b"\x00\x00\x00\x0cjP  \r\n\x87\n", "JPEG2000"),  # JP2 container
    (b"\xff\x4f\xff\x51", "JPEG2000"),                 # raw J2K codestream
    (b"GIF87a", "GIF"),
    (b"GIF89a", "GIF"),
    (b"BM", "BMP"),
    (b"II*\x00", "TIFF"),
    (b"MM\x00*", "TIFF"),
    (b"RIFF", "WEBP"),
]

# Which sniffed format each declared extension is expected to match.
# Extensions not listed here (or sniffed formats not in this table) are
# simply not checked — describe_ext_mismatch() stays silent rather than
# guessing at formats it doesn't know about.
_EXT_TO_EXPECTED_FORMAT = {
    "jpg": "JPEG", "jpeg": "JPEG",
    "png": "PNG",
    "jp2": "JPEG2000", "jpx": "JPEG2000", "j2k": "JPEG2000",
    "gif": "GIF",
    "bmp": "BMP",
    "tif": "TIFF", "tiff": "TIFF",
    "webp": "WEBP",
}


def sniff_image_format(path: str) -> Optional[str]:
    """Reads the first 16 bytes of `path` and matches them against known
    image magic numbers. Returns a human-readable format name (e.g.
    "JPEG2000"), or None if the file is empty/unreadable/doesn't match
    anything recognized — most often that means a truncated or otherwise
    corrupted extraction, not necessarily a format this function simply
    doesn't know about."""
    try:
        with Path(path).open("rb") as f:
            header = f.read(16)
    except Exception:
        return None
    for magic, name in _SIGNATURES:
        if header.startswith(magic):
            return name
    return None


def describe_ext_mismatch(path: str, declared_ext: str) -> Optional[str]:
    """One-line diagnosis for a log message, or None if there's nothing
    to report (the sniffed bytes match the declared extension, or the
    extension/sniff isn't one this module knows how to compare)."""
    declared_norm = declared_ext.lower().lstrip(".")
    expected = _EXT_TO_EXPECTED_FORMAT.get(declared_norm)
    if expected is None:
        return None  # unrecognized declared extension — nothing to compare against

    sniffed = sniff_image_format(path)
    if sniffed is None:
        return (f"declared .{declared_norm}, but the file's first bytes don't match any known "
                f"image format — likely a truncated or corrupted extraction, not just a labeling issue")
    if sniffed == expected:
        return None  # matches, nothing to report

    return (f"declared .{declared_norm}, but the actual bytes are {sniffed} — PyMuPDF's format "
            f"detection didn't match the true encoding for this image")
