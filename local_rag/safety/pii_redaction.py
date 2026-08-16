"""
Safety enhancement - PII redaction before storage.

Redacts common PII patterns from chunk text before it's embedded and
stored, so sensitive data isn't sitting in your vector database (which,
unlike a source document, might get queried/exposed in ways you didn't
originally intend — e.g. shared with a teammate, exposed via an API).

This uses plain regex — free, local, zero extra dependencies, catches
well-structured PII (emails, phone numbers, SSN-shaped numbers, credit-
card-shaped numbers, IP addresses). It will NOT catch unstructured PII
(a name in running prose, a home address written in free text) — for
that, upgrade to Microsoft Presidio (`pip install presidio-analyzer`,
free/local but needs a spaCy model download) which does proper NER-based
detection. This module is deliberately kept dependency-light as the
default; see the Presidio note at the bottom for the upgrade path.
"""

import json
import re

_PATTERNS = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    # (?<!\d) instead of a leading \b: \b requires a word/non-word transition,
    # but "(" is itself non-word, so a leading \b would force the match to
    # start at the digit right after "(" and leave the parenthesis behind.
    "PHONE": re.compile(r"(?<!\d)(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


def redact_pii(text: str, categories: list[str] = None) -> tuple[str, dict[str, int]]:
    """
    Returns (redacted_text, counts_by_category). Redacted spans are replaced
    with [REDACTED_<CATEGORY>] so the surrounding sentence structure stays
    intact for the embedder/LLM rather than leaving a gap.
    """
    categories = categories or list(_PATTERNS.keys())
    counts = {}
    redacted = text
    for category in categories:
        pattern = _PATTERNS.get(category)
        if pattern is None:
            continue
        matches = pattern.findall(redacted)
        # findall with no groups returns full matches; with groups it returns
        # group tuples — PHONE has an optional group, so count matches via finditer instead
        match_count = len(list(pattern.finditer(redacted)))
        if match_count:
            redacted = pattern.sub(f"[REDACTED_{category}]", redacted)
            counts[category] = match_count
    return redacted, counts


def redact_chunks(chunks: list, categories: list[str] = None) -> list:
    """
    In-place-style redaction over a list of Chunk objects: returns a new list
    with .text redacted and a `pii_redacted` metadata flag set where anything
    was actually found, so you can audit what got redacted later.

    `pii_redacted` is stored as a JSON string, not a nested dict — Chroma's
    metadata validation only allows str/int/float/bool/list/None values and
    rejects a dict outright:
        ValueError: Expected metadata value to be a str, int, float, bool,
        SparseVector, list, or None, got {'EMAIL': 1, 'PHONE': 1} which is a
        dict in upsert.
    This crashed the very first time --redact-pii was actually combined with
    content containing real PII and stored into Chroma. Read it back with
    `json.loads(chunk.metadata["pii_redacted"])` when auditing; every current
    caller (api.py, pipeline.py) only checks it for truthiness, which a
    non-empty JSON string still satisfies.
    """
    result = []
    for chunk in chunks:
        redacted_text, counts = redact_pii(chunk.text, categories)
        if counts:
            chunk.text = redacted_text
            chunk.metadata["pii_redacted"] = json.dumps(counts)
        result.append(chunk)
    return result


if __name__ == "__main__":
    sample = (
        "Contact John at john.doe@example.com or (555) 123-4567. "
        "His SSN is 123-45-6789 and the server IP is 192.168.1.1."
    )
    redacted, counts = redact_pii(sample)
    print(f"original:  {sample}")
    print(f"redacted:  {redacted}")
    print(f"counts:    {counts}")
    print()
    print("For unstructured PII (names/addresses in free text), upgrade to Presidio:")
    print("  pip install presidio-analyzer presidio-anonymizer")
    print("  python -m spacy download en_core_web_lg")
