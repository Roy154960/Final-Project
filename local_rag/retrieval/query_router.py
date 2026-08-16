"""
Retrieve enhancement - query routing.

Different question shapes want different retrieval strategies:
  - "What is X?" / "Explain Y" -> semantic vector search
  - "Show me all Y from 2024" / "List every Z in section 3" -> metadata filter
    (date, section, doc type...), optionally combined with vector search
  - "What does the report say verbatim about X" / exact codes, IDs, names
    -> keyword/hybrid search, since vector search blurs exact tokens

This starts with a fast, free, zero-inference regex/keyword router (no
LLM call, near-zero latency) and falls back to an LLM classifier only for
ambiguous cases, so you're not spending a generation call on every query.
"""

import re
from dataclasses import dataclass
from typing import Literal

RouteType = Literal["semantic", "metadata_filter", "keyword_hybrid"]

# Signals that suggest the user wants a filtered/enumerated set of items
# rather than a single explained answer.
_ENUMERATION_PATTERNS = [
    r"\ball\b", r"\bevery\b", r"\blist\b", r"\bshow me\b.*\b(all|every)\b",
]

# Signals that suggest exact-token matching matters (codes, IDs, names, quotes)
_EXACT_MATCH_PATTERNS = [
    r"\bverbatim\b", r"\bexact(ly)?\b", r'"[^"]+"', r"\b[A-Z]{2,}-\d+\b",  # e.g. "TICKET-1234"
]

_DATE_PATTERN = re.compile(r"\b(19|20)\d{2}\b|\bQ[1-4]\b|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\b", re.I)


@dataclass
class RouteDecision:
    route: RouteType
    metadata_filter: dict | None
    reason: str


def rule_based_route(question: str) -> RouteDecision:
    q_lower = question.lower()

    has_enumeration = any(re.search(p, q_lower) for p in _ENUMERATION_PATTERNS)
    has_date = _DATE_PATTERN.search(question)
    has_exact_match = any(re.search(p, question) for p in _EXACT_MATCH_PATTERNS)

    if has_enumeration and has_date:
        year_match = re.search(r"\b(19|20)\d{2}\b", question)
        metadata_filter = {"year": year_match.group(0)} if year_match else None
        return RouteDecision("metadata_filter", metadata_filter, "enumeration + date reference detected")

    if has_exact_match:
        return RouteDecision("keyword_hybrid", None, "exact code/name/quote detected — needs lexical matching")

    return RouteDecision("semantic", None, "default: open-ended question, semantic search fits best")


def route_and_retrieve(question: str, embedder, store, corpus_for_hybrid: list[dict] = None, top_k: int = 5):
    """
    Convenience wrapper: routes the question, then calls the matching
    retrieval function. `corpus_for_hybrid` is required only if the route
    resolves to keyword_hybrid (see retrieval/hybrid_retriever.py).
    """
    decision = rule_based_route(question)

    if decision.route == "semantic":
        from retrieval.vector_retriever import vector_retrieve
        results = vector_retrieve(question, embedder, store, top_k=top_k)

    elif decision.route == "metadata_filter":
        from retrieval.vector_retriever import vector_retrieve
        # Vector search narrowed by a metadata `where` clause — falls back to
        # unfiltered semantic search if the store doesn't support `where` or
        # the filter comes back empty. Requires your store's records to
        # actually carry that metadata field (e.g. "year") at ingest time.
        vec = embedder.embed_texts([question])[0]
        results = store.query(vec, top_k=top_k, where=decision.metadata_filter)
        if not results:
            results = vector_retrieve(question, embedder, store, top_k=top_k)

    else:  # keyword_hybrid
        if corpus_for_hybrid is None:
            raise ValueError("keyword_hybrid route needs corpus_for_hybrid (see HybridRetriever)")
        from retrieval.hybrid_retriever import HybridRetriever
        hybrid = HybridRetriever(embedder, store, corpus_for_hybrid)
        results = hybrid.retrieve(question, top_k=top_k)

    return results, decision


if __name__ == "__main__":
    test_questions = [
        "What is retrieval-augmented generation?",
        "Show me all incidents from 2024",
        'What does ticket "TICKET-4821" say verbatim?',
        "List every action item from Q3",
    ]
    for q in test_questions:
        decision = rule_based_route(q)
        print(f"{q!r}\n  -> route={decision.route}, filter={decision.metadata_filter}, reason={decision.reason}\n")
