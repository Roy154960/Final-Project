<!--
HOW TO USE THIS FILE
====================
This is a scaffold, not a finished report. Every section marked
    ▶ TODO (run: ...)
needs a real number pasted in from a command you run on your own machine
(this sandbox can't reach Ollama or download HF model weights, so none of
the numbers below could be generated here — filling them in is the one part
of this assignment I genuinely can't do for you).

Everything NOT marked TODO is already true and citable: the architecture
description, the bugs found and fixed, and the design reasoning are all
real, based on the actual code in this repo.

Once the TODOs are filled in, ask me to convert this to a polished Word
document (.docx) for submission — I can also fix a numbering/formatting
pass at that point.
-->

# Building a Production RAG System — Project Report

**Author:** Dominic
**Course:** [your course / assignment name]
**Repo:** local_rag_pipeline

---

## 1. System Overview

A fully local, fully free retrieval-augmented generation system — no paid
API keys anywhere. Every model is either served by a local Ollama instance
or downloaded once from Hugging Face and cached.

```
Ingest -> Chunk -> Embed -> Store -> Retrieve -> Generate
```

The corpus is [▶ TODO: describe your corpus — e.g. "N public-domain art and
painting treatises sourced from Archive.org, spanning English and French,
including scanned page images with no native text layer"].

### Required components (Part 1) — status

| Requirement | Status | Where |
|---|---|---|
| Ingest any document format | ✅ | `ingestion/loader.py` dispatches `.txt`/`.md`/`.pdf`/images |
| Handle scanned / image-only PDFs without silently returning empty text | ✅ | `ingestion/ingest_pdf.py`: OCR fallback (`ocr_on_empty_pages`) + full-page-scan-image filtering |
| End-to-end pipeline with source attribution | ✅ (was silently broken, see §3) | `pipeline.py`, `generation/prompts.py` |
| Vector DB | ✅ (two, benchmarked against each other) | `vectorstore/chroma_store.py`, `vectorstore/qdrant_store.py` |
| REST API (ingest + query endpoints) | ✅ | `api.py` (`POST /ingest`, `POST /query`) |
| Evaluation on a real dataset | ⚠️ tooling ready, needs a real run | `evaluation/build_eval_set.py`, `retrieval/benchmark_retrieval.py` |

### Optional components (Part 2) implemented

- **Multimodal**: CLIP (`embeddings/clip_embedder.py`) embeds images and text
  into one shared space — no separate captioning step needed for retrieval.
  A VLM (`vlm/ollama_vlm.py` / `vlm/hf_vlm.py`) then reasons over the
  retrieved image at generation time (`generation/multimodal_generator.py`).
- **Embeddings comparison**: HF (`all-MiniLM-L6-v2`, `bge-small/base`) vs
  Ollama (`nomic-embed-text`, `mxbai-embed-large`) — see §4.
- **Advanced retrieval**: hybrid BM25+vector (RRF fusion), cross-encoder
  reranking, multi-query expansion, query routing, parent-child chunking,
  contextual compression.
- **Multilingual**: [▶ TODO — you mentioned a French public-domain text for
  the multilingual component of the RAG lab; note here whether you used a
  multilingual embedding model or a language-specific one, and why].

---

## 2. Why These Specific Choices

*(Already answered well in README.md's "Why these specific free/local
choices" table — summarize the key reasoning here in your own words rather
than duplicating it verbatim.)*

[▶ TODO: 2-3 sentences per stage — e.g. why Chroma for dev vs Qdrant for
"production-like", why HF embeddings as the default over Ollama, why
Ollama for generation over raw HF `transformers`.]

---

## 3. The Journey

### 3.1 Where I started — baseline

Baseline configuration:
- Chunking: [▶ TODO — which method did you start with? `recursive` is the
  sensible default]
- Embedder: [▶ TODO]
- Store: [▶ TODO]
- Retrieval: `vector` only (no rerank, no hybrid)
- Generator: [▶ TODO]

Baseline eval numbers:

```
▶ TODO (run):
  python -m evaluation.build_eval_set --interactive --embedder hf --store chroma
  # answer 20-50 real questions against your own corpus, marking relevant chunks

  python -m retrieval.benchmark_retrieval
  # replace EVAL_SET in that file with your real eval_set.json first
```

| Metric | Baseline |
|---|---|
| Precision@5 | ▶ TODO |
| Recall@5 | ▶ TODO |
| MRR | ▶ TODO |
| Faithfulness (heuristic or RAGAS) | ▶ TODO |
| Avg. latency (retrieve+generate) | ▶ TODO |

### 3.2 What I saw failing, and how I reasoned about it

This is the section that actually makes the report worth reading — be
specific about *symptoms*, not just scores. Things you can genuinely draw
on from this project's real history:

- **Archive.org PDFs weren't yielding clean text.** Some pages had a full
  background scan image sitting underneath a perfectly good text layer
  (redundant), others had *no* text layer at all (pure scan) — two
  different failure modes that needed two different fixes
  (`skip_full_page_scans` vs `ocr_on_empty_pages` in `ingestion/ingest_pdf.py`).
  [▶ TODO: describe what a *bad* retrieval/answer looked like before this
  was fixed — e.g. "queries about page N returned nothing because the page
  had 4 characters of OCR noise as its only 'text'."]
- **Source attribution was silently broken.** While wiring up the REST API
  and testing it end-to-end, three separate bugs surfaced that had nothing
  to do with the API itself: none of the six chunking methods forwarded
  document-level metadata (`filename`, `page`) onto their output chunks,
  and `HybridRetriever.retrieve()` never returned `metadata` at all. Net
  effect: every generated answer's citations fell back to "unknown
  source", regardless of chunking method or retrieval strategy, and you'd
  never see it unless you specifically checked the `metadata` field on a
  retrieved chunk (the answer text still *looked* fine). See README.md's
  "Bugs found and fixed while adding the REST API" section for the exact
  diffs. This is a good example of the gap between "the demo works" and
  "the system is correct" — worth a paragraph in your report as a genuine
  dead-end/lesson.
- [▶ TODO: your own dead ends — e.g. a chunking strategy that scored worse
  than expected, an embedding model that was too slow, a reranker that
  didn't move the needle, semantic chunking's threshold needing tuning,
  etc. Pull from `chunking/benchmark_chunkers.py` and
  `embeddings/benchmark_embedders.py` output.]

### 3.3 How I improved it — each change, why, and its effect on the numbers

Fill in one row per real change you made and measured. Use
`retrieval/benchmark_retrieval.py` (rerun after each change) as your
consistent yardstick so the deltas are comparable — it now compares all
five wired strategies (`vector_only`, `hybrid`, `hybrid_plus_rerank`,
`router`, `multi_query`) in one run. You can also hit `POST
/query/compare` on the running API to get the same side-by-side comparison
live against your real indexed corpus instead of the toy one the script
builds.

| Change | Why | Precision@5 | Recall@5 | MRR | Notes |
|---|---|---|---|---|---|
| Baseline (vector only) | — | ▶ | ▶ | ▶ | |
| + Hybrid (BM25+vector, RRF) | Catches exact terms/names pure vector search misses | ▶ | ▶ | ▶ | ▶ TODO: did it actually help on *your* corpus? Art treatises have a lot of proper nouns (pigment names, artist names) — good candidate for hybrid to matter |
| + Cross-encoder reranking | Precision over a wider recall net | ▶ | ▶ | ▶ | |
| + Query routing | Routes enumeration/date questions to metadata filters, exact-quote questions to keyword-hybrid, everything else to semantic | ▶ | ▶ | ▶ | does routing actually change the winning strategy per question type, or mostly land on semantic anyway? |
| + Multi-query expansion | Paraphrases the question 3 ways, fuses results — costs one extra LLM call per query | ▶ | ▶ | ▶ | worth the latency? compare against hybrid_plus_rerank's cost/benefit |
| + Parent-child chunking | Small chunks retrieve precisely, larger parents give the LLM enough context | ▶ | ▶ | ▶ | |
| Chunking method swap: [X] → [Y] | ▶ TODO | ▶ | ▶ | ▶ | see `chunking/benchmark_chunkers.py` |
| Embedding model swap: [X] → [Y] | ▶ TODO | ▶ | ▶ | ▶ | see `embeddings/benchmark_embedders.py` |

Also worth reporting even without a numeric eval-set score (these have
their own benchmark scripts with latency/memory/quality proxies):

- SLM comparison — `python -m slm.benchmark_slms` → ▶ TODO
- Quantization tradeoff — `python -m quantization.benchmark_quantization` → ▶ TODO
- VLM comparison (if you used image retrieval) — `python -m vlm.benchmark_vlms ...` → ▶ TODO
- Vector store latency — `python -m vectorstore.benchmark_stores` → ▶ TODO

---

## 4. Final System

### 4.1 Architecture

```
                     ┌────────────────────────────────────────────┐
                     │                  api.py (FastAPI)            │
                     │   POST /ingest        POST /query            │
                     └───────────────┬──────────────┬───────────────┘
                                     │              │
                     ┌───────────────▼───┐   ┌──────▼─────────────────┐
                     │  Ingest → Chunk    │   │ Retrieve → (rerank) →  │
                     │  (parent-child,    │   │ Generate                │
                     │   OCR, PII redact, │   │ (hybrid/router/multi-   │
                     │   injection scan)  │   │  query + source attr.) │
                     └───────────────┬───┘   └──────┬─────────────────┘
                                     │              │
                              ┌──────▼──────────────▼──────┐
                              │   Embed (HF / Ollama / CLIP) │
                              │   Store (Chroma / Qdrant)    │
                              └───────────────────────────────┘
```

[▶ TODO: replace with your own diagram if you'd like something more
specific — I can generate one as an artifact if useful.]

### 4.2 Final configuration

- Chunking: ▶ TODO
- Embedder: ▶ TODO
- Vector store: ▶ TODO
- Retrieval: ▶ TODO (e.g. "hybrid + cross-encoder rerank, top_k=5")
- Generator: ▶ TODO

### 4.3 Final scores

| Metric | Baseline | Final | Δ |
|---|---|---|---|
| Precision@5 | ▶ | ▶ | ▶ |
| Recall@5 | ▶ | ▶ | ▶ |
| MRR | ▶ | ▶ | ▶ |
| Faithfulness | ▶ | ▶ | ▶ |
| Avg. latency | ▶ | ▶ | ▶ |

### 4.4 Honest limitations

- [▶ TODO — e.g. eval set size, corpus coverage, anything you know is
  fragile (semantic chunking's fixed similarity threshold, PII regex
  missing unstructured PII, prompt-injection scanning being pattern-based
  and not a complete defense, etc. — see the docstrings in `safety/` for
  the exact caveats already documented in-code).]

---

## 5. How to Reproduce

```bash
pip install -r requirements.txt --break-system-packages
ollama serve &
ollama pull llama3.2
ollama pull nomic-embed-text

python pipeline.py ingest --source data/raw --embedder hf --store chroma
python pipeline.py ask "your question" --embedder hf --store chroma --retrieval hybrid --rerank

# or, over HTTP:
uvicorn api:app --reload --port 8000
```
