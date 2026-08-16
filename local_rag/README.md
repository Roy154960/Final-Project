# Local Multimodal RAG Pipeline

A fully local, free RAG pipeline that ingests text, PDFs, and images; lets you
benchmark multiple models/techniques at every stage; and generates grounded
answers with a local LLM. No paid API keys anywhere.

## Architecture

```
Ingest -> Chunk -> Embed -> Store -> Retrieve -> Generate
```

Each step lives in its own folder, with one file per method/backend so you
can test and compare them independently:

```
local_rag/
├── config.py                       # central paths + model registry
├── pipeline.py                     # CLI orchestrator (ingest / ask), wires everything below
├── api.py                          # REST API (FastAPI): POST /ingest, POST /query, GET /documents, GET /health
├── stages.py                       # run Ingest/Chunk/Embed/Store/Retrieve/Generate as 6 independent commands, bridged only by disk checkpoints
│
├── ingestion/
│   ├── schema.py                   # RawDocument / Chunk dataclasses used everywhere
│   ├── ingest_text.py              # .txt / .md
│   ├── ingest_pdf.py               # PyMuPDF: text + embedded images per page, + page-complexity flagging for VLM description
│   ├── ingest_image.py             # standalone images
│   ├── loader.py                   # dispatches a mixed folder to the right loader; merges in table_extraction.py's output for PDFs
│   ├── ocr_fallback.py             # OCR for scanned/image-only PDF pages (pytesseract; reference copy — ingest_pdf.py has the wired-in version)
│   ├── table_extraction.py         # PDF tables -> markdown, kept as their own chunks (pdfplumber) — wired into loader.py, on by default
│   ├── incremental_indexer.py      # hash-based manifest: only re-ingest new/changed files
│   ├── deduplication.py            # exact-hash + near-duplicate (embedding similarity) removal
│   ├── image_captioning.py         # VLM captions each extracted image for --multimodal's text-index dual-indexing
│   └── page_description.py         # VLM describes whole pages ingest_pdf.py flagged as visually complex (Strategy 3) — --multimodal, --page-vlm
│
├── chunking/
│   ├── fixed_size.py                # N tokens + overlap
│   ├── recursive.py                 # paragraph -> sentence -> word -> char
│   ├── sentence_based.py            # N sentences per chunk
│   ├── semantic.py                  # embedding-similarity topic-shift detection
│   ├── structure_aware.py           # markdown headings / PDF pages
│   ├── parent_child.py              # small chunks retrieved, larger parents generated from
│   └── benchmark_chunkers.py        # compares all 4 non-semantic methods
│
├── embeddings/
│   ├── base.py                      # common interface
│   ├── ollama_embedder.py           # nomic-embed-text, mxbai-embed-large
│   ├── hf_embedder.py               # all-MiniLM-L6-v2, bge-small/base
│   ├── clip_embedder.py             # open_clip — TRUE multimodal (image+text, one space)
│   ├── cache.py                     # disk-cached embedder wrapper, dedupes within AND across runs
│   └── benchmark_embedders.py       # latency + sanity check across all of the above
│
├── vectorstore/
│   ├── base.py
│   ├── chroma_store.py              # embedded, zero setup
│   ├── qdrant_store.py              # client-server, production-grade
│   └── benchmark_stores.py          # upsert/query latency, both stores
│
├── retrieval/
│   ├── vector_retriever.py          # pure semantic search
│   ├── hybrid_retriever.py          # BM25 + vector, fused with RRF
│   ├── reranker.py                  # cross-encoder re-scoring
│   ├── multi_query.py               # LLM paraphrases the question, fuses results across variants
│   ├── query_router.py              # rule-based: semantic vs metadata-filter vs keyword-hybrid
│   ├── contextual_compression.py    # LLM strips each chunk to only the relevant sentences
│   ├── image_retriever.py           # CLIP text-encoder query against the dedicated image store (--multimodal)
│   └── benchmark_retrieval.py       # precision@k / recall@k / MRR across strategies
│
├── generation/
│   ├── prompts.py                   # RAG prompt template
│   ├── ollama_generator.py          # llama3.2, mistral, phi3
│   ├── hf_generator.py              # local HF causal LM fallback
│   └── dual_modality_generator.py   # --multimodal: independent text/image drafts, abstention, synthesis
│
├── evaluation/
│   ├── metrics.py                   # precision/recall/MRR + free faithfulness heuristic
│   ├── ragas_eval.py                # RAGAS with a local Ollama judge (no OpenAI key)
│   └── build_eval_set.py            # interactive CLI to build a REAL labeled eval set
│
├── safety/
│   ├── prompt_injection.py          # pattern scan + safe context-wrapping for the prompt
│   └── pii_redaction.py             # regex-based redaction (emails/phone/SSN/CC/IP) pre-storage
│
├── slm/
│   ├── model_registry.py            # candidate small language models, both backends, with metadata
│   └── benchmark_slms.py            # latency / memory / faithfulness across all candidates
│
├── quantization/
│   ├── bitsandbytes_quant.py        # 4-bit/8-bit HF quantization (GPU required for the benefit)
│   ├── gguf_quant.py                 # Ollama GGUF quant-tag levels (CPU-friendly)
│   └── benchmark_quantization.py    # memory/latency/quality across quant levels
│
├── vlm/
│   ├── ollama_vlm.py                 # llava/moondream via Ollama — true image+text reasoning
│   ├── hf_vlm.py                     # moondream2 / Qwen2-VL-2B via local HF weights
│   └── benchmark_vlms.py            # latency + keyword-hit-rate across VLM backends
│
├── serving/
│   ├── vllm_offline.py               # vLLM batch inference (Python API), continuous batching
│   ├── vllm_openai_server.py        # client for a running vLLM OpenAI-compatible local server
│   └── benchmark_serving.py         # throughput: HF transformers vs Ollama vs vLLM batch
│
└── utils/
    ├── retry.py                     # exponential backoff around Ollama calls (tenacity)
    ├── logging_config.py            # structured JSON logging
    └── tracing.py                   # local Phoenix tracing, falls back to timed log spans
```

## Why these specific free/local choices

| Stage | Options in this repo | Notes |
|---|---|---|
| Embed (text) | Ollama (`nomic-embed-text`, `mxbai-embed-large`) vs HF (`all-MiniLM-L6-v2`, `bge-small/base`) | Both entirely free/local. Ollama is simpler to run; HF gives you more model choice and easier fine-tuning later. |
| Embed (image) | CLIP via `open_clip` | You asked for true multimodal — image and text land in the *same* vector space, so a text query can retrieve an image chunk directly, no OCR/captioning step needed. |
| Store | ChromaDB vs Qdrant | Chroma = zero setup, great for dev. Qdrant = closer to how you'd actually deploy this (needs `docker run -p 6333:6333 qdrant/qdrant` locally, still free). |
| Generate | Ollama (`llama3.2`, `mistral`, `phi3`) vs HF (`Qwen2.5-1.5B-Instruct`, `Phi-3-mini`) | Ollama is easiest for CPU-only machines; HF gives raw `transformers` access if you want to inspect logits/attention later. |

## What the real-corpus evaluation found

The choices above are all viable; the accompanying technical report ran
each one against a real document collection with a real, labeled question
set and selected a production configuration based on the results, not
assumption:

- **Chunking**: `parent_child` was selected. Separately, `structure_aware`
  has a known, unresolved bug: it produced one chunk of 85,938 words on
  at least one real-corpus page (average for that method is 395 words),
  pointing to a page-extraction problem rather than a flaw in the chunking
  rule itself.
- **Embedding**: `all-MiniLM-L6-v2` was selected, fastest of the working
  candidates and the cleanest separation between similar and unrelated
  text (a paraphrase-pair vs. unrelated-pair similarity gap of 0.746,
  well ahead of bge/nomic's 0.545-0.547). `mxbai-embed-large` failed
  outright on this corpus (context length exceeded), and `bge-base-en-v1.5`/
  `nomic-embed-text` cost three to four times more compute for a lower
  gap, with nothing in the data justifying the extra cost.
- **Vector store**: Chroma was selected. It answers queries roughly nine
  times faster than Qdrant on this workload, in-process with no network
  hop, while Qdrant writes roughly twice as fast; for a single-user system,
  query speed matters more than write speed.
- **Retrieval**: plain `vector` search was selected. It tied or beat
  `hybrid` and `multi_query` on precision and recall across two
  independent real question sets (precision 0.308/0.265, recall
  0.63/0.537), while being two to three orders of magnitude faster
  (roughly 17-20ms vs. roughly 1.7s for hybrid and 4.4s for multi_query).
  `router` produced results identical to plain vector search on both sets,
  since this corpus's real questions are almost entirely ordinary
  conceptual questions rather than the date-filtered or exact-phrase
  questions `router`'s other paths are built for.
- **Generation**: `llama3.2` is the leading candidate based on completed
  comparisons; `mistral` scored highest on faithfulness but was the
  slowest of the completed candidates. `Qwen2.5-1.5B-Instruct` and
  `Phi-3-mini-4k-instruct` had evaluation pending at time of writing.

## Setup

```bash
# 1. Python deps
pip install -r requirements.txt --break-system-packages   # or use a venv without the flag

# 2. Ollama (for the Ollama-backed embed/generate options)
# install from https://ollama.com, then:
ollama serve &
ollama pull nomic-embed-text
ollama pull mxbai-embed-large
ollama pull llama3.2
ollama pull mistral

# 3. Qdrant (only if you want to benchmark it against Chroma)
docker run -p 6333:6333 qdrant/qdrant

# 4. OCR fallback (only if you'll ingest scanned PDFs)
sudo apt-get install tesseract-ocr poppler-utils

# 5. Local tracing UI (optional)
# arize-phoenix is in requirements.txt already; utils/tracing.py falls back
# to plain timed log lines automatically if you don't start it

# 6. GPU-only extras (skip if you're CPU-only — everything else in this
# project runs without a GPU, these three specifically do not):
#    - quantization/bitsandbytes_quant.py (needs bitsandbytes' CUDA kernels)
#    - vlm/hf_vlm.py's qwen2-vl-2b option (moondream2 runs fine on CPU)
#    - serving/vllm_offline.py and serving/vllm_openai_server.py (vLLM core is GPU-only)
```

## Running it

```bash
# Drop some files into data/raw/ (txt, md, pdf, png/jpg) first, then:

# Basic ingest -> chunk -> embed -> store
python pipeline.py ingest --source data/raw --embedder hf --store chroma

# Basic ask
python pipeline.py ask "What does the document say about X?" \
    --embedder hf --store chroma --generator ollama --rerank
```

## Running each stage independently (persistent-disk checkpoints)

`pipeline.py ingest` fuses Ingest+Chunk+Embed+Store into one command, and
`pipeline.py ask` fuses Retrieve+Generate into another. If you want to run
**each of the six stages as its own separate command** — stop after any
one of them, come back later, run the next stage in a fresh process with
nothing held in memory — use `stages.py` instead. Every stage reads its
input from a file on disk and writes its output to a file on disk; no
stage depends on any Python object still being alive from a previous one.

```bash
python stages.py ingest --source data/raw          # -> data/checkpoints/01_raw_documents.json
python stages.py chunk --method sentence_based     # -> data/checkpoints/02_chunks.json
python stages.py embed --embedder hf               # -> data/checkpoints/03_text_vectors.npz
python stages.py store --store chroma              # -> upserts into the persisted vector store
python stages.py retrieve "your question" \
    --embedder hf --store chroma --retrieval hybrid  # -> data/checkpoints/04_retrieved.json
python stages.py generate --generator ollama       # -> data/checkpoints/05_answer.json

python stages.py status   # see which checkpoints currently exist
python stages.py clean    # wipe checkpoints (leaves the vector store itself untouched)
```

`chunk --method` accepts `auto` (default), `fixed_size`, `recursive`,
`sentence_based`, `semantic` (needs `--embedder` to find topic breaks), or
`structure_aware` — the same five chunking implementations
`chunking/benchmark_chunkers.py` compares, now actually reachable from the
real pipeline instead of only from that standalone comparison script.
`--parent-child` is a separate, orthogonal flag on this stage (a
hierarchical scheme, not one of the five flat methods).

You can shut your machine down between any two of these and pick up later
— each command is a fresh process that only reads what's on disk. Once
`store` has run, the vector store itself (`data/chroma_db/`, or Qdrant) is
the durable result of the first four stages — the JSON/`.npz` checkpoint
files aren't needed anymore for `retrieve`/`generate`, only for re-running
or inspecting an earlier stage.

One thing to keep consistent yourself: `retrieve` needs the **same**
`--embedder` value used in `embed`/`store` (different embedding models
produce incompatible vector spaces) — nothing enforces this automatically
across separate invocations, since that's exactly the point of not sharing
in-memory state.

## Running every method per stage, evaluated and compared (`load_step.py`)

`stages.py` (above) runs ONE method per stage per invocation — you pick
`--method recursive` or `--embedder hf` and it does exactly that, no
comparison. `load_step.py` is the companion for when you want the
comparison itself checkpointed: at each stage it runs **every** applicable
method against the same persisted input from the previous stage, scores
each one with that stage's existing metrics (the same ones
`chunking/benchmark_chunkers.py`, `embeddings/benchmark_embedders.py`,
`vectorstore/benchmark_stores.py`, `retrieval/benchmark_retrieval.py`, and
`slm/benchmark_slms.py` already use), and writes both the raw outputs and
a metrics summary to disk — so the "baseline vs improved, with numbers"
story writes itself as you go, and any stage is independently re-runnable
later exactly like `stages.py`'s stages are (its checkpoint files use a
different naming scheme, so both scripts can coexist without colliding).

```bash
python load_step.py ingest   --source data/raw
python load_step.py chunk                                              # all 6 chunking methods
python load_step.py embed    --chunk-method recursive                  # all embedder candidates
python load_step.py store    --chunk-method recursive --embedder "hf:sentence-transformers/all-MiniLM-L6-v2"   # both vector stores
python load_step.py retrieve --chunk-method recursive --embedder "hf:sentence-transformers/all-MiniLM-L6-v2" --store chroma   # all 4 retrieval strategies
python load_step.py generate --chunk-method recursive --embedder "hf:sentence-transformers/all-MiniLM-L6-v2" --store chroma --strategy hybrid   # all generator candidates

python load_step.py status   # every checkpoint + metric produced so far
python load_step.py all --source data/raw   # run all six stages in order, still checkpointing each
```

A candidate that isn't available (model not pulled, no Ollama/Qdrant
server running, etc.) is skipped with a printed reason rather than
crashing the whole stage — same defensive pattern used throughout this
project. Verified end-to-end against a real tiny corpus, including
confirming a later stage (`retrieve`) succeeds in a **completely fresh
process** using only what the earlier stages left on disk — no shared
Python state, by construction, not just by convention.

## REST API

Everything above is also reachable over HTTP through `api.py` (FastAPI),
which wires the exact same ingestion/chunking/embedding/retrieval/generation
modules the CLI uses — nothing is reimplemented for the API.

```bash
uvicorn api:app --reload --port 8000
```

Config is via environment variables (all optional, free/local defaults):

```bash
export RAG_EMBEDDER=hf        # hf | ollama | clip
export RAG_STORE=chroma       # chroma | qdrant
export RAG_GENERATOR=ollama   # ollama | hf | vlm   (needs `ollama serve` + `ollama pull llama3.2`;
                               #                     use `hf` if you'd rather not run Ollama)
```

Endpoints:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/ingest` | Upload a document (any format the CLI supports, incl. scanned PDFs — OCR runs automatically) → parse → chunk → embed → store |
| `POST` | `/query` | Ask a question → retrieve (vector / hybrid / router / multi_query) → optional rerank/compress → generate a grounded, source-attributed answer |
| `POST` | `/query/compare` | Run one question through **multiple retrieval strategies side-by-side** (retrieval only, no generation) — latency + top score + sources per strategy, so you can see which one actually performs best on your real, already-indexed corpus |
| `GET` | `/documents` | List distinct source files currently indexed, with chunk counts |
| `GET` | `/health` | Readiness check: embedder/store/generator config, index size |

Interactive API docs (via FastAPI's auto-generated Swagger UI) are at
`http://localhost:8000/docs` once the server is running.

```bash
curl -X POST "http://localhost:8000/ingest" -F "file=@data/raw/your_document.pdf"

curl -X POST "http://localhost:8000/ingest?chunking=sentence_based" -F "file=@data/raw/your_document.pdf"
# chunking: auto (default) | fixed_size | recursive | sentence_based | semantic | structure_aware
# same five methods as pipeline.py's --chunking / stages.py's chunk --method

curl -X POST "http://localhost:8000/query" -H "Content-Type: application/json" \
    -d '{"question": "What does the document say about X?", "rerank": true, "retrieval": "hybrid"}'

curl -X POST "http://localhost:8000/query/compare" -H "Content-Type: application/json" \
    -d '{"question": "What does the document say about X?", "strategies": ["vector", "hybrid", "router"]}'
```

`/query/compare` mirrors the "compare multiple methods, measure, then
decide" pattern the offline benchmark scripts already use for chunking/
embeddings/vector-stores — it just does it live, over HTTP, against
whatever you've actually indexed, instead of a toy corpus. `multi_query`
is opt-in (not in the default `strategies` list) since it costs an LLM
call per comparison; a strategy that errors (e.g. `multi_query` with no
Ollama server running) is reported in its own `error` field rather than
failing the whole request — the same defensive per-candidate pattern
`slm/benchmark_slms.py` already uses.

The embedder and vector store load once at startup; the generator (an LLM)
and the reranker (a cross-encoder) both load lazily on first use, so the
API comes up instantly even before Ollama is running or a model has been
downloaded, and neither is reloaded from scratch on every request.

### Production-hardened ingest

```bash
python pipeline.py ingest --source data/raw --embedder hf --store chroma \
    --incremental \        # only (re-)ingest new/changed files, skip the rest
    --dedup \               # drop exact-duplicate chunks
    --dedup-near \          # also drop near-duplicates (costs one extra embedding pass)
    --redact-pii \          # scrub emails/phones/SSNs/credit cards/IPs before storage
    --scan-injection \      # flag chunks that look like prompt-injection attempts (logged, not auto-dropped)
    --cache                 # cache embeddings on disk, skip re-embedding repeated content
```

### Choosing a chunking method

`pipeline.py ingest` (and `stages.py chunk`) previously always hardcoded
`recursive` for plain text and `structure_aware` page-as-unit for PDFs,
silently ignoring the other chunking methods this project actually
implements (`fixed_size`, `sentence_based`, `semantic`). That's fixed —
`--chunking` now selects any of them explicitly:

```bash
python pipeline.py ingest --source data/raw --chunking fixed_size
python pipeline.py ingest --source data/raw --chunking sentence_based
python pipeline.py ingest --source data/raw --chunking semantic   # needs an embedder to find topic breaks — uses --embedder
python pipeline.py ingest --source data/raw --chunking structure_aware  # markdown heading-based
python pipeline.py ingest --source data/raw --chunking auto        # default: previous hardcoded behavior
```

Compare them on your own corpus first with `chunking/benchmark_chunkers.py`
before picking one for real ingestion — see the Evaluation section below.
`--chunking` is ignored when `--parent-child` is set (parent-child is a
different, hierarchical kind of chunking, orthogonal to the five flat
methods above).

### Parent-child chunking

```bash
python pipeline.py ingest --source data/raw --embedder hf --store chroma --parent-child
python pipeline.py ask "..." --embedder hf --store chroma --parent-child
# retrieval matches small, precise child chunks; --parent-child on `ask`
# swaps each matched child for its larger parent before generation
```

### Advanced retrieval

```bash
# Hybrid search: BM25 (keyword) + vector (semantic), fused with Reciprocal
# Rank Fusion — catches exact names/codes/IDs pure vector search can miss
python pipeline.py ask "..." --retrieval hybrid

# Rule-based query routing: "what is X" -> vector, "show me all Y from 2024"
# -> metadata filter, exact codes/quotes -> keyword-hybrid
python pipeline.py ask "Show me all incidents from 2024" --retrieval router

# Multi-query: LLM paraphrases the question 3 ways, results fused with RRF
python pipeline.py ask "..." --retrieval multi_query

# Contextual compression: LLM strips each retrieved chunk to just the
# relevant sentences before generation (adds one LLM call per chunk)
python pipeline.py ask "..." --compress
```

### Building a real evaluation set

The retrieval benchmark is only as good as its eval data. Build one
interactively against your own indexed corpus:

```bash
python -m evaluation.build_eval_set --interactive --embedder hf --store chroma
# retrieves candidates for each question you type, you mark which are
# actually relevant, saves incrementally to data/eval_set.json

python -m evaluation.build_eval_set --stats data/eval_set.json
# quick sanity check: how many examples you have, whether it's enough
```

Once `data/eval_set.json` has 20-50 real examples,
`retrieval/benchmark_retrieval.py` picks it up automatically instead of
its one-example placeholder. The builder also deduplicates repeated
questions across separate labeling sessions, keeping whichever labeled
version of a duplicate question has more relevant chunks recorded.

### Scanned PDFs (two different problems, both handled automatically)

`ingestion/ingest_pdf.py` now handles both scanned-PDF cases you'll run
into, on by default, no flags needed:

- **Text layer already exists, but one or more full-page images sit
  underneath it** (e.g. archive.org exports) — this includes both a plain
  background scan AND, for Internet Archive files specifically, a
  distorted/skewed "page-curl" thumbnail their BookReader uses for its
  page-turn animation. Both are redundant with the text layer, and both are
  detected the same way: by checking how much of the page's actual area the
  image is *placed* to cover (`skip_full_page_scans=True`), not by its pixel
  dimensions — that's what catches the warped thumbnail (whose pixel aspect
  ratio looks nothing like a page) and low-resolution "reading copy" scans
  (which can be well under typical archival scan resolution) that a
  pixel-size check would miss.
- **No text layer at all — the words are baked into the pixels** (a true
  flat scan) — detected via an empty/near-empty native text extraction,
  and OCR'd automatically instead (`ocr_on_empty_pages=True`, needs
  `tesseract-ocr` installed, see Setup).

```bash
python -m ingestion.ingest_pdf data/raw/your_file.pdf
# prints which pages came from OCR vs native text, and how many
# redundant page-scan images were filtered out
```

Table extraction is a separate concern (structure, not text presence/absence),
and is wired into ingestion by default — every PDF gets both a flattened
`pdf_text` page (as always) and, for each detected table, a separate
`modality='pdf_table'` document (markdown format, via `pdfplumber`), kept
as its own unsplit chunk (`chunking/structure_aware.py:chunk_pdf_table_as_unit`)
so a table's rows/columns never get split mid-chunk. Run it standalone if
you just want to see what it extracts, independent of the rest of ingestion:

```bash
python -m ingestion.table_extraction data/raw/report_with_tables.pdf
```

If `pdfplumber` isn't installed, table extraction is skipped with a
warning rather than failing the whole ingest — see `ingestion/loader.py`.

### Whole-page VLM description (Strategy 3: Vision LLMs for Document Understanding)

Different problem from both of the above: a chart or diagram drawn as
*vector graphics* (a matplotlib/Excel/PowerPoint-style figure embedded
directly in the PDF) is invisible to `page.get_images()` — it was never a
raster image — and mostly meaningless as flattened text (a scatter of
disconnected axis labels and legend entries with no visual relationship
between them). `ingestion/ingest_pdf.py` + `ingestion/page_description.py`
close that gap by handing a VLM a full render of the page itself, on
`--multimodal` ingests.

This is deliberately split into a cheap half and an expensive half so it
doesn't blow up ingest time on ordinary documents:

- **Cheap, always-on, no model call** (`ingest_pdf.py`'s `describe_pages`,
  default `"auto"`): a pure PyMuPDF geometry check — vector-drawing count
  vs. native text length (`PAGE_VISUAL_COMPLEXITY_*` in `config.py`) —
  flags only pages that actually look chart/diagram-heavy. A page that
  already yielded a real embedded raster image is skipped here too (that
  image gets its own caption via `image_captioning.py`; describing the
  whole page as well would just be redundant). Flagged pages get a
  full-page PNG rendered to disk and `metadata["page_image_path"]` set —
  nothing else happens yet.
- **Expensive, one VLM call per flagged page, gated behind `--multimodal`**
  (`pipeline.py`'s `describe_complex_pages()`): only ever processes the
  pre-flagged subset from the cheap half. A 200-page mostly-text PDF with
  3 chart-heavy pages costs exactly 3 VLM calls here, not 200. The
  description is dual-indexed into the text store
  (`source_type: "page_visual_description"`), additive alongside that
  page's normal text chunk, same pattern as tables and image captions.

```bash
# Default: only chart/diagram-heavy pages get described
python pipeline.py ingest --source data/raw --multimodal --page-vlm auto

# Force every page (slow — one VLM call per page, useful for a one-off
# exhaustive run on a small, known-visual document)
python pipeline.py ingest --source data/raw --multimodal --page-vlm always

# Disable entirely — skip the heuristic and rendering, not just the VLM call
python pipeline.py ingest --source data/raw --page-vlm never
```

`POST /ingest?multimodal=true&page_vlm=auto` exposes the same three modes
over the REST API, reusing the same loaded VLM instance as image
captioning (see `api.py`'s `_get_caption_vlm`) so it isn't loaded twice.
`--page-vlm auto`/`always`/`never` is also available on `stages.py ingest`,
with the actual VLM call happening in `stages.py chunk --multimodal`
(mirroring where image captioning happens in that six-stage flow).

Tuning: if `auto` is firing too often or too rarely on your corpus, adjust
`PAGE_VISUAL_COMPLEXITY_MIN_DRAWINGS` / `PAGE_VISUAL_COMPLEXITY_MAX_TEXT_CHARS`
in `config.py` — both are starting points, not calibrated against a labeled
set.

## Benchmarking each stage

Run these after you've dropped a few real documents into `data/raw/` —
the numbers are only meaningful on content that resembles what you'll
actually use this for:

```bash
python -m chunking.benchmark_chunkers data/raw
python -m embeddings.benchmark_embedders
python -m vectorstore.benchmark_stores
python -m retrieval.benchmark_retrieval
```

`retrieval/benchmark_retrieval.py` now compares **all five** wired
retrieval strategies side by side — `vector_only`, `hybrid`,
`hybrid_plus_rerank`, `router`, and `multi_query` (skipped with a printed
reason if no generator/Ollama is reachable, same defensive pattern as
`slm/benchmark_slms.py`) — not just a subset. It still ships with a
placeholder eval set of one example — replace `EVAL_SET` in that file (or
just build `data/eval_set.json` via `evaluation.build_eval_set`) with real
(query, relevant chunk ids) pairs from your own corpus before trusting the
precision/recall/MRR numbers it prints. The same five strategies are also
reachable live over HTTP via `POST /query/compare` (see the REST API
section above) if you'd rather compare against your real indexed corpus
interactively instead of the toy one this script builds.

## SLMs, quantization, VLMs, and vLLM serving

### Small language models (SLM)

`slm/model_registry.py` catalogs 6 free/local candidates across both
backends (Ollama: phi3, llama3.2, mistral; HF: Qwen2.5-1.5B, Qwen2.5-0.5B,
Phi-3-mini-4k) with param count and context length, so you can reason about
tradeoffs before pulling anything.

```bash
python -m slm.benchmark_slms
# latency, load memory, tokens/sec, and a free faithfulness proxy across all 6
```

### Quantization

Two independent, genuinely different quantization paths — pick based on
your hardware:

```bash
# CPU-friendly: Ollama's GGUF quant tags (q4_0, q4_K_M, q5_K_M, q8_0, fp16...)
python -m quantization.gguf_quant          # shows example tags to pull
python -m quantization.benchmark_quantization

# GPU-only: BitsAndBytes 4-bit/8-bit for HF models — same benchmark script
# also runs this half if bitsandbytes + a CUDA GPU are available
```

`quantization/benchmark_quantization.py` runs both and reports memory delta,
latency, and faithfulness side by side, so you can see the real
speed/memory/quality tradeoff instead of trusting either library's
marketing numbers.

### VLMs (vision-language models)

This is the piece that makes image retrieval actually useful for
*answering* questions, not just finding similar images: CLIP
(`embeddings/clip_embedder.py`) retrieves the right image by similarity, a
VLM then looks at it and reasons over it.

```bash
python -m vlm.ollama_vlm data/raw/sample.png "What is shown in this image?"
python -m vlm.hf_vlm data/raw/sample.png "What is shown in this image?"
python -m vlm.benchmark_vlms data/raw/sample.png "What is shown in this image?"
```

Wired into the main pipeline as a generator option:

```bash
python pipeline.py ask "What does the chart show?" \
    --embedder clip --store chroma --generator vlm --vlm-backend ollama
```

`generation/multimodal_generator.py` splits retrieved results into text vs
image chunks, passes the top-scoring image directly to the VLM alongside
any retrieved text as context, and raises a clear error if you point it at
a text-only result set (use `--generator ollama`/`hf` for those instead).

### vLLM serving infrastructure

Two serving shapes, both requiring a CUDA GPU (vLLM's core kernels are
GPU-only — this is the one piece in the project that doesn't run CPU-only):

```bash
# Offline batch inference (one Python process, hand it many prompts at once)
python -m serving.vllm_offline

# Live server other apps/questions can hit over HTTP (start separately)
vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000
python -m serving.vllm_openai_server
python pipeline.py ask "..." --generator vllm-server
```

```bash
python -m serving.benchmark_serving
# compares questions/sec: HF transformers (sequential) vs Ollama (sequential)
# vs vLLM (true batch) on the same question set — the gap should widen as
# you raise N_QUESTIONS in that file, since that's where continuous
# batching's advantage actually shows up
```

## Reliability utilities

`pipeline.py` already uses these; reach for them directly in your own scripts too:

- `utils/retry.py` — wraps flaky Ollama calls with exponential backoff (`@with_retry`)
- `utils/logging_config.py` — structured JSON logs (`get_logger(__name__)`)
- `utils/tracing.py` — `with trace_span("name", **attrs):` around any stage;
  auto-detects a running local Phoenix instance, otherwise logs timed spans

## Incremental re-indexing and vector cleanup

`ingestion/incremental_indexer.py` detects new/changed/deleted files via a
content-hash manifest; `vectorstore/base.py` now has a `delete(ids=...)`
method (implemented for both Chroma and Qdrant) so `pipeline.py --incremental`
actually removes stale vectors for changed/deleted files before storing the
new ones, rather than leaving orphaned entries behind.

## Dual-modality retrieval (`--multimodal`)

`--embedder clip` (above) embeds *everything* — text and images — into one
CLIP vector space. That's true multimodal, but it means text search quality
is capped by CLIP's own text encoder, which is meaningfully weaker at plain
semantic matching than a dedicated text embedder like `bge-base` or
`nomic-embed-text`. `--multimodal` is a separate, independent-indexes design
instead:

- Text chunks still go through whichever `--embedder` you chose (hf/ollama),
  into their normal collection — no quality tradeoff for text.
- Images always go through CLIP (`embeddings/clip_embedder.py`) into their
  **own** collection (`config.CHROMA_IMAGE_COLLECTION` /
  `QDRANT_IMAGE_COLLECTION`), regardless of `--embedder`.
- Each image also gets a VLM-written caption (`ingestion/image_captioning.py`)
  that's embedded via the TEXT embedder and dual-indexed into the TEXT
  collection — so an image can be found by a plain semantic query too, not
  only by CLIP's cross-modal similarity.
- Pages flagged as visually complex (vector-drawn charts/diagrams — see
  "Whole-page VLM description" above) also get a VLM-written description
  dual-indexed the same way (`ingestion/page_description.py`), additive
  alongside that page's own text chunk. Unlike image captions, this only
  runs for the (typically small) subset of pages the `--page-vlm` heuristic
  actually flags — not one call per page.

At query time, both indexes are searched independently and answered
independently — `generation/dual_modality_generator.py`'s
`DualModalityGenerator` drafts a text-branch answer and an image-branch
answer, drops either one that isn't actually viable (a retrieval-score
floor, then the model's own judgment via a strict sentinel it must return
if it can't really answer), and only calls a synthesis LLM to combine them
if both survived. If neither did, you get a fixed "not enough information"
message instead of a guess — see the module's docstring for the full
reasoning.

```bash
python pipeline.py ingest --source data/raw --embedder hf --store chroma --multimodal
python pipeline.py ask "Which figure illustrates glazing?" \
    --embedder hf --store chroma --multimodal --generator ollama

# same flag on stages.py (chunk captions+dual-indexes, embed always uses
# CLIP for images, store routes them into a separate collection, retrieve
# adds the image branch, generate wraps the DualModalityGenerator):
python stages.py chunk --multimodal
python stages.py embed --multimodal
python stages.py store --multimodal
python stages.py retrieve "Which figure illustrates glazing?" --multimodal
python stages.py generate --multimodal

# and over HTTP:
curl -X POST "http://localhost:8000/ingest?multimodal=true" -F "file=@data/raw/your_file.pdf"
curl -X POST "http://localhost:8000/query" -H "Content-Type: application/json" \
    -d '{"question": "Which figure illustrates glazing?", "multimodal": true}'
```

`--generator vlm` is incompatible with `--multimodal` (the image branch is
already handled internally) — use `--generator ollama` or `hf`; passing
`vlm` raises a clear error instead of silently doing the wrong thing.

Two starting-point constants worth tuning against your own corpus once you
have real retrieval scores to look at: `config.TEXT_RELEVANCE_SCORE_THRESHOLD`
(0.35) and `config.IMAGE_RELEVANCE_SCORE_THRESHOLD` (0.20) — the image
threshold is deliberately lower since CLIP's cross-modal similarity runs in
a lower, narrower band than same-modality text similarity even for a
genuine match.

## What's stubbed vs what's real

Every file here is real, runnable code — not pseudocode. I verified
everything that doesn't require a model download, a GPU, or a running
Ollama/Qdrant/vLLM server directly in this sandbox: chunking algorithms,
parent-child resolution, the query router's classification logic, PII
redaction, prompt-injection scanning, the embedding cache's dedup behavior
(including catching and fixing a real bug where same-batch duplicates
weren't deduped), the multimodal generator's text/image chunk-splitting
logic (including catching the no-image-chunks error case), retry/backoff,
structured logging, and tracing spans all ran and produced correct output.

What I couldn't run here — actual embedding/generation calls, OCR (needs
system tesseract), table extraction (needs a real PDF with tables), Qdrant
(needs a running server), BitsAndBytes and vLLM (both need a CUDA GPU) —
you should smoke-test on your machine first before relying on them. The
GPU-only pieces (`quantization/bitsandbytes_quant.py`,
`serving/vllm_offline.py`, `serving/vllm_openai_server.py`) are the one
part of this project that genuinely can't run CPU-only; everything else,
including the CPU-friendly GGUF quantization path and moondream2 VLM, works
without one.

## Known limitations (from the real-corpus evaluation)

Stated directly rather than omitted, per the accompanying technical report:

- **`structure_aware` chunking** has the unresolved extraction bug noted
  above, affecting at least one page in the real corpus.
- **Image-context extraction** (attaching the page text near an image to
  that image's own metadata) was described in early project notes but
  could not be confirmed as implemented in the delivered code.
- **RAGAS-based full-pipeline evaluation** (`evaluation/ragas_eval.py`)
  has not yet been exercised against real project data.
- **Two of five generation-model candidates** (`Qwen2.5-1.5B-Instruct`,
  `Phi-3-mini-4k-instruct`) had evaluation pending at time of writing.
- **Multimodal (`--multimodal`) retrieval** has not yet been benchmarked
  against a real, labeled multimodal question set the way the text-only
  retrieval strategies were above; the dual-index design and its
  abstention/synthesis behavior have been verified with unit and
  integration tests against mocked models, but not yet measured for
  precision/recall against real questions with real image content.

## Bugs found and fixed while adding the REST API

Building `api.py` and testing it end-to-end (with a mocked embedder/store/
generator, since this sandbox has no access to HuggingFace/Ollama) surfaced
four real bugs already present in the pipeline — worth knowing about for
the report's "how I improved it" section:

1. **No chunker propagated document-level metadata onto its chunks.**
   `fixed_size.py`, `recursive.py`, `sentence_based.py`, `semantic.py`,
   `structure_aware.py`, and `parent_child.py` all built `Chunk` objects
   without forwarding `doc.metadata` (which is where `filename`, `page`, and
   `extraction_method` live). In practice this meant
   `generation/prompts.py`'s citation logic — `chunk.get("metadata",
   {}).get("filename", "unknown source")` — fell back to "unknown source"
   for every single generated answer, regardless of chunking method. Fixed
   by spreading `doc.metadata` into every `Chunk.new(...)` call.
2. **`HybridRetriever.retrieve()` never returned `metadata` at all** —
   only `id`, `text`, `score`. Every hybrid-search result therefore lost
   source attribution downstream, independently of bug #1. Fixed by
   tracking a `meta_by_id` map (mirroring what `multi_query.py` already did
   correctly) and including it in the fused output.
3. **`--retrieval hybrid` didn't exist on the CLI** (`cmd_ask` raised
   `NotImplementedError`), and the `router` retrieval path would crash with
   `ValueError` the moment it picked the `keyword_hybrid` route, because
   nothing supplied `corpus_for_hybrid`. Fixed by adding `get_all()` to both
   vector stores (`ChromaStore`, `QdrantStore`) and a `hybrid_retrieve()`
   convenience wrapper, then wiring both into `pipeline.py` and `api.py`.
4. **`RawDocument`/`Chunk` IDs were `uuid.uuid4()` — random, regardless of
   content.** `store.upsert()` is ID-keyed, so this meant re-ingesting the
   identical file didn't update anything in place, it silently inserted a
   fresh duplicate copy under new IDs, every single time. On the CLI this
   was invisible unless you noticed the vector count creeping up; through
   `POST /ingest` it was easy to trigger by accident — upload the same file
   twice while testing and the index just grows, with no error and no
   signal anything's wrong. Confirmed directly: three identical uploads via
   `TestClient` produced three separate store entries with identical text.
   Fixed in `ingestion/schema.py` by deriving both IDs from content (a
   SHA-256 hash) instead of a random UUID — deliberately *not* from
   `source_path`, since `api.py` stages every upload under a fresh
   `uuid4()`-prefixed temp filename specifically to avoid collisions between
   concurrent uploads, which means the path itself is different on every
   upload even for byte-identical files. Images hash their actual bytes
   (read from `image_path`) rather than the path string, for the same
   reason. Re-verified the same three-upload scenario after the fix:
   converges to one entry, while genuinely different content still gets its
   own distinct entry — and a full `stages.py` six-stage rerun on an
   unchanged corpus no longer doubles the store either, which was a bonus
   catch beyond the original API-specific symptom.

None of this is hypothetical — a 6-line pytest-style smoke test (fake
embedder/store/generator, no real models needed) reproduced all four
before the fix and passed cleanly after. If you're writing the report's
"where I started / what I saw failing" section, this is a genuine,
reproducible before/after you can cite instead of a synthetic one.

## Changelog

### Multimodal Documents in RAG — table extraction wired in, whole-page VLM description added

Prompted by an audit of "advanced RAG" strategies (better text extraction,
OCR, vision-LLM document understanding, multimodal embeddings) against
what this pipeline actually does end-to-end. Two real gaps found and
closed:

**1. Table extraction (`ingestion/table_extraction.py`) is now wired in — always on.**

Previously the code existed (pdfplumber → markdown tables, with `page`/
`table_index`/`n_rows`/`n_cols` metadata) but was never called by
`loader.py`, `pipeline.py`, `api.py`, or `stages.py` — only runnable
standalone. Every PDF ingest silently lost table row/column structure,
flattened into paragraph text via `page.get_text()`.

- `ingestion/schema.py`: added `"pdf_table"` to `RawDocument.modality`'s
  type hint (was already being used at runtime without ever being declared).
- `ingestion/loader.py`: `ingest_path()`/`ingest_directory()` now also
  call `extract_tables()` for every PDF and merge the results in. Wrapped
  in try/except — a missing `pdfplumber` install degrades to "no separate
  table chunks this run" instead of breaking ingestion.
- `chunking/structure_aware.py`: new `chunk_pdf_table_as_unit()` — one
  chunk per table, never split mid-row.
- `pipeline.py` (`cmd_ingest`), `api.py` (`/ingest`), `stages.py`
  (`cmd_chunk`): added the `modality == "pdf_table"` dispatch branch
  (all three had their own independent copy of this chunking dispatch
  logic, so all three needed the fix). Tables are additive alongside the
  page's own flattened text, not a replacement for it — same pattern
  used for image captions.
- `api.py`: `IngestResponse` gained `n_table_chunks`.

**2. Whole-page VLM description added (Strategy 3: Vision LLMs for Document Understanding) — off by default cost, on by default coverage.**

Not implemented at all before this change. The gap: a chart/diagram drawn
as *vector graphics* (matplotlib/Excel/PowerPoint-style output embedded
directly in a PDF) is invisible to `page.get_images()` — it's never a
raster image — so neither native text extraction nor the existing
per-image VLM captioning ever saw it. The concern raised against adding
this straightforwardly: running a VLM call on every single page of every
PDF is slow and expensive at any real document count. Resolved by
splitting the feature into a cheap always-on half and an expensive
opt-in half:

- `config.py`: added `PAGE_DESCRIPTION_PROMPT` and two tunable thresholds,
  `PAGE_VISUAL_COMPLEXITY_MIN_DRAWINGS` / `PAGE_VISUAL_COMPLEXITY_MAX_TEXT_CHARS`.
- `ingestion/ingest_pdf.py`: new `_page_looks_visually_complex()` — a
  pure PyMuPDF geometry check (vector-drawing count vs. native text
  length), no model call, runs on every page for free. New
  `describe_pages: Literal["auto","always","never"] = "auto"` parameter;
  only pages the heuristic (or an explicit `"always"`) flags get a
  full-page PNG rendered and `metadata["page_image_path"]` set. Verified
  against a synthetic two-page PDF (one text-heavy page, one page with
  ~70 vector-drawn lines/rects and sparse text): the heuristic correctly
  flagged only the chart-like page under `"auto"`, flagged both under
  `"always"`, and flagged neither under `"never"`.
- `ingestion/page_description.py` (new file): the actual VLM call,
  structurally mirroring `image_captioning.py` — same fail-soft pattern
  (a bad page returns `""` and a warning instead of aborting the batch),
  same VLM backends (Ollama/HF). Only ever processes the pre-flagged
  subset handed to it; never decides which pages to run on itself.
- `pipeline.py`: new `build_page_description_chunks()` (pairs a flagged
  page with its description, additive alongside that page's normal text
  chunk — never a replacement) and `describe_complex_pages()` (filters
  to the flagged subset, calls the VLM, dual-indexes into the text
  store). Wired into `cmd_ingest`, gated behind `--multimodal` — the same
  flag that already gates VLM usage for image captions, so a user who
  doesn't opt into multimodal ingestion pays zero VLM cost for this,
  exactly like images. New `--page-vlm {auto,always,never}` CLI flag
  (default `auto`).
- `api.py`: `/ingest` gained a `page_vlm` query param (same three modes,
  same default) and calls `describe_complex_pages()` when
  `multimodal=true`, reusing the same cached VLM instance as image
  captioning (`_get_caption_vlm`) rather than loading a second one.
  `IngestResponse` gained `n_pages_flagged_for_description` and
  `n_page_descriptions_indexed`.
- `stages.py`: `cmd_ingest` threads `--page-vlm` through to
  `ingest_directory()`; `cmd_chunk --multimodal` VLM-describes the
  flagged subset and dual-indexes it, mirroring exactly where and how
  image captioning already happens in that stage (embedding/storing
  still happens in the later `embed`/`store` stages, same as captions).

Cost model in practice: a 200-page mostly-text PDF with 3 chart-heavy
pages costs 3 VLM calls under `--page-vlm auto`, not 200 — the expensive
half (`describe_complex_pages()` / `page_description.py`) never sees an
unflagged page.

**Type hints:** added/corrected throughout every file touched above —
`BaseEmbedder`/`BaseVectorStore` return types on `get_embedder()`/
`get_store()`/`get_image_store()` in `pipeline.py` (previously untyped),
`Optional[str]` fixes where a `None` default had been typed as bare `str`
(`get_generator()`'s `vlm_model`, `ingest_pdf()`'s `image_out_dir`), and
full signatures on every new function.

**Testing:** all of the above was verified by actually running it, not
just read — installed the previously-absent dependencies in this sandbox
(PyMuPDF, pdfplumber, nltk, langchain-text-splitters, fastapi,
python-multipart) and: generated synthetic PDFs (one with a real
pdfplumber-detectable table, one with a text page + a vector-drawn
chart-like page) and ran them through the real ingestion/chunking code;
exercised `page_description.py` and `describe_complex_pages()` with stub
VLM/embedder/store objects to confirm prompt correctness, the fail-soft
path, and correct filtering to only the flagged subset; confirmed
`pipeline.py`'s and `stages.py`'s CLI argument parsers correctly validate
`--page-vlm`; imported `api.py` for real and confirmed via its live
OpenAPI schema that `page_vlm` is registered as a documented query
parameter and `IngestResponse`'s new fields are present. `python -m
compileall` passes clean across the whole package.

**Known gap, not addressed here:** the report-generation/evaluation
modules (`evaluation/`) don't yet have labeled examples exercising table
chunks or page-description chunks specifically — same caveat as the
existing multimodal image-retrieval note above (dual-index design
verified with unit/integration tests against mocked models, not yet
measured for precision/recall against a real labeled question set).
