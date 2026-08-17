"""
Central configuration for the local RAG pipeline.
Every model/backend used anywhere in this project is FREE (no paid API
keys required) -- either via a local Ollama server, locally-downloaded
Hugging Face weights, or Groq's hosted free tier (the one deliberate
hosted exception, see the Groq section below -- and even that always
falls back to a local Ollama model automatically if unset/unreachable).
"""

from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DOCS_DIR = DATA_DIR / "raw"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"
BENCHMARK_RESULTS_DIR = DATA_DIR / "benchmark_results"

# Where local_rag/personal_rag.py persists the actual bytes of an
# uploaded IMAGE (never a PDF -- PDFs stay text-only there), one
# subdirectory per thread_id. Separate from RAW_DOCS_DIR on purpose:
# RAW_DOCS_DIR is a staging area agents/api.py's upload endpoint always
# cleans up after ingestion (see that endpoint's own comment on why),
# while this directory is the thing that makes "show the image the
# person actually sent, captioned, in the chat" possible at all -- the
# raw bytes have to live somewhere past the staging copy's deletion, or
# there is nothing left to render. personal_rag.delete_thread_data()
# removes a thread's own subdirectory here, same "nothing uploaded
# outlives its conversation" guarantee it already gives the Chroma side.
PERSONAL_UPLOADS_DIR = DATA_DIR / "personal_uploads"

for d in (DATA_DIR, RAW_DOCS_DIR, CHROMA_PERSIST_DIR, BENCHMARK_RESULTS_DIR, PERSONAL_UPLOADS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Ollama (local server, free models pulled with `ollama pull <name>`)
# ---------------------------------------------------------------------------
# Overridable via env var so a containerized backend can reach an Ollama
# server that ISN'T on its own localhost -- e.g. Docker Desktop's
# "http://host.docker.internal:11434" to reach an Ollama already running
# on the host machine (docker-compose.yml sets this), or a Linux host
# where that hostname doesn't resolve and the host's own LAN/bridge IP
# has to be used instead. Defaults to the original hardcoded value so
# every non-Docker workflow (`python -3.12 -m agents.<module>` from the
# project root, same as always) is completely unaffected.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Overridable via env var so a smaller/larger context window can be set
# without editing this file -- every Ollama call in this project
# (agents/llm_provider.py's fallback, generation/ollama_generator.py,
# vlm/ollama_vlm.py) reads THIS shared default rather than each
# hardcoding its own number, so there's one place to change if it's ever
# wrong for your hardware.
#
# Why this exists at all: Ollama's own default context length when
# nothing overrides it isn't a fixed, conservative number -- it can be
# the MODEL's own max trained context (some models support 128K+),
# which asks for a KV-cache buffer far bigger than most machines have
# free RAM for, and fails as an out-of-memory crash at model-load time
# rather than a clean error. 4096 is a safe, generous-for-a-chatbot
# default; raise it if your corpus/answers need longer context and your
# hardware can afford it, lower it if you're still hitting OOM at 4096.
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "4096"))

OLLAMA_EMBED_MODELS = [
    "nomic-embed-text",   # 768d, 8192 tokens, general purpose
    "mxbai-embed-large",  # 1024d, 512 tokens, higher quality
]

OLLAMA_GENERATION_MODELS = [
    "llama3.2",   # good general-purpose, small footprint
    "mistral",    # strong reasoning for size
    "phi3",       # fast, small, decent quality
]

# ---------------------------------------------------------------------------
# Hugging Face (downloaded locally the first time, then cached — all free)
# ---------------------------------------------------------------------------
HF_EMBED_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",   # 384d, fast, small
    "BAAI/bge-small-en-v1.5",                   # 384d, strong MTEB for size
    "BAAI/bge-base-en-v1.5",                    # 768d, better quality
]

HF_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # free cross-encoder

HF_GENERATION_MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",  # small, free, runs on CPU reasonably
    "microsoft/Phi-3-mini-4k-instruct",
]

# ---------------------------------------------------------------------------
# VLMs (vision-language models) - free, local
# ---------------------------------------------------------------------------
OLLAMA_VLM_MODELS = [
    "llava",       # strong general-purpose VLM, ~7B, needs a decent GPU or patience on CPU
    "moondream",   # tiny (~1.8B), fast, built for CPU/edge use
]

HF_VLM_MODELS = {
    "moondream2": "vikhyatk/moondream2",
    "qwen2-vl-2b": "Qwen/Qwen2-VL-2B-Instruct",
}

# ---------------------------------------------------------------------------
# Personal-RAG single-image captioning backend
# ---------------------------------------------------------------------------
# Which VLM backend captions a personal-RAG upload that's a SINGLE,
# stand-alone image (see personal_rag.py's own docstring for exactly what
# counts as "single") -- "groq" (below) is the default, and already
# falls back to the local "ollama" VLM automatically on any failure (see
# vlm/fallback_vlm.py) or if GROQ_API_KEY simply isn't set at all. Valid
# values are just "groq" and "ollama" -- set this to "ollama" yourself to
# opt out of the hosted path entirely, e.g. if you'd rather keep every
# request local even for single images. Multi-image PDF uploads always
# use the local "ollama" VLM regardless of this setting, specifically to
# avoid burning a free-tier hosted API's rate limit on a document with
# many embedded figures.
PERSONAL_RAG_SINGLE_IMAGE_VLM_BACKEND = "groq"

# ---------------------------------------------------------------------------
# Groq (hosted, FREE tier) -- the new first-choice backend for every
# reasoning/generation/vision call in this project that used to go
# straight to a local Ollama model. See groq_client.py's own module
# docstring for the shared HTTP layer every Groq integration goes
# through, generation/fallback_generator.py + vlm/fallback_vlm.py +
# agents/llm_provider.py for the three call-site wrappers that actually
# use it, and usage_tracker.py for where the free-tier rate-limit
# numbers (shown at the top of the chat UI) and the dev-only cost log
# get written.
#
# The whole point of "fallback": leaving GROQ_API_KEY unset, or Groq
# being down/rate-limited, never breaks anything -- every wrapper above
# just always ends up using the local Ollama model instead. Groq is the
# ONLY hosted/online backend in this project -- every other model here
# is local Ollama, no exceptions.
#
# Get a free key (no card required) at https://console.groq.com/keys,
# put it in your .env file at the project root:
#     GROQ_API_KEY=your-real-key-here
# and restart whichever server process picks it up (agents/api.py,
# mcp_server/server.py) so this file's own load_dotenv() call sees it.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# "Large"/"small" text reasoning tiers -- mirrors agents/specialists.py's
# own _LARGE_REASONING_MODEL ("llama3.2" locally) / _SMALL_REASONING_MODEL
# ("phi3" locally) split, see that module's "Model routing by difficulty"
# section for the rationale this project already applies.
#
# UPDATED 2026-08 after a confirmed live-run failure, not a hypothetical
# one: the original values here (llama-3.3-70b-versatile,
# llama-3.1-8b-instant) started returning HTTP 404 "model_not_found" on
# every single call -- not the 429 rate-limit this project's retry/
# fallback logic is designed around, a hard "this model doesn't exist
# anymore." Confirmed against Groq's own deprecations page
# (console.groq.com/docs/deprecations): Groq announced on 2026-06-17
# that BOTH of the original model IDs above were being decommissioned,
# with a full cutoff "by August 2026" -- which is now. Every Groq call
# in this whole project (System A's own agents/llm_provider.py AND
# System B's framing_agent/agent.py, which has its own separate default
# matching this one) was silently falling straight to local Ollama on
# EVERY call as a result, not just under real rate-limit pressure --
# which is what made local generation feel so much slower than usual:
# it wasn't occasional overflow to the fallback, it was 100% of calls
# landing there.
#
# Replaced with Groq's own official recommended replacements from that
# same deprecation notice -- not a guess at a plausible-sounding model
# name: "We recommend migrating to openai/gpt-oss-20b (for Llama 3.1 8B
# Instant) and openai/gpt-oss-120b ... (for Llama 3.3 70B Versatile)."
# If Groq deprecates THESE too down the line, the exact same failure
# shape will recur (a 404, not a 429) -- check
# https://console.groq.com/docs/deprecations first if Groq starts
# looking unreliable again; it may not be a rate limit either.
#
# Current free-tier RPM/RPD/TPM for both at
# https://console.groq.com/docs/rate-limits (checked 2026-08) -- both
# are comfortably generous for one person's own testing, which is the
# only use case this project's chatbot needs to cover.
GROQ_LARGE_MODEL = "openai/gpt-oss-120b"
GROQ_SMALL_MODEL = "openai/gpt-oss-20b"

# Groq's own hosted multimodal (text + vision) model -- see
# https://console.groq.com/docs/vision for current support, limits, and
# whether a newer/cheaper vision model has since been added to Groq's
# catalog (checked 2026-08: qwen/qwen3.6-27b is Groq's only vision-capable
# model at time of writing).
GROQ_VISION_MODEL = "qwen/qwen3.6-27b"

# ---------------------------------------------------------------------------
# CLIP (true multimodal embeddings — image and text share one vector space)
# Using open_clip, fully local, no API key.
# ---------------------------------------------------------------------------
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"  # free open-source weights

# ---------------------------------------------------------------------------
# Vector stores
# ---------------------------------------------------------------------------
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION = "rag_chunks"
CHROMA_COLLECTION = "rag_chunks"

# How vectorstore/chroma_store.py's ChromaStore reaches Chroma -- see that
# module's _build_chroma_client() for the two modes this switches between:
#   "embedded" (default): chromadb.PersistentClient straight against
#       CHROMA_PERSIST_DIR's on-disk SQLite file. Original, zero-setup
#       local-dev behavior -- completely unaffected unless this env var is
#       set. Fine for one process at a time; two independent processes
#       opening their own PersistentClient against the same path and
#       writing concurrently can hit SQLite's "database is locked".
#   "http": chromadb.HttpClient talking to a separate Chroma SERVER
#       process (docker-compose's chroma-server service, or `chroma run
#       --path ... --port ...` locally) over CHROMA_SERVER_HOST:
#       CHROMA_SERVER_PORT. That server owns the single SQLite connection
#       and every caller becomes a plain HTTP client with no direct file
#       access -- this is what actually removes the concurrent-write
#       "database is locked" risk two containers (backend + mcp-server)
#       sharing one Chroma volume can hit under real concurrent writes.
CHROMA_CLIENT_MODE = os.environ.get("CHROMA_CLIENT_MODE", "embedded").strip().lower()
CHROMA_SERVER_HOST = os.environ.get("CHROMA_SERVER_HOST", "localhost")
CHROMA_SERVER_PORT = int(os.environ.get("CHROMA_SERVER_PORT", "8000"))

# ---------------------------------------------------------------------------
# OCR (Tesseract)
# ---------------------------------------------------------------------------
# On Windows, installing Tesseract does NOT reliably put it on PATH, so
# pytesseract can't find the binary even after a successful install and a
# terminal restart. Set this to your actual install path if you hit
# "tesseract is not installed or it's not in your PATH" despite having
# installed it. Common Windows default:
#     TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# Leave as None on Linux/Mac, where it's normally found on PATH automatically.
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------------------------------------------------------------------------
# Chunking defaults
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE_TOKENS = 500
DEFAULT_CHUNK_OVERLAP_TOKENS = 100

# ---------------------------------------------------------------------------
# Dual-modality retrieval (pipeline.py --multimodal, embeddings/clip_embedder.py,
# generation/dual_modality_generator.py)
# ---------------------------------------------------------------------------
# Images get their own vector-store collection, kept separate from the text
# collection above (CHROMA_COLLECTION / QDRANT_COLLECTION) — this is what
# lets text chunks stay on a dedicated text embedder (better same-modality
# semantic quality) instead of being forced through CLIP's own, weaker,
# text encoder just so everything can share one index.
CHROMA_IMAGE_COLLECTION = "rag_images"
QDRANT_IMAGE_COLLECTION = "rag_images"

# Prompt used to caption each image at ingest time (ingestion/image_captioning.py).
# The caption is dual-indexed into the TEXT store (so images are reachable by
# plain text search too) and also stashed as metadata on the image's own CLIP
# entry (so the image branch of generation can hand it to the VLM as context).
IMAGE_CAPTION_PROMPT = (
    "Describe this image in detail: subject matter, style, composition, "
    "colors, and any visible text."
)

# ---------------------------------------------------------------------------
# Whole-page VLM description (ingestion/ingest_pdf.py's `describe_pages`,
# ingestion/page_description.py, pipeline.py's describe_complex_pages())
# ---------------------------------------------------------------------------
# Prompt used ONLY for pages flagged as visually complex (see
# PAGE_VISUAL_COMPLEXITY_* thresholds below) — deliberately broader than
# IMAGE_CAPTION_PROMPT above, since this is describing a whole page's
# layout/content (text + charts + diagrams + tables), not one extracted
# figure.
PAGE_DESCRIPTION_PROMPT = (
    "Describe ALL content on this page in detail, including any charts, "
    "diagrams, tables, or figures and what data or relationships they show."
)

# Cheap, local, no-model-call heuristic (ingestion/ingest_pdf.py's
# `_page_looks_visually_complex`) used by describe_pages="auto" (the
# default) to decide which pages are actually worth a VLM call, instead of
# running one on every page. A page needs BOTH: (a) at least this many
# vector-drawn paths (page.get_drawings()) — a strong signal of a
# matplotlib/Excel/PowerPoint-style chart or diagram drawn as vectors,
# which page.get_images() can never catch since it was never embedded as a
# raster image; and (b) native text under this length — so a normal
# text-heavy page that merely happens to contain a few decorative lines or
# a small table's grid (already handled separately by
# ingestion/table_extraction.py) doesn't trigger a VLM call it doesn't
# need. Both are starting points — tune them against your own corpus if
# --page-vlm auto is firing too often or too rarely (pass --page-vlm always
# or --page-vlm never to bypass the heuristic entirely while testing).
PAGE_VISUAL_COMPLEXITY_MIN_DRAWINGS = 25
PAGE_VISUAL_COMPLEXITY_MAX_TEXT_CHARS = 600

# Retrieval-score floors below which a branch is dropped before spending any
# LLM/VLM call on it (see DualModalityGenerator). CLIP's cross-modal
# (text-query-vs-image) cosine similarities run in a lower, narrower band
# than same-modality text-vs-text similarities even for a genuine match, so
# the image threshold is deliberately lower rather than sharing one constant.
# Both are starting points — tune them against your own corpus once you have
# real retrieval scores to look at (e.g. via `POST /query/compare`-style
# inspection or plain print output from pipeline.py ask --multimodal).
TEXT_RELEVANCE_SCORE_THRESHOLD = 0.35
IMAGE_RELEVANCE_SCORE_THRESHOLD = 0.20

# ---------------------------------------------------------------------------
# Personal (per-conversation) RAG -- local_rag/personal_rag.py,
# mcp_server/server.py's search_personal_documents tool,
# agents/api.py's POST /chat/{thread_id}/upload and DELETE /chat/{thread_id}
# ---------------------------------------------------------------------------
# Every image/PDF a person attaches inside a chat thread is ingested into
# this ONE shared Chroma collection -- literally named "temp" -- never a
# separate collection per thread. "Separated simply by an ID" means every
# chunk carries its owning thread_id as plain metadata (see
# personal_rag.py's THREAD_ID_METADATA_FIELD) and every search/delete
# filters on that field via Chroma's own `where`, rather than the heavier
# "one Chroma collection per thread" alternative, which would mean opening
# a fresh collection handle per conversation and its own separate cleanup
# pass for orphaned collections. Deleting a thread (DELETE
# /chat/{thread_id}) deletes every chunk whose metadata thread_id matches --
# see personal_rag.py's delete_thread_data -- so nothing from this
# collection outlives the conversation it was uploaded into.
PERSONAL_RAG_COLLECTION = "temp"
