# mcp_server — Phase 1 (+ four new tools)

Wraps the Production RAG System's retrieval pipeline as an MCP server.
Originally two tools (`retrieve`, `generate_answer`) and one resource
(`corpus://documents`) for Phase 1; now six tools and two resources —
see "New tools" below for the four added on top.

## New tools (image retrieval, internet search, invoicing)

Added to support four new agent specialists (`image_qa`,
`painting_lookup`, `product_search`, `invoice` — see
`agents/README.md`). Each tool wraps a small, independently-testable
module (`image_tools.py`, `web_tools.py`, `invoice_tools.py`) rather than
inlining logic directly into `server.py` — same reasoning `server.py`
already gives for wiring against `retrieval/hybrid_retriever.py` instead
of reimplementing it here.

| Tool | Backing module | External service |
|---|---|---|
| `retrieve_images(query, k)` | `image_tools.py` | none — reuses the project's own CLIP image store + ingest-time VLM captions |
| `search_painting_online(painting_name)` | `web_tools.py` | Wikipedia REST API (free, no key) + `ddgs` general web search |
| `search_art_supplies(query, max_results)` | `web_tools.py` | `ddgs` general web search, restricted to `site:amazon.com` / `site:ebay.com` |
| `generate_invoice(items, customer_note)` | `invoice_tools.py` | none — pure Python arithmetic |

Plus one new resource, `policy://allowed-link-domains`, listing every
domain the two internet-facing tools above are allowed to return a link
from (see `local_rag/safety/domain_allowlist.py`) — exposed purely for
transparency, not needed for retrieval or generation.

**`search_painting_online` cleans its own query before looking anything
up** — `web_tools._clean_painting_query` strips common question-phrasing
wrappers ("tell me about", "who painted", etc.) before either the
Wikipedia summary lookup or the search-API fallback. Confirmed live-run
fix: passing a full question like "Tell me about the Mona Lisa" straight
through resolved to the WRONG Wikipedia page ("Mona Lisa Smile", the
2003 film) — the wrapper words were noise the search relevance ranking
had no way to know wasn't part of the subject. Fixed at the tool level
(not the calling specialist), so it protects every caller, not just
`painting_lookup_node`. See `test_new_tools_smoke.py`'s tests for the
cleaning regex and for confirmation it's actually wired into every
lookup this function makes.

**`search_art_supplies` returns one flat list — tiering happens on the
caller's side.** The `product_search` agent specialist (not this tool)
splits results into "beginner-friendly" and "professional-grade" buckets
(5 of each, where available) via a keyword + price-median heuristic —
call this tool with a higher `max_results` (e.g. 12) if you want enough
raw candidates for a caller-side split like that to have real material
to work with. See `agents/README.md`'s own section on this for the full
design and its stated heuristic limitations.

**Free, no paid API key**, matching this project's own "everything free
and local" framing (`config.py`'s docstring) — Wikipedia's REST API and
`ddgs` are both free and keyless. Install the extra deps:

```bash
pip install -r requirements.txt   # now also installs requests + ddgs
```

**Honest limitations, worth stating directly in a report rather than
glossing over**:
- `search_art_supplies`'s `price` field is a best-effort regex extraction
  from a search-result *snippet*, not a live price lookup — it can be
  stale or simply absent. A production version would use Amazon's
  Product Advertising API or eBay's Browse API (both require developer
  keys), which this project's "no paid APIs" constraint rules out.
- `retrieve_images` only returns results if the corpus was ingested with
  `pipeline.py --multimodal` — otherwise it returns `[]`, the same
  "empty is a valid answer, not an error" contract `retrieve()` already
  documents below.
- The domain allowlist (`local_rag/safety/domain_allowlist.py`) is a
  small, hand-curated set of reputable domains, not a dynamic reputation
  classifier — it guarantees a link's *domain* is on a short pre-approved
  list, which is a materially weaker claim than "this specific page is
  accurate or fairly priced." See that module's own docstring for the
  full reasoning and the honest tradeoff this represents.

Every function in `web_tools.py`/`image_tools.py`/`invoice_tools.py`
degrades to an empty result (never raises) if the network, `ddgs`, or the
CLIP/VLM stack is unavailable — verified in
`mcp_server/test_new_tools_smoke.py`, which fakes the network layer and
runs everywhere, no live internet or corpus needed.

```bash
python mcp_server/test_new_tools_smoke.py
```

## Original two tools + resource (Phase 1)

Wraps the Production RAG System's retrieval pipeline as an MCP server:
two tools (`retrieve`, `generate_answer`) and one resource
(`corpus://documents`).

Wired directly against your real source — `config.py`,
`embeddings/hf_embedder.py`, `vectorstore/chroma_store.py`,
`retrieval/hybrid_retriever.py`, `retrieval/reranker.py`,
`generation/ollama_generator.py` — and verified end-to-end with a
smoke test (fake chromadb/sentence_transformers/ollama/rank_bm25, real
everything else): seeded a fake corpus, called `retrieve()` →
confirmed correct shape and scores, called `generate_answer()` on the
result → got a grounded answer, called `list_documents()` → counts
matched the seed data, and confirmed the empty-corpus path returns `[]`
instead of crashing. No guessed signatures remain.

## Install

```bash
pip install -r requirements.txt          # this file — fastmcp, langchain-mcp-adapters
pip install -r ../requirements.txt       # the project's own deps, if not already installed
```

## Where this file goes

Place the whole `mcp_server/` folder at the project root, as a sibling
of `config.py`, `retrieval/`, `embeddings/`, etc. — not nested inside
any of those.

`server.py` inserts the project root onto `sys.path` itself, based on
its own file location, so `from config import ...` etc. resolve no
matter what working directory the MCP client happens to launch it
from. That's also why every client config below invokes it by
**absolute path** rather than relying on the client to set the right
`cwd` — Claude Code, Cursor, OpenCode, and langchain-mcp-adapters don't
all handle that the same way, so sidestepping it is more reliable than
trusting any one of them to get it right.

## Prerequisites

- A corpus already ingested into the Chroma collection defined by
  `config.CHROMA_COLLECTION` (`"rag_chunks"`), via `pipeline.py` or
  `stages.py`. If nothing's ingested yet, `retrieve()` returns `[]`
  rather than crashing — but `generate_answer()` will just generate
  an ungrounded answer if you pass it an empty chunk list, so ingest
  first.
- `ollama serve` running locally, with `ollama pull llama3.2` done —
  `generate_answer()` calls this directly.

## Known limitation: the BM25 corpus snapshot

`HybridRetriever` needs the full corpus client-side to build its BM25
index, so `server.py` snapshots it once at startup via
`store.get_all()` rather than re-fetching on every `retrieve()` call
(rebuilding the BM25 index from scratch per call would add real
latency). This means: **if you re-ingest documents while this server
is running, restart it** to pick up the change. `list_documents()`
always reflects the live store, so it can briefly disagree with what
`retrieve()` can actually find — worth noting as a real, documented
limitation in your report rather than something to quietly paper over.

## Run the server standalone (sanity check)

```bash
python mcp_server/server.py
```

It'll sit waiting on stdin/stdout — expected, not a hang. Ctrl+C to stop.

## Running over HTTP instead of stdio

Set `MCP_TRANSPORT=http` (optionally `MCP_SERVER_HOST`/`MCP_SERVER_PORT`,
default `0.0.0.0:8765`) to serve over a real network port instead:

```bash
MCP_TRANSPORT=http python mcp_server/server.py
# now listening at http://0.0.0.0:8765/mcp
```

This is what lets this server run as its own Docker container, reachable
by `agents/api.py` over the network rather than spawned as a stdio
subprocess — see `../docs/DOCKER.md`. `agents/mcp_client.py`'s
`build_client()` reads the matching `MCP_TRANSPORT`/`MCP_SERVER_URL` env
vars so both sides agree on transport. Leaving `MCP_TRANSPORT` unset
keeps the original stdio behavior above — nothing about Consumers 1 or 2
below changes unless you set it.

## Consumer 1 — Claude Code / Cursor / OpenCode

Copy `mcp_config.example.json` into whichever config file your client
expects (Claude Code: `.mcp.json` in the project root, or `claude mcp
add`; Cursor: `.cursor/mcp.json`; OpenCode: check its docs — the JSON
shape is the same across all three), and fix the absolute path.

Restart the client, then ask it something that should trigger a tool
call, e.g.:

> Use the local-rag tool to search for glazing techniques.

Screenshot the tool call and result — this is one of the two required
screenshots for Part 1.

## Consumer 2 — LangGraph via langchain-mcp-adapters

Fix the absolute path in `test_langgraph_client.py`, then:

```bash
python test_langgraph_client.py
```

This connects the same server via `MultiServerMCPClient`, lists the
discovered tools, and calls `retrieve()` once. Screenshot the output —
the second required screenshot, proving the same server serves two
independent clients.

## What's real vs. what's a design choice worth defending in your report

- `retrieve()` over-fetches `k * 3` candidates before reranking down to
  `k`, rather than mirroring `pipeline.py`'s `cmd_ask` (which reranks
  the same `top_k` it retrieved — a reorder, not a narrowing). This is
  a deliberate change, not an inconsistency: reranking only adds value
  if it has more candidates than it returns to choose from. Worth
  confirming against your labeled eval set if you want real numbers
  behind it rather than just the argument above.
- `retrieve` and `generate_answer` are kept as two separate tools
  rather than one fused `answer_question` tool. This is what lets a
  multi-hop specialist (Phase 2) call `retrieve` several times and
  `generate_answer` once at the end with the combined chunks — fusing
  them would make that decomposition impossible.

## Before moving to Phase 2

- Run `python mcp_server/server.py` against your real, already-ingested
  corpus (not the smoke test's fake data) and confirm
  `metadata.filename` actually shows up in `retrieve()` results.
- Confirm `list_documents()`'s counts match your last real ingestion run.
