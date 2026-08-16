# Multi-Agent RAG Pipeline for Art and Painting Treatises

A fully local, multi-agent question-answering system built on top of a Retrieval-Augmented
Generation (RAG) pipeline over historical painting and drawing treatises. No paid APIs,
no cloud calls, everything runs on your own machine.

This README summarizes how to understand, run, and extend the project. For the full
narrative, including every design decision, the benchmarks behind them, and a complete
history of problems encountered and how they were fixed, see `Project_Status_Report.docx`.

---

## 1. What This Is

The project has two layers stacked on top of each other, plus a chat interface on top of both.

- **`local_rag/`** — the RAG engine. Ingests a corpus of historical art and painting
  treatises, chunks and embeds the text, stores it in a vector database, and answers
  questions by retrieving relevant passages and generating an answer with a local model.
- **`mcp_server/`** — wraps `local_rag/` as a set of named tools using the Model Context
  Protocol (MCP), so anything that speaks MCP (this project's own agents, or an external
  tool like Claude Code or Cursor) can use the pipeline without importing its code directly.
- **`agents/`** — a multi-agent system built with LangGraph. A supervisor reads each
  question and routes it to one of seven specialists, with input and output guardrails
  and a persistent chat API on top.
- **`frontend/`** — a browser chat interface (a React app, plus a simpler single-file
  HTML page) for actually talking to the system.

Everything runs locally through Ollama for the language model and ChromaDB for vector
storage. There are no paid API keys anywhere in the stack.

---

## 2. Architecture

```
local_rag/     RAG engine: ingest, chunk, embed, store, retrieve, generate.
                    |  (imported directly, as normal Python code)
                    v
mcp_server/    Exposes the engine as MCP tools: retrieve, generate_answer,
               retrieve_images, search_painting_online, search_art_supplies,
               generate_invoice.
                    |  (reached only through MCP, never imported directly)
                    v
agents/        Supervisor + seven specialists + guardrails, wired together
               with LangGraph, exposed as a REST chat API.
                    |  (reached over plain HTTP)
                    v
frontend/      The browser chat interface.
```

A question moves through the agent graph as:

```
START -> input_guard -> contextualize -> supervisor -> specialist -> supervisor
                                                            (loops until FINISH)
                                                                  |
                                                                  v
                                                            output_guard -> answer
```

`input_guard` blocks unsafe or oversized input before anything else runs.
`contextualize` rewrites a vague follow-up question into a standalone one using recent
history. `supervisor` picks the next specialist, or decides the turn is done, bounded by
a fixed iteration cap. `output_guard` scans every message produced since the user's last
turn for personal information and unapproved link domains before the answer goes out.

---

## 3. The Seven Specialists

| Specialist | Tools | Model calls | Purpose |
|---|---|---|---|
| `retrieval_qa` | `retrieve`, `generate_answer` (loop) | Yes, iterative | Answers a single-topic question from the corpus |
| `corpus_meta` | None (static prompt) | Yes, one call | Answers questions about the corpus itself (titles, counts) |
| `multi_hop` | `retrieve` x2 (fixed script) | Yes, one call | Splits a compound question into two parts and combines the answers |
| `image_qa` | `retrieve_images` | No | Shows corpus images with captions |
| `painting_lookup` | `retrieve`, `search_painting_online` | Yes, one call | Answers about one named painting using the corpus and the web |
| `product_search` | `search_art_supplies` | Yes, one call | Finds art supplies online, split into beginner/professional tiers |
| `invoice` | `generate_invoice` | No | Totals prior `product_search` results in plain Python arithmetic |

---

## 4. Project Layout

```
project-root/
|-- local_rag/
|     |-- ingestion/, chunking/, embeddings/, vectorstore/,
|     |-- retrieval/, generation/, safety/, evaluation/
|     |-- slm/, quantization/, vlm/, serving/   (benchmarking only, not wired in)
|     |-- config.py, pipeline.py, stages.py, api.py
|
|-- mcp_server/
|     |-- server.py, image_tools.py, web_tools.py, invoice_tools.py
|     |-- test_new_tools_smoke.py, test_langgraph_client.py
|
|-- agents/
|     |-- state.py, prompts.py, mcp_client.py
|     |-- specialists.py, supervisor.py, contextualize.py
|     |-- guardrails.py, graph.py, api.py
|     |-- eval_phase5.py, agent_mcp_server.py
|     |-- static/chat.html
|     |-- test_*.py   (nine files)
|
|-- frontend/
|     |-- src/api.ts, runtime.ts, types.ts
|     |-- src/components/Thread.tsx, Message.tsx, Composer.tsx,
|                        ChatHistory.tsx, ToolSelector.tsx, MarkdownText.tsx
|
|-- docker/
|     |-- backend.Dockerfile, mcp_server.Dockerfile,
|     |-- frontend.Dockerfile, chroma_server.Dockerfile
|-- docker-compose.yml, .env.docker.example
|-- docs/DOCKER.md
```

`config.py` is the single source of truth for every path and model name used anywhere
in the project; nothing else hardcodes a path.

---

## 5. Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com), running locally
- Node.js 18+ and npm (for the React frontend only)

### Pull the required local models

```bash
ollama pull llama3.2
ollama pull all-minilm      # or the equivalent embedding model configured in config.py
```

### Install Python dependencies

```bash
pip install -r requirements.txt
pip install -r agents/requirements-api.txt
```

### Ingest the corpus

Place the source treatises (PDF/text/image files) in the folder referenced by
`local_rag/config.py`, then run:

```bash
python -m local_rag.pipeline ingest
```

Re-running this command is safe: incremental ingestion only processes new or changed
files, tracked by a content-hashed manifest.

---

## 6. Running the System

### Start the MCP server (optional, standalone)

```bash
python -m mcp_server.server
```

This is normally spawned automatically by `agents/mcp_client.py`, so you do not need to
run it by hand unless you are connecting an external MCP client (Claude Code, Cursor,
OpenCode) directly to it.

### Start the chat backend

```bash
uvicorn agents.api:app --reload
```

This builds the LangGraph agent graph once at startup and keeps it alive for the whole
process, backed by a SQLite checkpointer so conversation history survives across requests
and restarts.

### Try it without a frontend

Open `agents/static/chat.html` in a browser (served automatically by the backend above),
or send a request directly:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What did Cennini say about tempera grounds?"}'
```

### Run the React frontend

```bash
cd frontend
npm install
npm run dev
```

### Ask a one-off question from the command line

```bash
python -m local_rag.pipeline ask "What pigments were used for ultramarine?"
```

---

## 7. Running in Docker

The whole stack (chroma-server, mcp-server, backend, frontend) also runs as four
containers on a shared network, in two equivalent ways: a manual `docker network
create` + `docker build` + `docker run` walkthrough, or a `docker-compose.yml` that
does the same thing in one command. See **`docs/DOCKER.md`** for the full guide,
including the environment-variable reference and troubleshooting.

Two things changed to make this possible, both controlled by env vars so the
non-Docker workflow above is completely unaffected unless you set them:

- **`mcp_server/server.py` can serve over a real network port** instead of only
  being spawnable as a stdio subprocess (`MCP_TRANSPORT=http`), so it can run as
  its own container that `agents/api.py` reaches over the network
  (`agents/mcp_client.py`'s `build_client()`) rather than spawning as a child
  process.
- **`vectorstore/chroma_store.py` can talk to a separate Chroma server** over HTTP
  instead of only opening its own local `PersistentClient`
  (`CHROMA_CLIENT_MODE=http`) — this is what actually avoids the SQLite
  "database is locked" risk two containers sharing one Chroma volume can hit
  under real concurrent writes.

Ollama itself is not containerized — every container reaches out to `ollama serve`
running on your host machine, same as the non-Docker workflow.

---

## 8. Testing

Two tiers of tests exist throughout the project:

```bash
# Fast offline smoke tests (fake LLM and fake tool responses, no dependencies)
pytest agents/ mcp_server/ local_rag/ -k smoke

# Live tests against a real Ollama model and the real corpus
python agents/test_specialists_live.py
```

Two of the offline smoke tests are a bit different from the rest: rather than
mocking the network, `agents/test_mcp_client_transport_smoke.py` and
`local_rag/test_chroma_store_http_smoke.py` start a real (throwaway) server
subprocess and round-trip a real call through it, to prove the Docker-mode
transport switches above actually work over a socket, not just that they build
the right-looking config object.

Run the evaluation harness (ten designed questions covering routing, multi-step,
out-of-scope, and adversarial cases):

```bash
python agents/eval_phase5.py
```

This grades routing correctness mechanically and flags rows that need a human read for
answer quality.

---

## 9. Known Limitations

These are documented tradeoffs, not hidden gaps:

- `search_art_supplies` prices are read from search-result snippets, not a live pricing
  lookup, and can be stale or missing; a real product pricing API would fix this but
  requires a paid developer key.
- The domain allowlist for internet links is a small, hand-picked list, not a live
  reputation system.
- Output-guard PII redaction only catches structured patterns (emails, phone numbers, card
  numbers), not names or addresses written in plain prose.
- No streaming in the chat layer; answers arrive as a single block.
- No authentication in front of the chat API.

See Section 9 of `Project_Status_Report.docx` for the full list and context.

---

## 10. Further Reading

`Project_Status_Report.docx`, included alongside this README, is the authoritative
reference for this project. It covers the full system architecture, every folder and file
in detail, the benchmark data behind each production configuration choice, a complete
history of the real problems encountered while building the system and how each was fixed,
the current status of every component, and a stated list of limitations and future work.
