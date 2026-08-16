# Chat API (`agents/api.py`)

A FastAPI wrapper around the agent graph (`agents/graph.py`) for talking
to it like a normal chatbot: multi-turn, one `thread_id` per conversation,
history that survives across requests and server restarts. No Open WebUI,
no other chat framework — a self-contained browser UI is served by the
same process at `/`.

## Why this exists on top of `agents.graph.ask()`

`ask()` builds a brand-new graph — new MCP client, new spawned
`mcp_server/server.py` subprocess, empty `messages` list — on **every**
call. Right for the CLI and the Phase 5 eval script, where each question
is independent. Wrong for a chatbot: every turn would pay several
seconds of subprocess startup, and every turn would start with amnesia
— including breaking the `invoice` specialist, which specifically reads
past `product_search` messages back out of `state["messages"]` and is
silently unusable across separate `ask()` calls no matter how you phrase
the request.

`agents/api.py` instead builds **one** graph, once, at process startup,
compiled with a real LangGraph checkpointer (`AsyncSqliteSaver`). Calling
that same compiled graph repeatedly with the same
`config={"configurable": {"thread_id": ...}}` accumulates `messages`
across calls via `state.py`'s `add_messages` reducer, instead of each
call starting over. As a side effect, this also fixes the "spawns a
fresh MCP subprocess per call" cost `agents/agent_mcp_server.py`'s own
docstring names as a known, un-engineered-around tradeoff — here it's
one subprocess for the server's whole lifetime.

`route`, `iteration_count`, `blocked`, and `injection_patterns` are
explicitly reset to their turn-zero values on every `/chat` call (see
`_new_turn_state` in `agents/api.py`) — those four have no reducer in
`state.py`, so without an explicit reset they'd otherwise silently carry
the previous turn's values forward (checked by hand against a minimal
synthetic `StateGraph` with the same shape before wiring this up for
real). A fresh turn needs its own iteration cap and its own guardrail
check, not an accumulating one across a whole conversation.

## Install

```bash
pip install -r agents/requirements.txt
pip install -r agents/requirements-api.txt
```

## Run

```bash
python -m agents.api
# or: uvicorn agents.api:app --reload --port 8000
```

Then open `http://localhost:8000/` in a browser and chat. Conversation
history is stored in `agents/chat_history.sqlite3` (path overridable via
`AGENT_API_DB_PATH`) and survives server restarts.

Environment variables (all optional):

| Variable | Default | Purpose |
|---|---|---|
| `AGENT_API_DB_PATH` | `agents/chat_history.sqlite3` | checkpoint SQLite file |
| `AGENT_API_ITERATION_CAP` | `8` (supervisor.py's `DEFAULT_ITERATION_CAP`) | per-turn supervisor loop cap |
| `AGENT_API_ROUTE_FORMAT` | `json_schema` | passed through to `build_supervisor` |
| `AGENT_API_HOST` / `AGENT_API_PORT` | `127.0.0.1` / `8000` | only used by `python -m agents.api` |
| `AGENT_API_RELOAD` | unset | set to any value to enable `uvicorn --reload` behavior |

## Endpoints

- `POST /chat` — `{"message": "...", "thread_id": "optional"}` → answer
  plus `answered_by` (specialist name, or `input_guard` if refused),
  `blocked`, `iteration_count`, and `turn_messages` (every message
  produced this turn, for seeing the routing path).
- `GET /chat/{thread_id}/history` — full stored conversation for that thread.
- `DELETE /chat/{thread_id}` — forget a thread.
- `GET /health` — readiness probe + current config.
- `GET /` — the browser chat UI (`agents/static/chat.html`).

## Test

```bash
python -m agents.test_api_smoke
```

Mirrors `agents/test_graph_smoke.py`'s faking strategy (scripted
supervisor LLM, fake specialists — no Ollama or MCP subprocess needed)
but drives the actual FastAPI app over HTTP via `TestClient`, and adds
the one thing that test file can't check: that `messages` really persists
across **separate HTTP requests** sharing a `thread_id`, and that
`DELETE` really clears it.

## Known limitations

- Persistence is a single local SQLite file — fine for one person's local
  use, not multi-process-safe, no auth in front of it (see the Part 2
  "Auth" extension in the project spec if you want to add a bearer token
  in front of this too).
- The checkpointer makes the *full* conversation visible in
  `state["messages"]`, and `invoice` specifically uses that to look back
  at prior `product_search` results — but most other specialists
  (`retrieval_qa`, `corpus_meta`, `multi_hop`, ...) still only read the
  *latest* `HumanMessage` as their question. Giving them real
  cross-turn context (resolving "that painting I asked about earlier")
  would mean changing how each specialist builds its question, which is
  outside this API layer.
- No streaming — `/chat` blocks until the supervisor loop reaches
  `FINISH` or the iteration cap fires a partial answer, same as `ask()`
  always has.
