# InMind Framing & Shipping Quote Agent (System B)

An independent agent service, built with **Google ADK + FastAPI** -- a
different stack from System A's LangGraph pipeline, in its own
container, talked to only over HTTP. This is the "Agent System B"
half of the two-agent-systems assignment: given an artwork's
dimensions, medium, and shipping destination, it returns a framing,
glazing, and shipping cost estimate.

## Why this exists

System A (`agents/`) already has `product_search` (finds art supplies)
and `invoice` (totals a purchase), but nothing that handles getting a
*finished piece* framed and shipped -- a distinct enough domain that a
real framing/shipping company could plausibly own it independently,
the same way a travel agent calls out to a separately-run hotel
booking agent. System A never imports anything from this package;
`mcp_server/framing_tools.py` on System A's side calls this service's
`POST /quote` route over plain HTTP, inside the docker-compose
network.

## Architecture inside this service

- **`pricing.py`** -- all the arithmetic, zero LLM calls. Frame cost is
  USD-per-cm-of-perimeter, glazing is USD-per-cm² of area (defaulted on
  for paper-based media, off for canvas-based media), shipping is a
  flat base rate + per-kg rate by destination zone (domestic /
  regional / international). Every number is illustrative coursework
  pricing, clearly labeled as such in the response's own `disclaimer`
  field -- not a real rate card.
- **`agent.py`** -- the actual Google ADK agent. Wraps `pricing.compute_quote`
  as a tool; the LLM's only job is to call that tool for the real
  numbers and write one short paragraph explaining the quote. It never
  invents a price -- the same "LLM writes the narrative, code owns the
  numbers" split System A's own `product_search` specialist already
  uses. Inference goes through **Groq first, local Ollama second** --
  matching this whole project's own "no paid APIs, Groq is the one
  hosted exception, always with a free local fallback" convention (see
  `local_rag/groq_client.py`'s own module docstring). Google ADK is
  still the actual agent *framework* here; the model behind it is
  routed through `google.adk.models.lite_llm.LiteLlm`, not Gemini.
- **`server.py`** -- the network boundary. `POST /quote` always returns
  the full deterministic quote; the natural-language `explanation` tries
  three tiers in order and reports exactly which one answered via
  `explanation_source`: `"groq"` (first choice, needs `GROQ_API_KEY`),
  `"ollama"` (free, local, tried automatically if Groq isn't configured
  or fails), or `"template"` (deterministic, if neither LLM tier
  worked) -- see that file's own module docstring. `GET /health` for
  liveness. `GET /.well-known/agent.json` is a minimal, hand-written
  A2A-style agent card -- see that route's own docstring for exactly
  how much of the real A2A protocol this does and doesn't implement.

## Running it standalone (no Docker)

```bash
cd framing_agent
pip install -r requirements.txt
cp .env.example .env   # optional -- fill in GROQ_API_KEY for the fastest explanations
uvicorn server:app --host 0.0.0.0 --port 8090
```

Works with **zero configuration** too -- no `.env` at all still gives
you full pricing data on every `/quote` call; it just tries your local
`ollama serve` for the explanation (falling back to a template if that
isn't running either).

Then, from anywhere:

```bash
curl -X POST http://localhost:8090/quote \
  -H "Content-Type: application/json" \
  -d '{"width_cm": 40.6, "height_cm": 50.8, "medium": "oil on canvas", "destination_country": "France"}'
```

## Running it via Docker Compose (with the rest of the project)

`docker compose up --build` from the project root brings this up as
the `framing-agent` service alongside `chroma-server`, `mcp-server`,
`backend`, and `frontend` -- see the project root's own
`docker-compose.yml` and `docker/framing_agent.Dockerfile`.

## Contract with System A

`mcp_server/framing_tools.py` is the only System-A file that knows this
service exists. It POSTs the same JSON body shown in the curl example
above to `FRAMING_AGENT_URL` (default `http://localhost:8090`, set to
`http://framing-agent:8090` inside docker-compose), with a timeout, and
degrades to a structured "framing service unavailable" result on any
failure -- so a stopped or unreachable System B never crashes a System
A conversation turn, it just means that one specialist's answer says
the framing service is unavailable right now.
