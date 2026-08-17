# Running InMind in Docker

Two ways to run the full stack in containers, same images either way:

- **Method 1 — manual**: `docker network create` + `docker build` +
  `docker run` for each service, step by step, no compose file.
- **Method 2 — docker compose**: `docker-compose.yml` at the project
  root does the same thing in one command.

Both wire up the same four System-A containers on the same kind of
custom bridge network:

```
chroma-server (8000) <---- mcp-server (8765) <---- backend (8001) <---- frontend (8080)
      ^__________________________________________________|
```

- **chroma-server** — owns the one Chroma SQLite connection. Both
  mcp-server and backend reach it over HTTP instead of each opening
  their own `PersistentClient` against a shared volume.
- **mcp-server** — `mcp_server/server.py`, now listening on a real
  network port (`MCP_TRANSPORT=http`) instead of only being spawnable as
  a stdio subprocess.
- **backend** — `agents/api.py`, talks to mcp-server over HTTP and to
  chroma-server over HTTP for its own direct `personal_rag` calls.
- **frontend** — the built React/assistant-ui app, served statically.
  Talks to the backend from your **browser**, via the backend's
  published host port — not over the docker network at all.

There's also a **fifth, genuinely independent container**:
**framing-agent** (System B — see `framing_agent/README.md`), a
separate Google ADK + FastAPI service mcp-server reaches over plain
HTTP (`mcp_server/framing_tools.py`), never a Python import, and never
given a `depends_on` relationship to any System-A container — the two
systems don't share a startup order any more than they share code. It
has its own build/run steps below (Method 1) and its own service block
in `docker-compose.yml` (Method 2), separate from the four above on
purpose.

**Ollama is not containerized.** Every container reaches out to Ollama
running on your host machine (`ollama serve`), same as your non-Docker
workflow — nothing here changes how Ollama itself runs.

**Heads up on build time/image size:** both `backend` and `mcp-server`
install the full `local_rag/requirements.txt` — the same file your
non-Docker workflow uses, including `torch`, `vllm`, and `bitsandbytes`
(for the GPU serving/quantization benchmark scripts, not anything the
running API path calls). Expect multi-GB images and a slow first build.
I didn't trim this file for Docker specifically since that's a separate
change from "dockerize this" — say the word if you want a leaner
runtime-only requirements file for the images.

---

## Model weights: what's downloaded, what's persisted

Two completely different kinds of "local model" in this project, and
Docker treats them completely differently:

**Ollama models** (`llama3.2`, `phi3`, `mistral`, `nomic-embed-text`,
`mxbai-embed-large`, any VLM like `llava`) are **not touched by Docker
at all**. `ollama serve` and everything it's already pulled stay exactly
where they are on your host machine; every container just makes HTTP
calls to `OLLAMA_HOST` (`http://host.docker.internal:11434`) the same
way your non-Docker workflow already does. Nothing here re-downloads
them, ever — they're not inside any image or volume this project
controls.

**Hugging Face models** — specifically `sentence-transformers/all-
MiniLM-L6-v2` (the embedder, `local_rag/embeddings/hf_embedder.py`) and
`cross-encoder/ms-marco-MiniLM-L-6-v2` (the reranker,
`local_rag/retrieval/reranker.py`, used by mcp-server's `retrieve()`
only — `personal_rag.search_personal()` doesn't rerank) — are a
different story. Neither ships baked into the image; both download from
Hugging Face the first time each container actually uses them (a few
seconds to a minute or so, ~90MB combined, needs internet access).

Without persisting that download, it would repeat on every container
**recreation** — not just an image rebuild, but every plain `docker
compose down && up` or `docker rm` + `docker run`, since a fresh
container starts from the image's layers with nothing written at
runtime carried over. Both Dockerfiles fix the cache to
`/app/.cache/huggingface` (`HF_HOME`) specifically so `docker-
compose.yml` (and the manual method's `-v` flags) can mount a persistent
volume there — `hf-cache`, shared between `backend` and `mcp-server`
since they both need the same embedder, so the download happens once
total, not once per container, the first time either one handles a
request after a fresh volume.

If you're migrating an existing non-Docker setup and want to skip that
first download entirely, copy your host's `~/.cache/huggingface` (Linux/
Mac) or `%USERPROFILE%\.cache\huggingface` (Windows) into the
`hf-cache` volume before first start — same cache format, so this should
just work, though I haven't verified that copy step directly.

---

## Prerequisites

- Docker Desktop (Windows/Mac) or Docker Engine + Compose plugin
  (Linux).
- `ollama serve` running on your host machine, with the models you use
  (`llama3.2`, `phi3`, ...) already pulled.
- Run every command below from the **project root** (the directory
  containing `agents/`, `local_rag/`, `mcp_server/`, `frontend/`,
  `docker/`) unless noted otherwise.

---

## Method 1 — manual (network + build + run)

### 1. Create the network

```
docker network create inmind-net
```

### 2. Build all four images

```
docker build -f docker/chroma_server.Dockerfile -t inmind-chroma-server:latest .
docker build -f docker/mcp_server.Dockerfile -t inmind-mcp-server:latest .
docker build -f docker/backend.Dockerfile -t inmind-backend:latest .
docker build -f docker/frontend.Dockerfile --build-arg VITE_API_BASE_URL=http://localhost:8001 -t inmind-frontend:latest .
```

### 3. Run chroma-server first (the others depend on it)

```
docker run -d --restart unless-stopped --name chroma-server --network inmind-net -p 8000:8000 -v inmind-chroma-data:/chroma/data inmind-chroma-server:latest
```

`--restart unless-stopped` on every container from here on isn't just
convenience — it also self-heals the startup race described in step 4
below, automatically retrying a container whose first startup attempt
lost that race.

Wait for it to come up (a few seconds), then confirm:

```
curl http://localhost:8000/api/v2/heartbeat
```

### 4. Run mcp-server -- then WAIT for it to report healthy

`host.docker.internal` resolves automatically on Docker Desktop
(Windows/Mac). On Linux, `--add-host` below adds the same mapping
manually — harmless to include on Desktop too.

```
docker run -d --restart unless-stopped --name mcp-server --network inmind-net -p 8765:8765 --add-host host.docker.internal:host-gateway -e MCP_TRANSPORT=http -e MCP_SERVER_HOST=0.0.0.0 -e MCP_SERVER_PORT=8765 -e CHROMA_CLIENT_MODE=http -e CHROMA_SERVER_HOST=chroma-server -e CHROMA_SERVER_PORT=8000 -e OLLAMA_HOST=http://host.docker.internal:11434 -e GROQ_API_KEY=your-key-here -e FRAMING_AGENT_URL=http://framing-agent:8090 -v inmind-backend-rag-data:/app/local_rag/data -v inmind-hf-cache:/app/.cache/huggingface inmind-mcp-server:latest
```

**Do not move on to step 5 yet.** This image imports the same heavy ML
stack (torch, sentence-transformers, ...) as backend, so it can easily
take 30-60+ seconds to actually start listening on 8765. Backend's own
startup calls `mcp-server` immediately to fetch its tool list — if
mcp-server isn't listening yet, that call fails and backend's whole
startup fails with it (`httpx.ConnectError: All connection attempts
failed`, thrown from `agents/mcp_client.py`'s `load_tools_by_name`).
`docker-compose.yml`'s `depends_on: condition: service_healthy` prevents
this automatically; the manual method has no equivalent, so it's on you
to wait here. Poll until it says `healthy`:

```
docker ps
```

Two volumes worth noting here:
- `inmind-backend-rag-data` — the **same** volume name the backend
  container mounts in step 5, at the same `/app/local_rag/data` path,
  deliberately. The `personal_docs` specialist's `latest_uploaded_image`
  tool runs inside this mcp-server container and reads a raw
  uploaded-image file straight off disk, but that file is written by the
  *backend* container's own upload endpoint. Two separate volumes here
  would mean this container never actually has the file the backend just
  wrote — the feature degrades to silently returning no image rather
  than an error, so it's an easy bug to miss. Docker creates the volume
  on first reference regardless of which container runs first, so the
  order here (mcp-server before backend) is fine.
- `inmind-hf-cache` — also shared with backend in step 5. Without this,
  the ~90MB Hugging Face embedder + reranker weights
  (`local_rag/embeddings/hf_embedder.py`, `local_rag/retrieval/
  reranker.py`) would re-download from scratch every time either
  container is recreated (every `docker rm` + `docker run`, not just an
  image rebuild) — see the next section for the full explanation.

### 5. Run backend

Only once step 4's `docker ps` shows mcp-server as `healthy`:

```
docker run -d --restart unless-stopped --name backend --network inmind-net -p 8001:8001 --add-host host.docker.internal:host-gateway -e AGENT_API_HOST=0.0.0.0 -e AGENT_API_PORT=8001 -e AGENT_API_CORS_ORIGINS=http://localhost:8080,http://127.0.0.1:8080 -e AGENT_API_DB_PATH=/app/data/chat_history.sqlite3 -e MCP_TRANSPORT=http -e MCP_SERVER_URL=http://mcp-server:8765 -e CHROMA_CLIENT_MODE=http -e CHROMA_SERVER_HOST=chroma-server -e CHROMA_SERVER_PORT=8000 -e OLLAMA_HOST=http://host.docker.internal:11434 -e GROQ_API_KEY=your-key-here -v inmind-backend-data:/app/data -v inmind-backend-rag-data:/app/local_rag/data -v inmind-hf-cache:/app/.cache/huggingface inmind-backend:latest
```

Note `MCP_SERVER_URL=http://mcp-server:8765` — `mcp-server` here is the
**container name from step 4**, resolved by Docker's embedded DNS
because both containers are on `inmind-net`. This is the actual
network-port communication the backend and mcp-server use to talk to
each other now, instead of the backend spawning mcp-server as a stdio
subprocess.

### 6. Run frontend

```
docker run -d --restart unless-stopped --name frontend --network inmind-net -p 8080:8080 inmind-frontend:latest
```

### 6b. Build and run framing-agent (System B — optional, independent)

Not part of the four-container chain above, and not built in step 2 —
called out separately because it genuinely doesn't depend on (or get
depended on by) anything else here. Skip this entirely and every other
service keeps working exactly as before; `get_framing_quote` (the one
System-A tool that talks to it) just reports the framing service as
unreachable, the same as if you'd started it and then stopped it.

```
docker build -f docker/framing_agent.Dockerfile -t inmind-framing-agent:latest .
docker run -d --restart unless-stopped --name framing-agent --network inmind-net -p 8090:8090 --add-host host.docker.internal:host-gateway -e GROQ_API_KEY=your-key-here -e OLLAMA_HOST=http://host.docker.internal:11434 inmind-framing-agent:latest
```

`-e GROQ_API_KEY=...` is the same key/variable as mcp-server's and
backend's own `-e GROQ_API_KEY=...` above — optional either way; omit
it (or leave it empty) and this container tries `-e OLLAMA_HOST=...`
next automatically (same host-reaching setup as every other container
here, hence the `--add-host` flag), falling back further to a templated
explanation if that isn't reachable either. `/quote` always returns
full pricing data regardless of which of the three actually answered —
see `framing_agent/server.py`'s own module docstring for the exact
Groq → Ollama → template order. If mcp-server is
already running, no restart is needed for it to start reaching
framing-agent — `mcp_server/framing_tools.py` reads `FRAMING_AGENT_URL`
fresh on every call, and step 4's own `docker run` for mcp-server
already includes `-e FRAMING_AGENT_URL=http://framing-agent:8090`.

### 7. Verify

```
curl http://localhost:8001/health
curl http://localhost:8090/health   # only if you ran step 6b
```

Then open `http://localhost:8080` in your browser.

### Tearing down (manual method)

```
docker rm -f frontend backend mcp-server chroma-server framing-agent
docker network rm inmind-net
```

Add `-v` isn't needed on `rm -f` — the named volumes
(`inmind-chroma-data`, `inmind-backend-data`,
`inmind-backend-rag-data`, `inmind-hf-cache`) persist independently.
Remove them explicitly if you actually want a clean slate:

```
docker volume rm inmind-chroma-data inmind-backend-data inmind-backend-rag-data inmind-hf-cache
```

`framing-agent` has no named volume of its own (see
`docker-compose.yml`'s own note on this — it's fully stateless), so
there's nothing to remove for it beyond the container itself.

---

## Method 2 — docker compose

```
cp .env.docker.example .env
```

Edit `.env` if `OLLAMA_HOST` needs to point somewhere other than
`http://host.docker.internal:11434`, or if any of the default ports
(8000/8001/8080/8765) collide with something already running on your
machine.

```
docker compose up --build
```

Same four System-A services, same network shape, same volumes — just
declared in `docker-compose.yml` instead of typed out by hand — plus
the fifth, `framing-agent` (System B), also declared there but with no
`depends_on` linking it to the other four, matching Method 1's step 6b
above. `depends_on` with `condition: service_healthy` means compose
waits for chroma-server (and mcp-server, for backend) to pass its
healthcheck before starting the next service, so you don't hit a race
on first startup — framing-agent starts independently, in parallel,
since nothing else here is allowed to wait on it or vice versa.

**If you already built the images with Method 1**, each service's
`image:` here (`inmind-backend:latest`, etc.) is the exact same tag
Method 1's `docker build` commands use — see "Does building manually
first make compose faster?" below.

Tear down:

```
docker compose down          # keeps volumes (chat history, corpus, uploads)
docker compose down -v       # also deletes volumes -- clean slate
```

---

## Does building manually first make `docker compose` faster?

Yes, for two different reasons depending on how you run the `compose`
step afterward.

**`docker compose up --build`** re-runs the build, but Docker's build
cache lives in the daemon, keyed by each Dockerfile instruction's
content — not by which command (`docker build` vs `docker compose
build`) triggered it. Method 1's `docker build` and Method 2's compose
build point at the exact same Dockerfiles and the exact same build
context (the project root), so as long as nothing in that context
changed between the two, compose's build hits cache on every layer and
mostly just re-tags the result — seconds, not the original 20-40 minutes.
This is also why both Dockerfiles copy `requirements.txt` and run `pip
install` *before* copying the rest of the source: a source-only change
(editing `agents/specialists.py`, say) doesn't invalidate that cached
`pip install` layer at all, in either method.

**Plain `docker compose up` (no `--build`)** skips building entirely if
an image already exists under the tag `docker-compose.yml` expects.
That's exactly what the `image:` line under each service is for here —
it's set to match Method 1's tags on purpose. So: build manually with
Method 1's exact commands once, then `docker compose up` (dropping
`--build`) reuses those images directly, with **no build step at all**,
not even a cache-hit one.

Two things that break this:
- **Any change to the build context** (edited requirements.txt, edited
  source files you then rebuild for) invalidates the affected layer and
  everything after it in that Dockerfile, same as a normal Docker cache
  bust — expected, not a bug.
- **`docker builder prune` / Docker Desktop's "clean up"** clears the
  build cache (not the same as removing images) — the tagged images
  from Method 1 would still exist and `docker compose up` without
  `--build` would still skip building, but a future `--build` run would
  be back to a cold cache.

---



## Environment variables reference

| Variable | Where it's read | Default | Purpose |
|---|---|---|---|
| `OLLAMA_HOST` | `local_rag/config.py`, `framing_agent/agent.py` | `http://localhost:11434` (unset) | Where every container (System A AND System B) reaches your host's Ollama server |
| `CHROMA_CLIENT_MODE` | `local_rag/config.py` → `vectorstore/chroma_store.py` | `embedded` | `embedded` = local `PersistentClient` (original, non-Docker default); `http` = talk to a separate Chroma server |
| `CHROMA_SERVER_HOST` / `CHROMA_SERVER_PORT` | same | `localhost` / `8000` | Only read when `CHROMA_CLIENT_MODE=http` |
| `MCP_TRANSPORT` | `agents/mcp_client.py`, `mcp_server/server.py` | `stdio` | `stdio` = original local-subprocess behavior; `http` = real network port on both sides |
| `MCP_SERVER_URL` | `agents/mcp_client.py` | `http://127.0.0.1:8765` | Only read when `MCP_TRANSPORT=http`; the backend's client points here |
| `MCP_SERVER_HOST` / `MCP_SERVER_PORT` | `mcp_server/server.py` | `0.0.0.0` / `8765` | Only read when `MCP_TRANSPORT=http`; what the server itself binds to |
| `AGENT_API_HOST` / `AGENT_API_PORT` | `agents/api.py` | `127.0.0.1` / `8001` | Already existed before Docker; `0.0.0.0` is required inside a container |
| `AGENT_API_CORS_ORIGINS` | `agents/api.py` | `http://localhost:5173,...` | Must include wherever the frontend is actually reachable in the browser |
| `AGENT_API_DB_PATH` | `agents/api.py` | next to `agents/api.py` | Pointed at a mounted volume in Docker so chat history survives a restart |
| `GROQ_API_KEY` | `local_rag/groq_client.py`, `framing_agent/server.py` | unset | Same key, all three of mcp-server/backend/framing-agent. Was previously **not wired into any container at all** even when set in your real `.env` (see mcp-server's own `environment:` block comment in `docker-compose.yml` for why) — fixed. Unset, everything falls back to local Ollama automatically |
| `GEMINI_API_KEY` | `local_rag/config.py` | unset | Optional, for personal-upload single-image captioning's online VLM backend. A *different* Google product/key from `GROQ_API_KEY` — unrelated to framing-agent, which now uses Groq/Ollama, not Gemini |
| `FRAMING_AGENT_URL` | `mcp_server/framing_tools.py` | `http://localhost:8090` | System B's own base URL; docker-compose.yml sets this to `http://framing-agent:8090` for mcp-server, the only System-A file that reads it |
| `FRAMING_AGENT_GROQ_MODEL` | `framing_agent/agent.py` | `llama-3.3-70b-versatile` | Only read if `GROQ_API_KEY` is set — System B's own Groq model, same default as `local_rag/config.py`'s `GROQ_LARGE_MODEL` |
| `FRAMING_AGENT_OLLAMA_MODEL` | `framing_agent/agent.py` | `llama3.2` | System B's own local-Ollama model, tried automatically if Groq isn't configured or fails |
| `FRAMING_AGENT_PORT` | docker-compose.yml only | `8090` | Published host port for the `framing-agent` container |

Every one of these keeps its exact original default — none of this
changes non-Docker behavior (`python -3.12 -m agents.<module>` from the
project root) unless the env var is explicitly set.

---

## Troubleshooting

- **`host.docker.internal` doesn't resolve (Linux)** — confirm your
  Docker Engine version supports `host-gateway` (20.10+). If it still
  doesn't work, replace `OLLAMA_HOST`'s host with your machine's actual
  bridge IP (`ip addr show docker0`, usually `172.17.0.1`).
- **A container is `unhealthy` and dependents won't start** — check that
  container's logs (`docker logs <name>` / `docker compose logs
  <service>`) before anything else; `depends_on: condition:
  service_healthy` is deliberately strict about this so a half-started
  chroma-server can't cause a confusing failure two containers away.
- **Port already in use** — change the left-hand side of the relevant
  `-p`/`ports:` mapping (manual method) or the matching `*_PORT` variable
  in `.env` (compose method); the containers' own internal ports never
  need to change.
- **First `docker build` is very slow / image is huge** — expected, see
  the note above about `local_rag/requirements.txt`'s full dependency
  set (`torch`, `vllm`, `bitsandbytes`).
- **Manual method: `backend` exits right after starting, logs show
  `httpx.ConnectError: All connection attempts failed` from
  `agents/mcp_client.py`'s `load_tools_by_name`** — a real startup race,
  confirmed in practice, not a config error: backend's own startup calls
  mcp-server immediately to fetch its tool list, and mcp-server's image
  carries the same heavy ML stack backend does, so it can easily still be
  mid-import when backend tries. Step 4's "wait for healthy before step
  5" instruction exists specifically to prevent this. `--restart
  unless-stopped` (in every command above) also self-heals it after the
  fact — Docker retries the crashed container, and by the second attempt
  mcp-server is usually ready — but waiting for `healthy` first avoids
  the crash-and-retry cycle entirely. `docker-compose.yml`'s `depends_on:
  condition: service_healthy` prevents this automatically, which is one
  real advantage of Method 2 over Method 1.
- **A container that was running is gone after closing/sleeping your
  machine** — expected on Windows/Mac: closing the machine stops Docker
  Desktop's VM, which stops every container in it. `docker ps -a` (not
  `docker ps`) shows them still existing but `Exited`. Every command
  above includes `--restart unless-stopped` specifically so this
  self-heals — Docker brings them back automatically once its daemon is
  running again — but if you started containers without it earlier,
  they need a manual `docker start <name>` (or `docker compose up`
  again) after a restart.
