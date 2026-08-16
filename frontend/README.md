# Frontend (assistant-ui)

A [assistant-ui](https://www.assistant-ui.com/) chat interface for
`agents/api.py`. Same backend as `agents/static/chat.html` (the plain
hand-rolled UI built earlier) — this is a second, framework-based option
sitting alongside it, not a replacement. Run whichever you want; both
talk to the same `/chat`, `/chat/{thread_id}/history`, and
`/chat/{thread_id}` (DELETE) endpoints.

## Why `useExternalStoreRuntime`, not `useLocalRuntime`

assistant-ui's default (`useLocalRuntime`) expects to own the message
list itself on the client, with your `ChatModelAdapter` only handling
inference. That's backwards for this project: `agents/api.py` already
owns conversation history server-side, persisted via LangGraph's own
checkpointer and keyed by `thread_id` — see that file's module docstring
for why (`invoice` specifically reads back prior `product_search` turns
from that persisted history, which only works if history is real and
server-owned).

`useExternalStoreRuntime` is built for exactly this shape: you hand it a
plain `messages` array you control, plus an `onNew` callback, and it
renders whatever you give it. `src/runtime.ts` is the whole bridge:
`messages` is React state seeded from `GET /chat/{thread_id}/history` on
load, appended to locally after each `POST /chat` response.

Getting `useLocalRuntime` to the same place would mean implementing a
full `ThreadHistoryAdapter` (assistant-ui's branching message-repository
format) just to load history from the backend on mount — real
work for a feature `ExternalStoreRuntime` gives for free by construction.

## Layout

```
src/
  api.ts              fetch wrapper for agents/api.py's endpoints
  runtime.ts           the ExternalStoreRuntime bridge (the important file)
  types.ts             local ChatMessage shape
  App.tsx               header, "New conversation" button, error banner
  components/
    Thread.tsx          ThreadPrimitive.Root/Viewport/Messages/ScrollToBottom
    Message.tsx          UserMessage / AssistantMessage (MessagePrimitive)
    Composer.tsx          input box + send (ComposerPrimitive)
  styles.css             plain CSS, no Tailwind/shadcn -- see note below
```

Built from assistant-ui's headless primitives (`ThreadPrimitive`,
`MessagePrimitive`, `ComposerPrimitive`) directly, styled with plain CSS
targeting the `aui-*` class names in `styles.css`, rather than via
`npx assistant-ui init` / `npx shadcn add`. Those CLIs pull a polished
pre-built `Thread` component from assistant-ui's own component registry
over the network — a fine option if you have access to it, just not
something to route through here. The primitives underneath are the same
ones that generates; this skips the CLI step and hand-writes the
composition, which is also easier to restyle without fighting Tailwind
config.

The specialist name shown above each assistant bubble (`retrieval_qa`,
`corpus_meta`, `input_guard` when refused, etc.) rides along as
`metadata.custom.name` on each message (set in `runtime.ts`'s
`toThreadMessageLike`) and is read back out via `useAuiState` in
`Message.tsx` — the one place this app reaches past the primitives'
default rendering into assistant-ui's lower-level state hook.

## Run

Backend first, in the project root:

```bash
pip install -r agents/requirements.txt -r agents/requirements-api.txt
python -m agents.api
```

Then, in `frontend/`:

```bash
npm install
npm run dev
```

Open `http://localhost:5173`. `agents/api.py`'s `CORSMiddleware` already
allows that origin by default (`AGENT_API_CORS_ORIGINS` to change it).

If the backend runs somewhere other than `http://localhost:8000`, copy
`.env.example` to `.env.local` and set `VITE_API_BASE_URL`.

## Verify

```bash
npm run typecheck
```

This was run against the actual installed `@assistant-ui/react` types
before being handed over (not written from memory against possibly-stale
docs) — zero errors. It doesn't call the backend, so this is a static
check only; `npm run dev` plus a real conversation is the actual test.

## Build

```bash
npm run build
```

Outputs static files to `dist/` — `agents/api.py` doesn't serve this
directory itself (it only serves `agents/static/chat.html`). Serve
`dist/` with any static file host, or point at
`http://localhost:8000` from wherever you host it (adjust
`AGENT_API_CORS_ORIGINS` to match).

## Known limitations

- No streaming, same as `agents/api.py` itself: `/chat` blocks until the
  supervisor loop finishes, so `onNew` resolves all at once rather than
  token-by-token. `ThreadPrimitive`'s streaming affordances are present
  in the primitives but have nothing to stream here.
- Editing/regenerating a past message isn't wired up (`ExternalStoreAdapter`'s
  optional `onEdit`/`onReload`/`setMessages` are omitted) — sending is
  the only supported action, matching what the plain HTML UI already did.
- Single active thread per browser tab (`thread_id` in `localStorage`),
  no assistant-ui `ThreadList` sidebar of past conversations — `agents/api.py`
  has no "list all threads" endpoint to back one with yet.
