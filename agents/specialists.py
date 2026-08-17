"""
Phase 2 specialists: retrieval-QA, corpus-meta, multi-hop -- plus four
later additions (image-qa, painting-lookup, product-search, invoice)
added on top of the same architecture, each documented in its own
section of build_specialists() below. Every design rule this module's
original docstring states for the first three ("all pipeline access
goes through the Phase 1 MCP server", "structural guardrail over prompt
wording", "iteration count fixed and knowable in advance" where
applicable) applies identically to the four later ones -- see
mcp_server/web_tools.py, mcp_server/image_tools.py, and
mcp_server/invoice_tools.py for the tool-side implementations these four
specialists call through the same MCP boundary as the original three.

Each specialist is a LangGraph node: a function of AgentState that reads
the latest HumanMessage and returns a partial state update shaped like
{"messages": [AIMessage(...)]}, per state.py's schema. None of them touch
the retrieval pipeline directly -- all pipeline access goes through the
Phase 1 MCP server via agents/mcp_client.py, so Claude Code/Cursor/OpenCode
and this graph are provably hitting the same code path (see
mcp_client.py's module docstring for why that matters).

Three specialists, three deliberately different tool footprints:

  - retrieval_qa_node:  a create_react_agent bound to {retrieve,
        generate_answer}. Letting the LLM loop (rather than hardcoding
        "call retrieve exactly once") earns its keep here -- a vague or
        broad single-topic question may need a second retrieve() call
        before there's enough grounding, and RETRIEVAL_QA_SYSTEM_PROMPT
        already tells it never to call generate_answer on chunks it
        didn't just retrieve for this question.

  - corpus_meta_node:   plain LLM call, NO tools at all. The document
        list is fetched ONCE by build_specialists() and baked into this
        node's system prompt as static text. It structurally cannot
        answer a content question, because it has never been given any
        content -- see prompts.py's CORPUS_META_SYSTEM_PROMPT docstring
        for why that's a design choice, not an oversight.

  - multi_hop_node:     NOT a react agent. Its two retrieval calls and one
        generation call are explicit Python -- decompose once, retrieve
        twice, synthesize once, always exactly that shape, never however
        many times an LLM decides to loop. This is what makes it the
        clean "multi-step routing" eval case Phase 5 asks for: the
        iteration count for this node is always knowable in advance,
        which matters once the supervisor (Phase 3) is counting it
        against the graph-wide iteration cap.

build_specialists() is the one async entry point this module exposes. It
builds one MCP client, loads tools, fetches the corpus resource once, and
returns three ready-to-call node functions closing over that shared
state. Call it once per graph run -- see mcp_client.py's build_client()
docstring for why one client per run, not one per call, is the right
granularity (specialists sharing one server process means they see the
same BM25 snapshot and the same live Chroma store within a conversation).
"""

import json
import os
import re
import statistics
import sys
import base64
import mimetypes
from pathlib import Path
from typing import Awaitable, Callable, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from agents.llm_provider import get_chat_model
from agents.mcp_client import (
    build_client,
    fetch_corpus_documents,
    load_tools_by_name,
    unwrap_tool_result,
)
from agents.prompts import (
    CORPUS_META_SYSTEM_PROMPT,
    IMAGE_QA_NO_RESULTS_MESSAGE,
    PERSONAL_DOCS_NO_RESULTS_MESSAGE,
    MULTI_HOP_DECOMPOSE_SYSTEM_PROMPT,
    MULTI_HOP_SYNTHESIZE_PROMPT_TEMPLATE,
    PAINTING_LOOKUP_SYNTHESIZE_PROMPT_TEMPLATE,
    PRODUCT_SEARCH_SYSTEM_PROMPT,
    RETRIEVAL_QA_SYSTEM_PROMPT,
)
from agents.state import AgentState

def _find_pipeline_root() -> Path:
    """
    Locate the directory that actually contains config.py, the same way
    mcp_server/server.py's own _find_pipeline_root() does -- checked in
    order, relative to this file's own location (agents/):
      1. agents/../config.py              (config.py directly at project root)
      2. agents/../local_rag/config.py     (config.py nested under a local_rag/ folder)
      3. agents/../../config.py            (agents/ itself nested one level deeper)
      4. agents/../../local_rag/config.py

    Kept as its own copy rather than imported from server.py, since
    agents/ has no dependency on mcp_server/'s internals otherwise (only
    on the running server process, via mcp_client.py) and shouldn't gain
    one just to reuse eight lines of path-checking.

    If none of these match, the error message tells you exactly what was
    checked, so you can see at a glance which candidate to add for your
    actual layout.
    """
    here = Path(__file__).resolve().parent  # agents/
    parent = here.parent
    grandparent = parent.parent
    candidates = [
        parent / "config.py",
        parent / "local_rag" / "config.py",
        grandparent / "config.py",
        grandparent / "local_rag" / "config.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.parent
    raise ModuleNotFoundError(
        "Could not find config.py near agents/. Checked:\n"
        + "\n".join(f"  - {c}" for c in candidates)
        + "\nEdit _find_pipeline_root() in specialists.py to add your actual path."
    )


_PIPELINE_ROOT = _find_pipeline_root()
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from config import OLLAMA_GENERATION_MODELS  # noqa: E402

# This is the *reasoning* model -- the one deciding which tool to call
# next, or how to split a compound question. It is a separate concern
# from generate_answer()'s own model choice (that one lives inside
# server.py / generation/ollama_generator.py and is not touched here).
# Reusing OLLAMA_GENERATION_MODELS[0] rather than inventing a second model
# constant keeps this a one-model local setup, matching the project's
# "no paid APIs, everything local" constraint -- swap this if you want a
# smaller/faster model doing routing-style reasoning than the one doing
# final-answer generation.
_REASONING_MODEL = OLLAMA_GENERATION_MODELS[0]

# --- Model routing by difficulty ---------------------------------------
# Every reasoning call anywhere in this project -- the supervisor's own
# routing decision (supervisor.py), contextualize's follow-up rewrite
# (contextualize.py), and every specialist's own LLM call below -- used
# to hardcode this SAME _REASONING_MODEL regardless of how easy or hard
# the actual request was. That is the gap this section closes: a second,
# smaller/faster model for requests a cheap heuristic judges simple, so
# the larger _REASONING_MODEL is reserved for requests that actually
# need it. Mirrors the class's own "Model Routing: The Biggest Single
# Win" guidance -- route on difficulty, not on topic, and do it with a
# heuristic rather than a second LLM call (an LLM call spent deciding
# which LLM to call would spend exactly the cost this exists to save).
#
# _REASONING_MODEL above is now specifically the LARGE tier -- kept
# under its original name (rather than renamed) so every existing import
# of it (supervisor.py, contextualize.py both already do
# `from agents.specialists import Specialist, _REASONING_MODEL`) keeps
# working unchanged; _LARGE_REASONING_MODEL is just a readability alias
# for call sites that want to name both tiers side by side.
_LARGE_REASONING_MODEL = _REASONING_MODEL
_SMALL_REASONING_MODEL = OLLAMA_GENERATION_MODELS[2]  # "phi3" -- fast, small, decent quality; the easy-request tier

# Short, nameable signals that push a request into the LARGE tier --
# deliberately NOT a learned classifier and NOT a second LLM call (see
# this section's own comment above), so a misroute here is something a
# human reading this list can immediately explain, the same
# "mechanically checkable, not a black box" preference this project's
# guardrails (guardrails.py) and eval harness already apply elsewhere.
_COMPLEX_REQUEST_KEYWORDS = (
    "compare", "comparison", "versus", " vs ", "both",
    "difference between", "combine", "and also",
    "step by step", "step-by-step", "explain in detail",
    "why does", "why do", "how does", "relationship between",
)
_COMPLEX_REQUEST_WORD_COUNT = 25  # a long question is rarely a one-fact lookup


def classify_request_difficulty(text: Optional[str]) -> Literal["simple", "complex"]:
    """
    Cheap, explainable heuristic -- word count plus a short keyword list
    -- for whether `text` (a question, or a follow-up needing a rewrite)
    is worth the LARGE reasoning tier or can be handled by the SMALL one.
    Never calls a model itself -- see this module's "Model routing by
    difficulty" section comment above for why that would defeat the
    point.

    Deliberately conservative: this only ever escalates a request to
    "complex", never downgrades one back to "simple" once any signal
    fires. A wrong "complex" guess costs a little extra latency on the
    large model; a wrong "simple" guess risks a genuinely multi-part
    question getting the weaker model's routing/rewrite/generation. When
    in doubt, escalate -- same asymmetric-cost reasoning
    DEFAULT_SKIP_REROUTE_IF_ANSWERED's own docstring (supervisor.py)
    already applies to a different tradeoff in this codebase.
    """
    if not text:
        return "simple"
    lowered = text.lower()
    if len(text.split()) > _COMPLEX_REQUEST_WORD_COUNT:
        return "complex"
    if any(kw in lowered for kw in _COMPLEX_REQUEST_KEYWORDS):
        return "complex"
    if text.count("?") > 1:
        return "complex"
    return "simple"


def select_reasoning_model(text: Optional[str]) -> str:
    """
    The one function every reasoning call site in this project should
    call instead of hardcoding _REASONING_MODEL directly -- returns
    _SMALL_REASONING_MODEL or _LARGE_REASONING_MODEL depending on
    classify_request_difficulty(text). Centralized here (not
    reimplemented per call site) so every reasoning call in the project
    tiers the same way off the same rule, and so that rule only ever
    needs updating in one place. supervisor.py and contextualize.py both
    already import _REASONING_MODEL from this module -- import this
    alongside it the same way.
    """
    return (
        _LARGE_REASONING_MODEL
        if classify_request_difficulty(text) == "complex"
        else _SMALL_REASONING_MODEL
    )

Specialist = Callable[[AgentState], Awaitable[dict]]


def _escape_markdown_caption(caption: Optional[str]) -> Optional[str]:
    """
    Neutralize a caption before it's interpolated into hand-built
    `![caption](url)` markdown syntax (see _format_image_result /
    _format_image_result_embedded below).

    Duplicated from mcp_server/image_tools.py's own
    _escape_markdown_caption rather than imported, same reasoning
    _format_image_result's own docstring already gives for duplicating
    its formatting logic instead of crossing the agents/<->mcp_server
    boundary for it.

    Captions are free-form VLM output (config.IMAGE_CAPTION_PROMPT has
    no constraint on punctuation), so an unescaped "]" prematurely
    closes the image's `![...]` alt-text span -- everything after it
    (the whole `(url)` destination) then falls through as plain
    paragraph text instead of being parsed as an image. Harmless-looking
    for a short `/images/...` path; for a data: URI carrying a
    multi-hundred-KB base64 payload (_format_image_result_embedded),
    the same bug spills the raw base64 string into the chat as visible
    text -- this is the fix for exactly that failure mode.

    Escapes backslash and square brackets and collapses newlines/
    repeated whitespace to single spaces (a caption spanning multiple
    lines would also break this single-line syntax). Returns the input
    unchanged if it's empty/None, so `caption or "(no caption
    available)"` fallback logic at each call site still works untouched.
    """
    if not caption:
        return caption
    collapsed = " ".join(caption.split())
    return "".join(f"\\{ch}" if ch in ("\\", "[", "]") else ch for ch in collapsed)


# Mirrors frontend/src/attachments.ts's own send() -- the exact
# `<attachment name=... status="...">` shape it builds, and
# frontend/src/runtime.ts's onNew appends onto whatever the person
# actually typed (see that file's own `messageForServer` construction)
# before the message ever reaches POST /chat. Duplicated here rather
# than imported from anywhere -- there's no shared build step across the
# Python backend / TypeScript frontend boundary, the same tradeoff this
# project already accepts elsewhere for tightly-coupled string formats
# (agents/api.py used to keep its own copy of this exact regex for the
# same reason, before that code was reverted -- see this constant's own
# call sites below and supervisor.py's own docstring for what replaced
# it).
_ATTACHMENT_MARKER_RE = re.compile(r'<attachment name=(.*?) status="[^"]*"(?:\s+chunks=\d+)?>')


def _message_carries_attachment(request_text: str) -> bool:
    """
    True when `request_text` contains at least one `<attachment ...>`
    marker -- i.e. this message is a real, first-party signal that
    something was JUST uploaded into this conversation, generated by the
    system itself (attachments.ts's send()), never something a person
    typed by hand. Used by contextualize.py (to skip rewriting a message
    carrying one, so the marker's exact text survives into routing) and
    by supervisor.py (as a deterministic pre-LLM routing check straight
    to `personal_docs` -- see that module's own docstring for the
    confirmed live-run mis-route this exists to close). Unlike
    `_looks_like_invoice_followup`, this needs no separate intent-word
    gate: an attachment marker can't coincidentally appear in an
    unrelated message the way a product's own distinguishing words
    could -- it's only ever present because the system put it there.
    """
    return bool(_ATTACHMENT_MARKER_RE.search(request_text))


# Minimum CLIP cosine-similarity score (see vectorstore/chroma_store.py's
# own `"score": 1 - distance` convention -- 1.0 is a perfect match, 0.0
# is orthogonal/unrelated) a retrieve_images/retrieve_images_embedded hit
# must clear before retrieval_qa_node auto-attaches it to an otherwise
# text-only content answer. Unlike image_qa (where showing an image IS
# the point of the specialist, so it shows whatever its top-k comes back
# with, unfiltered), retrieval_qa only ever shows one as a SUPPLEMENT to
# a real text answer nobody explicitly asked to see -- an irrelevant
# image bolted onto an otherwise-correct answer is worse than no image
# at all, since it reads as the system misunderstanding the question.
# Deliberately conservative and easy to retune: `[specialists] retrieval_qa:
# auto-image` lines below log the actual top score on every attempt, so
# watch those against your own corpus's real score distribution and
# adjust this constant once you've seen where genuinely-relevant vs.
# coincidental matches actually land for your own embedding model.
_AUTO_IMAGE_MIN_SCORE = 0.24

# Safety cap on how large a personal upload's image file this process
# will read+base64-encode before embedding it in a chat message. Same
# value and reasoning as mcp_server/image_tools.py's own
# MAX_IMAGE_BYTES_FOR_B64 and local_rag/personal_rag.py's own copy of the
# same constant -- kept in sync by eye across all three rather than
# imported, since each of the three modules that needs it has its own,
# separate reason not to depend on either of the other two (see this
# module's own docstring on the agents/<->mcp_server boundary, and
# personal_rag.py's docstring on why it has no dependency on mcp_server/
# at all).
_MAX_PERSONAL_IMAGE_BYTES_FOR_B64 = 5 * 1024 * 1024


def _read_local_image_data_uri(image_path: Optional[str]) -> Optional[str]:
    """
    Read a personal-upload image's bytes straight off THIS process's own
    filesystem and encode them as a `data:<mime>;base64,...` URI.

    This is a deliberate, narrow exception to "agents/ never touches the
    pipeline's files directly, only through the MCP server" (see this
    module's own top docstring): the path being read here was written by
    THIS SAME PROCESS, moments (or turns) earlier, by
    local_rag/personal_rag.py's _persist_personal_image() -- called
    in-process from agents/api.py's own upload endpoint, the same
    established "ingest is a direct pipeline call" exception
    personal_rag.py's own module docstring already documents. Reading
    that file back is the read-side of the exact same exception, not a
    new one: there is no MCP tool that would make this a cross-process
    call anyway, since the bytes were never handed to the MCP server
    subprocess in the first place.

    Never raises: a missing/unreadable/oversized file all degrade to
    None (the caller then falls back to text-only), the same "one bad
    image never breaks the whole answer" contract
    mcp_server/image_tools.py's own _encode_image_base64 and
    personal_rag.py's own _image_to_data_uri both already follow.
    """
    if not image_path:
        return None
    path = Path(image_path)
    try:
        if not path.is_file() or path.stat().st_size > _MAX_PERSONAL_IMAGE_BYTES_FOR_B64:
            return None
        raw_bytes = path.read_bytes()
    except OSError:
        return None
    mime_type, _ = mimetypes.guess_type(str(path))
    if not mime_type:
        mime_type = "application/octet-stream"
    return f"data:{mime_type};base64,{base64.b64encode(raw_bytes).decode('ascii')}"


def _format_personal_image_chunk(chunk: dict) -> str:
    """
    Render one search_personal_documents() hit that originated from an
    uploaded IMAGE (metadata["original_modality"] == "image", see
    personal_rag.ingest_upload's own docstring for how that gets set) as
    a markdown image block -- the actual picture the person uploaded,
    embedded via _read_local_image_data_uri above, with its VLM caption
    underneath. `chunk["text"]` IS that caption (personal_rag.py stores
    the caption text itself as the chunk's embedded text, see
    ingest_upload), so there is no separate caption lookup needed.

    Falls back to caption-only text (no image block) when
    metadata["image_path"] is missing or the file can no longer be read
    -- same "show what you have, don't error" degrade every other
    image-rendering helper in this project already follows.

    `chunk` is expected to be a dict (the same {"text", "score",
    "metadata"} shape every search_personal/latest_personal_image hit
    uses) -- but a confirmed live crash showed a non-dict CAN reach this
    far (see _best_personal_image_result's own docstring for where that
    came from and how it's now filtered upstream). Guarded here too, as
    a last line of defense: a non-dict degrades to a plain apology line
    instead of an AttributeError taking the whole turn down.
    """
    if not isinstance(chunk, dict):
        print(f"[specialists] _format_personal_image_chunk got a non-dict hit "
              f"({type(chunk).__name__}) -- rendering a plain fallback instead of "
              f"crashing on it.", file=sys.stderr)
        return "*(found something, but couldn't read its details)*"
    metadata = chunk.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    caption = _escape_markdown_caption(chunk.get("text")) or "(no caption available)"
    data_uri = _read_local_image_data_uri(metadata.get("image_path"))
    if not data_uri:
        return f"*{caption}*"
    return f"![{caption}]({data_uri})\n*{caption}*"


def _personal_image_display_block(chunk: dict) -> str:
    """
    Just the picture itself (embedded, captioned via markdown alt text)
    -- no caption line repeated below it, unlike
    _format_personal_image_chunk above. For combining with a REAL
    generated answer immediately after (see personal_docs_node /
    image_qa_node's own image_hit branch below): restating the caption a
    second time directly above an answer that's ALSO grounded in that
    same caption would just be noisy, and a confirmed live report showed
    the un-combined version reads badly on a genuine question ("explain
    this image") -- getting back only the caption verbatim, unchanged,
    feels like the question was ignored rather than answered, especially
    right after that same caption was already shown once at upload time.

    Falls back to the caption as plain text (same as
    _format_personal_image_chunk's own fallback) when there's no image
    to embed, so there's always SOMETHING to combine with the generated
    answer even without a picture.
    """
    if not isinstance(chunk, dict):
        return ""
    metadata = chunk.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    caption = _escape_markdown_caption(chunk.get("text")) or "(no caption available)"
    data_uri = _read_local_image_data_uri(metadata.get("image_path"))
    if not data_uri:
        return f"*{caption}*"
    return f"![{caption}]({data_uri})"


def _image_answer_content(image_block: str, image_answer: str, question: str) -> str:
    """
    Combines a personal image's display block with its real generated
    answer -- except when `question` itself carries an `<attachment ...>`
    marker (see `_message_carries_attachment`'s own docstring), meaning
    THIS turn's own upload already showed this exact image once, via
    `_post_captioned_images_to_chat`'s preview pair (agents/api.py).
    Text-only in that case: re-embedding the same picture again here
    would be the visual half of a "one prompt, two answers" duplication
    (see supervisor.py's own safety-net-0a comment for the routing half
    of the same fix) -- the image itself is already visible one turn up,
    so only the answer text adds anything new.

    Shared by all three of this module's "personal image, paired with a
    real generated answer" branches (`retrieval_qa_node`,
    `personal_docs_node`, `image_qa_node`) so the three can't quietly
    drift into three different answers to the exact same situation.
    """
    if not image_block:
        return image_answer
    if _message_carries_attachment(question):
        return image_answer
    return f"{image_block}\n\n{image_answer}"


async def _best_personal_image_result(
    search_personal_tool, latest_personal_image_tool, thread_id: Optional[str], question: str
) -> Optional[dict]:
    """
    THE structural fix for "show the image the person actually uploaded,
    never a corpus image that merely resembles it" -- called from both
    personal_docs_node and image_qa_node below, so that guarantee holds
    regardless of which of the two the supervisor's own routing decision
    happens to pick for a given turn.

    Why this needs to exist in BOTH nodes rather than just relying on
    prompts.py's supervisor routing rule ("a question about a file the
    user personally uploaded -> personal_docs, never retrieval_qa or
    corpus_meta"): a live run confirmed the small local routing model
    does not reliably follow that rule for image_qa specifically -- a
    follow-up question about a just-uploaded image was routed to
    image_qa, which then ran its own corpus-wide CLIP retrieval and
    returned a DIFFERENT, merely-similar-looking treatise image instead
    of the one actually uploaded. Per this project's established
    "structural guardrail over prompt wording" preference (see this
    module's own top docstring, and guardrails.py's identical reasoning
    for input_guard/output_guard), the fix is not a stronger prompt --
    it's making the specialist itself structurally prefer the thread's
    own upload whenever one is relevant, so a misroute can no longer
    surface the wrong picture.

    TWO lookups, tried in order, both scoped to THIS thread_id alone
    (never anyone else's, never the shared corpus):

      1. search_personal_documents -- semantic search over everything
         this thread has uploaded, same as before. Returns the
         best-scoring hit whose metadata marks it as having come from an
         uploaded image (search_personal already returns hits ordered by
         relevance, so the first image-origin hit IS the best one).

      2. latest_personal_image -- tried ONLY if (1) found nothing. A
         confirmed live scenario (not hypothetical) shows why (1) alone
         isn't enough: a thread that uploads two visually/texturally
         similar images can have a generic follow-up ("what is this?")
         rank the OLDER upload higher in embedding space than the one
         that was actually just sent, or -- for a truly generic
         question with no strong lexical anchor to any one caption --
         miss ranking the just-uploaded image in the top-k at all. Pure
         cosine similarity has no concept of "just now"; recency does.
         Falling back to "the most recently uploaded image in this
         thread" is a strictly better default than falling through to
         corpus-wide retrieval, which is exactly the "shows a similar
         corpus picture instead" bug this whole function exists to rule
         out.

    Both lookups' results are type-checked before use -- confirmed live
    crash (`AttributeError: 'str' object has no attribute 'get'`) came
    from exactly this function trusting unwrap_tool_result()'s return
    shape unconditionally: unwrap_tool_result() falls back to returning
    a raw TEXT STRING whenever its JSON parse fails (see its own
    docstring -- this is intentional for generate_answer's plain-string
    responses, but wrong here, where a list of dicts is always what's
    expected). Iterating a string yields its individual CHARACTERS, each
    one a one-letter string with no `.get` method -- exactly the crash
    observed. Every item is now isinstance-checked before `.get()` is
    ever called on it; anything else is skipped (or, for the
    latest-image lookup, treated the same as "nothing found") and
    logged, rather than raised on.

    Returns None only if NEITHER lookup finds a genuine, well-shaped hit
    -- meaning this thread genuinely has no uploaded image at all, the
    tools aren't available, or thread_id is unset (e.g. graph.py's own
    ask()/CLI/eval script, none of which have a persisted thread to
    scope to) -- the one case where falling through to corpus-wide image
    retrieval in the caller is actually the right behavior.
    """
    if not thread_id:
        return None

    if search_personal_tool is not None:
        raw_chunks = await search_personal_tool.ainvoke(
            {"thread_id": thread_id, "query": question, "k": 8}
        )
        unwrapped = unwrap_tool_result(raw_chunks)
        chunks = unwrapped if isinstance(unwrapped, list) else []
        if unwrapped and not isinstance(unwrapped, list):
            print(f"[specialists] search_personal_documents returned an unexpected shape "
                  f"({type(unwrapped).__name__}, expected a list) -- treating as no results "
                  f"rather than crashing on it.", file=sys.stderr)
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            metadata = chunk.get("metadata")
            if isinstance(metadata, dict) and metadata.get("original_modality") == "image":
                return chunk

    if latest_personal_image_tool is not None:
        raw_latest = await latest_personal_image_tool.ainvoke({"thread_id": thread_id})
        latest = unwrap_tool_result(raw_latest)
        if isinstance(latest, dict) and latest:
            return latest

    return None


def _last_human_text(state: AgentState) -> str:
    """
    Pull the most recent HumanMessage's content out of state["messages"].

    Every specialist answers the latest question, not the full transcript
    -- if the supervisor (Phase 3) re-routes a follow-up question to a
    different specialist mid-conversation, that specialist should see the
    new question, not re-answer an earlier one it never saw.
    """
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    raise ValueError("No HumanMessage found in state['messages']")


def _format_document_list(corpus: dict) -> str:
    """
    Render the corpus://documents resource's parsed dict as the flat
    bullet list CORPUS_META_SYSTEM_PROMPT's {document_list} slot expects.
    Kept as its own function so the empty-corpus case (nothing ingested
    yet) is handled once, in one place, with an explicit message rather
    than an empty string that would silently make the specialist's
    system prompt look truncated or broken.
    """
    documents = corpus.get("documents", [])
    if not documents:
        return "(corpus is empty -- no documents have been ingested yet)"
    return "\n".join(f"- {d['filename']} ({d['chunk_count']} chunks)" for d in documents)


# ---------------------------------------------------------------------
# Shared helpers for the new specialists (image_qa, painting_lookup,
# product_search, invoice)
# ---------------------------------------------------------------------

# The hidden, machine-parseable footer product_search_node embeds in its
# own AIMessage, and invoice_node later parses back out of the message
# history. An HTML comment rather than a fenced code block, deliberately:
# it renders invisibly in any markdown client, so the user only sees the
# human-readable comparison + numbered list above it, while the exact
# structured data (name/price/url/source) invoice_node needs survives
# byte-for-byte in the transcript -- no LLM re-parsing of the numbered
# list's prose is ever needed to reconstruct it. Same "structural
# guardrail over prompt wording" preference as everywhere else in this
# project: the number invoice_node bills the user is read from this JSON,
# never re-derived from an LLM's own summary of it.
_PRODUCT_DATA_RE = re.compile(r"<!--PRODUCT_DATA:(\[.*?\])-->", re.DOTALL)

_SELECT_ALL_PHRASES = (
    "all of them", "everything", "all the items", "all items",
    "buy them all", "get them all", "both of them", "the whole list",
    "all products", "everything you found",
)

# Keyword cues for product_search_node's beginner/professional split --
# checked against a candidate's title + snippet, lowercased. Order
# matters only in that BOTH lists are checked before falling back to the
# price-relative-to-pool tiebreak in `_classify_tier` below; within a
# list, any single match is enough.
_PROFESSIONAL_TIER_KEYWORDS = (
    "professional", "artist grade", "artist-grade", "studio", "premium",
    "kolinsky", "sable", "fine art", "master", "high-end", "high end",
    "pro grade", "pro-grade", "conservation", "archival", "heavyweight",
)
_BEGINNER_TIER_KEYWORDS = (
    "student", "beginner", "starter", "kids", "value pack", "budget",
    "basic", "learn", "starter kit", "for beginners", "value set",
    "entry level", "entry-level", "introductory", "kids'",
)


def _classify_tier(item: dict) -> str:
    """
    Best-effort "beginner" vs "professional" tag for one
    search_art_supplies() result, from keyword cues in its title/snippet.
    Returns "unclassified" when neither keyword list matches -- resolved
    against the candidate pool's median known price by the caller (see
    product_search_node), not decided here, since a price-relative
    judgment only makes sense in the context of the whole pool, not one
    item in isolation.

    Honest limitation, same "say what a free/keyless approach can't
    promise" pattern search_art_supplies' own price-extraction docstring
    already states: a search snippet rarely states a product's intended
    skill level outright, so this is a keyword heuristic, not a real
    product-attribute lookup. A production version would use a real
    product API's category/attribute data instead.
    """
    text = f"{item.get('title', '')} {item.get('snippet', '')}".lower()
    if any(kw in text for kw in _PROFESSIONAL_TIER_KEYWORDS):
        return "professional"
    if any(kw in text for kw in _BEGINNER_TIER_KEYWORDS):
        return "beginner"
    return "unclassified"


def _tier_candidates(candidates: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Split `candidates` (already tagged with a "tier" key by the caller
    resolving `_classify_tier`'s "unclassified" cases against the pool's
    median price) into (beginner_list, professional_list), each still in
    the search backend's own original relative order -- the actual
    top-5-per-tier selection (price-known-first within each tier) is
    done by the caller, not here, so this function has exactly one job:
    partition by tier.
    """
    beginner = [c for c in candidates if c["tier"] == "beginner"]
    professional = [c for c in candidates if c["tier"] == "professional"]
    return beginner, professional


def _pick_top(pool: list[dict], n: int = 5) -> list[dict]:
    """
    Up to `n` items from `pool`, price-known items first (stable sort
    preserves the search backend's own relative ranking within each
    group) -- the same selection rule product_search_node used for its
    original single top-5 list, now applied independently within each
    tier so a tier with fewer price-known listings doesn't lose slots to
    the OTHER tier's price-known items.
    """
    ranked = sorted(pool, key=lambda c: c.get("price") is None)
    return ranked[:n]


def _embed_product_data(text: str, items: list[dict]) -> str:
    return f"{text}\n\n<!--PRODUCT_DATA:{json.dumps(items)}-->"


def _parse_product_data(message_content: str) -> list[dict]:
    """
    Pull the structured item list back out of one product_search
    message's content. Returns [] if the marker is missing or its JSON
    is malformed -- never raises, since a hand-edited or truncated
    transcript (e.g. output_guard redacting something nearby) shouldn't
    crash invoice_node; it should just contribute zero items from that
    particular message.
    """
    match = _PRODUCT_DATA_RE.search(message_content)
    if not match:
        return []
    try:
        data = json.loads(match.group(1))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def _collect_product_catalog(messages: list) -> list[dict]:
    """
    Scan the ENTIRE conversation history (not just the current turn --
    deliberately different scope from every other specialist's
    _last_human_text/_current_turn_context, which only look at the
    latest turn) for every product_search AIMessage, parse each one's
    embedded PRODUCT_DATA, and return a de-duplicated, order-preserving
    catalog of every product the user has been shown so far in this
    conversation.

    Full-history scope is intentional here, but only for ANSWERING "has
    anything ever been searched in this conversation" (invoice_node's
    first check) -- it is deliberately NOT what item SELECTION is scoped
    to; see _select_invoice_items' own docstring for why matching itself
    is restricted to the latest batch only. Deduplicated by (name, url)
    so the same item found by two separate searches doesn't get counted
    twice just for this existence check.
    """
    seen: set[tuple[str, str]] = set()
    catalog: list[dict] = []
    for msg in messages:
        if not (isinstance(msg, AIMessage) and getattr(msg, "name", None) == "product_search"):
            continue
        for item in _parse_product_data(msg.content if isinstance(msg.content, str) else ""):
            key = (str(item.get("name", "")).lower(), str(item.get("url", "")))
            if key not in seen:
                seen.add(key)
                catalog.append(item)
    return catalog


def _latest_product_search_batch(messages: list) -> list[dict]:
    """
    The structured item list embedded in the MOST RECENT product_search
    AIMessage in `messages` -- [] if none exists yet.

    Factored out of invoice_node (which used to inline this exact
    backward walk itself) so supervisor.py's pre-LLM invoice/
    product_search disambiguation (see that module's "safety net 0")
    matches against the EXACT SAME batch invoice_node itself will select
    from -- if the router and invoice_node ever disagreed about which
    batch "the most recent search" means, a request the router correctly
    recognized as an invoice follow-up could still land on the WRONG
    batch once it actually got there. Scoping to "the latest search"
    rather than the full multi-search catalog is itself deliberate; see
    _select_invoice_items' own docstring for why.
    """
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "name", None) == "product_search":
            return _parse_product_data(msg.content if isinstance(msg.content, str) else "")
    return []


def _normalize_for_match(text: str) -> str:
    """
    Lowercase, non-alphanumeric characters collapsed to spaces, repeated
    whitespace collapsed to one -- shared normalization for both the
    exact-name check and the word-tokenizing below, so a listing's own
    punctuation-heavy title ("15Pcs Paint Brushes Value Pack | 15
    Different Types...") and a person's plain-text mention of the same
    product line up regardless of casing or punctuation differences.
    """
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _significant_words(name: str) -> set[str]:
    """4+ letter/digit tokens from `name`, normalized -- same length
    floor _select_invoice_items always used for "a word that actually
    means something" (excludes "of", "for", "the", etc. without a
    hardcoded stopword list)."""
    return {w for w in re.findall(r"[a-z0-9]+", _normalize_for_match(name)) if len(w) >= 4}


def _common_words(batch: list[dict]) -> set[str]:
    """
    Words shared by MORE THAN HALF of `batch`'s item names -- generic
    descriptors that don't actually distinguish one item from another
    (e.g. "amazon", "paint", "brushes" across a batch that's entirely
    paint-brush listings). Computed fresh per batch, not a hardcoded
    domain stopword list, since what counts as "generic" depends
    entirely on what this particular search actually returned -- a batch
    of easels and canvases would exclude a completely different set of
    words.
    """
    if not batch:
        return set()
    freq: dict[str, int] = {}
    for item in batch:
        for w in _significant_words(str(item.get("name", ""))):
            freq[w] = freq.get(w, 0) + 1
    half = len(batch) / 2
    return {w for w, c in freq.items() if c > half}


def _match_by_id(request_text: str, batch: list[dict]) -> list[dict]:
    """
    Item(s) in `batch` whose own `id` -- the short "b1"/"g3"-style label
    product_search_node itself assigns and renders as the numbered-list
    marker in its own answer (see that node's _render_tier_section:
    "b1. **Title** ... ") -- appears as a standalone word in the request.

    CONFIRMED FAILURE this closes: product_search_node's own rendered
    answer trains the user to refer to an item by that short id ("I'll
    take b3", "add g1 and g4"), not by its full listing title -- but
    neither of _score_items_by_name's two name-based passes below ever
    looked at `id` at all, only at `name`. A bare id like "b3" has no 4+
    letter word and never appears as a substring of any item's own name,
    so it scored zero against every item in BOTH of those passes and
    fell all the way through to the generic-item fallback (pass 3
    below), or -- if that also came up empty -- to
    _select_invoice_items' own last-resort "assume the whole batch"
    fallback, which silently invoiced EVERY item in the batch instead of
    just the one the person actually named. Checked first, before any
    name-based pass: an id is an unambiguous reference to exactly one
    item (assigned fresh, once, per search), so there's no reason to
    fall through to a fuzzier match once an id match is found.
    """
    if not batch:
        return []
    request_words = set(re.findall(r"[a-z0-9]+", _normalize_for_match(request_text)))
    return [item for item in batch if str(item.get("id", "")).strip().lower() in request_words]


def _score_items_by_name(request_text: str, batch: list[dict]) -> list[dict]:
    """
    The single best-matching item(s) in `batch` for a request that names
    one (or a few) of them directly -- deterministic and repeatable for
    the same inputs.

    CONFIRMED FAILURE this replaces: the previous rule selected any item
    with even ONE 4+ letter word from its name appearing anywhere in the
    request. That's fine for a batch of dissimilar items, but a
    product_search batch is often ten listings of the SAME kind of
    product ("Amazon.com: Brushes", "Amazon.com: Best Paint Brushes",
    "15Pcs Paint Brushes Value Pack...") -- "paint" and "brushes" alone
    appear in nearly every title, so naming ONE item by a couple of its
    own words matched almost the entire batch instead of the one item
    actually asked for.

    Three passes, most confident first:
      0. Id: see `_match_by_id` above. If any item's own short id ("b1",
         "g3", ...) appears as a standalone word in the request, ONLY
         those id matches are returned -- an id reference is never worth
         diluting with a name-based guess.
      1. Exact(ish): an item's own normalized full name appears verbatim
         inside the normalized request (covers pasting or closely
         quoting a listing's title back). If any item matches this way,
         ONLY exact matches are returned -- a verbatim name match is
         never worth diluting with a looser candidate.
      2. Distinguishing-word: score each item by how many of ITS OWN
         significant (4+ letter) words -- excluding words shared by more
         than half the batch, see `_common_words` -- also appear in the
         request. Only the item(s) at the single HIGHEST score (and only
         if that score is > 0) are returned, not every item with any
         overlap at all -- this is the actual fix for the over-matching
         above.
      3. Generic-item fallback: ONLY reached if pass 2 found nothing at
         all (every item scored zero on its distinguishing words). Items
         whose every word is common batch-wide (e.g. a listing literally
         titled "Amazon.com: Brushes", where "amazon" and "brushes" are
         both shared by most of the batch) never get a chance to compete
         in pass 2 -- this pass rescoring THOSE items only, on their
         full unfiltered word set, so a generically-named item can still
         match a generic request instead of being structurally
         unmatchable. Deliberately scoped to generic items only, not
         re-run for every item: re-scoring everything on unfiltered words
         would just reintroduce pass 2's over-matching problem for a
         batch that's already mostly one shared word.
    """
    if not batch:
        return []

    id_matches = _match_by_id(request_text, batch)
    if id_matches:
        return id_matches

    normalized_request = _normalize_for_match(request_text)

    exact = [
        item
        for item in batch
        if _normalize_for_match(str(item.get("name", ""))) in normalized_request
        and _normalize_for_match(str(item.get("name", "")))
    ]
    if exact:
        return exact

    common = _common_words(batch)
    request_words = set(re.findall(r"[a-z0-9]+", normalized_request))

    def _scored(candidates: list[dict], words_for) -> list[tuple[int, dict]]:
        out = []
        for item in candidates:
            score = len(words_for(item) & request_words)
            if score > 0:
                out.append((score, item))
        return out

    def _distinguishing(item: dict) -> set[str]:
        return _significant_words(str(item.get("name", ""))) - common

    scored = _scored(batch, _distinguishing)
    if not scored:
        generic_items = [item for item in batch if not _distinguishing(item)]
        scored = _scored(generic_items, lambda it: _significant_words(str(it.get("name", ""))))

    if not scored:
        return []

    top_score = max(score for score, _ in scored)
    return [item for score, item in scored if score == top_score]


def _select_invoice_items(request_text: str, latest_batch: list[dict]) -> tuple[list[dict], str]:
    """
    Decide which items from `latest_batch` the invoice request refers to,
    and return (selected_items, assumption_note). `assumption_note` is ""
    when the match was unambiguous (explicit "all", or a clear name
    match); otherwise it explains the fallback taken, so invoice_node can
    surface it to the user rather than silently guessing.

    Scope is deliberately limited to `latest_batch` -- the items from the
    MOST RECENT product_search call, i.e. "the previous chat" turn --
    not the full multi-turn catalog a conversation may have accumulated
    over many separate searches. Matching against the whole conversation
    ran into the same problem this module's docstring on
    `_score_items_by_name` describes, one level up: an item found several
    turns ago (and never mentioned since) could get silently invoiced
    alongside whatever the user actually just asked for. Billing should
    only ever include items the user could plausibly still be looking at
    -- `invoice_node` already walks backward through the FULL history to
    find that most-recent batch, so "search for it in previous chat"
    still holds across turns, it's just anchored to the last search
    shown, not every search ever run.

    Matching strategy, in order:
      1. An explicit "all of them" / "everything" style phrase -> every
         item in `latest_batch`.
      2. `_score_items_by_name` -- the exact-name-first, then
         distinguishing-word-scored match described in that function's
         own docstring. Deterministic: the same request against the same
         batch always selects the same item(s).
      3. If nothing matched at all, fall back to the full `latest_batch`
         with an explicit assumption note -- "these items" most plausibly
         means whatever was just shown.
    """
    lowered = request_text.lower()

    if any(phrase in lowered for phrase in _SELECT_ALL_PHRASES):
        return latest_batch, ""

    matched = _score_items_by_name(request_text, latest_batch)
    if matched:
        return matched, ""

    if latest_batch:
        return (
            latest_batch,
            "(I didn't see specific item names or \"all\" in your message, so "
            "I assumed you meant the products from the most recent search above.)",
        )

    return [], ""


# Action words/phrases that signal "this message is about PAYING for
# something", not just mentioning a product -- checked as plain
# substrings against the lowercased request, same looser-than-strict
# tradeoff _SELECT_ALL_PHRASES and the tier-keyword lists above already
# accept. Deliberately includes the exact phrasing
# SPECIALIST_ROUTING_EXAMPLES["invoice"] (prompts.py) already teaches the
# supervisor's own LLM ("how much would ... cost in total") -- this list
# and that worked example should keep agreeing on what "sounds like an
# invoice request" means, not drift into two different definitions of
# the same idea.
#
# "want"/"i'd like"/"i would like" ADDED after a confirmed live-run
# failure: "I want the Dainayw Fine Detail Paint Brush Set" and "I want
# these" (naming items _score_items_by_name below DID match against the
# latest batch) both fell through this gate with the ORIGINAL list --
# "want" wasn't on it -- so _looks_like_invoice_followup returned False
# and the message went to the LLM-based supervisor routing instead,
# which misrouted it back to product_search THREE times in a row before
# the user finally said "buy" explicitly. Each of those extra
# product_search rounds was a live web search, not a cache, and
# returned progressively different (sometimes worse) results for the
# same items -- by the time "buy" finally landed, the batch it matched
# against had lost the real pricing data entirely, and the resulting
# invoice could price NONE of the items. Adding "want" here is safe
# specifically because _looks_like_invoice_followup ALSO requires a real
# _score_items_by_name match against the latest batch (see that
# function's own docstring) -- a bare "I want to learn more about oil
# painting" still won't match with nothing in latest_batch resembling
# it, so this doesn't turn every use of "want" into a false invoice
# trigger, only "want" PLUS a genuine named-product match.
_INVOICE_INTENT_PHRASES = (
    "buy", "purchase", "order", "invoice", "receipt", "checkout",
    "check out", "bill me", "add to cart", "how much", "cost",
    "total", "i'll take", "i will take", "get me", "want", "i'd like",
    "i would like",
)


def _looks_like_invoice_followup(request_text: str, latest_batch: list[dict]) -> bool:
    """
    True when `request_text` is confidently an invoice follow-up to
    `latest_batch` (the most recent product_search's own items, see
    `_latest_product_search_batch`) -- used by supervisor.py's pre-LLM
    routing check (its "safety net 0") to route straight to `invoice`
    without asking a small local model to make the product_search-vs-
    invoice call itself. See that module's own comment for the confirmed
    failure mode this exists to head off.

    Two ways to qualify, in order:
      1. An explicit "all of them"/"everything"-style phrase
         (`_SELECT_ALL_PHRASES`) -- self-sufficient on its own, no
         separate intent word required, since these phrases are already
         a curated, specific set unlikely to appear in an unrelated
         message.
      2. An explicit purchase/invoice action word or phrase
         (`_INVOICE_INTENT_PHRASES`) -- REQUIRED here, then combined
         with `_score_items_by_name` finding a real id/name match in
         `latest_batch`. The intent-word requirement is deliberate, not
         redundant: `_score_items_by_name`'s distinguishing-word pass can
         legitimately fire on a genuine, unrelated TECHNIQUE question
         that happens to reuse a listing's own distinguishing words
         (e.g. a "Kolinsky Sable Brush" listing found earlier, and a
         later, perfectly ordinary "how do I clean a kolinsky sable
         brush" question) -- gating on an explicit action word as well
         is what keeps that from being misread as an invoice follow-up.
         This is the same "the user's own action words decide the
         route" principle SUPERVISOR_SYSTEM_PROMPT's own routing-example
         section (prompts.py) already states outright.

    Returns False immediately if `latest_batch` is empty -- there is
    nothing to be a follow-up TO.
    """
    if not latest_batch:
        return False

    lowered = request_text.lower()

    if any(phrase in lowered for phrase in _SELECT_ALL_PHRASES):
        return True

    if not any(phrase in lowered for phrase in _INVOICE_INTENT_PHRASES):
        return False

    return bool(_score_items_by_name(request_text, latest_batch))


# Keyword cue that the current question is asking to find OTHER images
# that visually resemble something the person themselves uploaded --
# "find images like the one I sent", "does the corpus have anything
# similar to this", "show me paintings that look like my photo" --
# rather than asking to see or have explained the upload itself ("what
# is this", "explain this image"). Deliberately narrow and keyword-
# based, same "cheap, cheap-to-miss, catches the common/confirmed
# phrasing" tradeoff _SELECT_ALL_PHRASES/_INVOICE_INTENT_PHRASES above
# already accept for the same kind of intent-word gate -- a missed
# phrasing here just falls through to image_qa_node's existing "show
# the upload back" behavior below (see that node's own comment), never
# a routing decision, so a false negative costs nothing structural.
_SIMILAR_IMAGE_PHRASES = (
    "similar to", "similar images", "similar-looking", "resemble",
    "resembles", "resembling", "look like", "looks like", "look similar",
    "looks similar", "anything like", "something like this",
    "images like", "paintings like", "pictures like", "artwork like",
    "alike",
)


def _looks_like_similar_image_request(request_text: str) -> bool:
    """
    True when `request_text` is asking to find OTHER (corpus) images
    that visually resemble something the person uploaded, rather than
    asking about the upload itself. See `_SIMILAR_IMAGE_PHRASES`'s own
    comment for the exact phrasing this looks for and why a miss here
    is cheap.
    """
    lowered = request_text.lower()
    return any(phrase in lowered for phrase in _SIMILAR_IMAGE_PHRASES)


# ---------------------------------------------------------------------
# Shared helpers for color_palette_node
# ---------------------------------------------------------------------

# A SMALL, independent set of basic color-family words -- deliberately
# NOT the full extended named-color dictionary mcp_server/color_tools.py
# uses for "closest name" lookups and library-backed names. That data
# stays server-side, behind the MCP boundary, same "all pipeline access
# goes through the Phase 1 MCP server" rule this module's own docstring
# states for every specialist. This list only has to be good enough to
# notice "the user named an actual color" versus "the user described a
# mood" -- the real, precise color resolution (parsing a longer/compound
# name, hex, rgb, or looking up whichever library color_tools.py has
# available) happens entirely on the tool side, from the raw substring
# extracted here.
_BASIC_COLOR_WORDS = (
    "red", "orange", "yellow", "green", "blue", "purple", "violet", "pink",
    "brown", "black", "white", "gray", "grey", "teal", "cyan", "magenta",
    "gold", "silver", "indigo", "turquoise", "maroon", "navy", "olive",
    "beige", "ivory", "crimson", "scarlet", "azure", "lavender", "mint",
    "coral", "salmon", "peach", "mustard", "ochre", "sienna", "umber",
    "cobalt", "ultramarine", "viridian", "vermilion", "cerulean",
    "emerald", "ruby", "sapphire", "amber", "rust", "charcoal", "tan",
    "burgundy", "plum", "rose", "lilac", "khaki", "chocolate", "terracotta",
)

_HEX_COLOR_RE = re.compile(r"#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?\b")
_RGB_TRIPLET_TEXT_RE = re.compile(
    r"rgb\s*\(\s*\d{1,3}\s*,\s*\d{1,3}\s*,\s*\d{1,3}\s*\)"
    r"|\b\d{1,3}\s*,\s*\d{1,3}\s*,\s*\d{1,3}\b"
)

# scheme name -> the word(s) in a request that ask for it. Word list
# values are checked as plain substrings, same looser-than-strict
# tradeoff this project's other keyword matching (_classify_tier,
# _select_invoice_items) already accepts.
_SCHEME_WORD_MAP = {
    "monochromatic": ("monochromatic", "monochrome", "mono"),
    "analogous": ("analogous", "analog"),
    "complementary": ("complementary", "complement", "opposite color", "opposite colour"),
    "triadic": ("triadic", "triad"),
}


def _detect_scheme(lowered: str) -> Optional[str]:
    for scheme_name, words in _SCHEME_WORD_MAP.items():
        if any(w in lowered for w in words):
            return scheme_name
    return None


def _extract_explicit_color(text: str, lowered: str) -> Optional[str]:
    """
    A hex code, an rgb-looking triplet, or a basic color-family word
    found directly in the request -- returned as the RAW matched
    substring, for color_tools.parse_color_text to do the real parsing
    against (hex/rgb math, or the library/dictionary name lookup) on the
    tool side. Returns None if nothing that looks like an explicit color
    reference is present, so the caller falls back to treating the whole
    message as a mood description instead.

    A basic-family match also captures one word immediately BEFORE it,
    if present (e.g. "forest" ahead of "green"), so a compound named
    color like "forest green" or "sky blue" survives as one phrase for
    color_tools.py's own (much larger) dictionary to resolve precisely,
    rather than being truncated to the generic base word alone. Safe
    either way if that preceding word isn't actually part of a real
    color name (e.g. "the green") -- color_tools._lookup_named_color
    still finds "green" as a substring match within the wider phrase.
    """
    m = _HEX_COLOR_RE.search(text)
    if m:
        return m.group(0)
    m = _RGB_TRIPLET_TEXT_RE.search(text)
    if m:
        return m.group(0)

    best_match: Optional[re.Match] = None
    for w in _BASIC_COLOR_WORDS:
        found = re.search(rf"\b(?:(\w+)\s+)?{re.escape(w)}\b", lowered)
        if found and (best_match is None or len(found.group(0)) > len(best_match.group(0))):
            best_match = found
    return best_match.group(0) if best_match else None


def _parse_color_request(text: str) -> dict:
    """
    Light, deterministic triage for color_palette_node: does this
    request name an actual color, or describe a mood/feeling? Returns
    {"color": str | None, "mood": str | None, "scheme": str | None} --
    exactly the three arguments the generate_color_palette MCP tool
    expects, with `color` and `mood` mutually exclusive (an explicit
    color reference always wins over treating the message as a mood,
    the same priority color_tools.generate_palette itself applies if
    both were ever somehow both non-empty).

    If no explicit color is found, the ENTIRE original request (minus
    whichever words asked for a specific scheme, longest first so
    "complementary" is stripped whole rather than leaving a stray "ary"
    behind after a shorter alias like "complement" matches first) is
    passed through as `mood` -- color_tools.py's own keyword scorer does
    the real interpretation from there, and reports plainly if nothing
    matched.
    """
    lowered = text.lower()
    scheme = _detect_scheme(lowered)

    explicit_color = _extract_explicit_color(text, lowered)
    if explicit_color:
        return {"color": explicit_color, "mood": None, "scheme": scheme}

    mood_text = text
    all_scheme_words = sorted(
        {w for words in _SCHEME_WORD_MAP.values() for w in words},
        key=len,
        reverse=True,
    )
    for w in all_scheme_words:
        mood_text = re.sub(re.escape(w), "", mood_text, flags=re.IGNORECASE)
    mood_text = re.sub(r"\s+", " ", mood_text).strip()
    return {"color": None, "mood": mood_text or text, "scheme": scheme}


_SCHEME_DISPLAY_NAMES = {
    "monochromatic": "Monochromatic",
    "analogous": "Analogous",
    "complementary": "Complementary",
    "triadic": "Triadic",
}


def _format_swatch_line(item: dict) -> str:
    """
    One color's markdown line: a small swatch image, its hex/rgb, its
    closest name, and a short line on the feeling it can inspire. The
    swatch is a `data:image/svg+xml;base64,...` URI color_tools.py's own
    `swatch_data_uri` already builds -- embedded directly as a markdown
    image, no static file server involved. This needed ZERO frontend
    changes to render: frontend/src/components/MarkdownText.tsx's
    `allowImageDataUris` already allowlists any `data:image/*;base64`
    src (added earlier to unblock image_qa's own embedded corpus
    images), and `svg+xml` matches that same allowlist regex.
    """
    rgb = item["rgb"]
    alt = f"{item['name']} swatch {item['hex']}"
    swatch_md = f"![{alt}]({item['swatch']})"
    return (
        f"{swatch_md} **{item['name']}** -- `{item['hex']}` "
        f"/ rgb({rgb['r']}, {rgb['g']}, {rgb['b']})  \n"
        f"_{item['feeling']}_"
    )


def _format_color_palette_answer(result: dict) -> str:
    """
    Render generate_color_palette's structured dict into the markdown
    answer shown to the user -- every hex/rgb/name/feeling value is
    taken directly from the tool's own output, never re-typed or
    re-derived by an LLM (this specialist makes zero LLM calls, see
    color_palette_node's own comment).
    """
    if result.get("error"):
        return result["error"]

    lines = []
    base = result["base_color"]
    if result.get("input_type") == "mood" and result.get("resolved_from_mood"):
        matched = ", ".join(result["resolved_from_mood"])
        lines.append(f"Matched that mood to **{base['name']}** (from: {matched}):")
    else:
        lines.append("**Base color:**")
    lines.append("")
    lines.append(_format_swatch_line(base))
    lines.append("")

    for scheme_name, items in result.get("schemes", {}).items():
        display = _SCHEME_DISPLAY_NAMES.get(scheme_name, scheme_name.title())
        lines.append(f"**{display}:**")
        for item in items:
            lines.append(_format_swatch_line(item))
        lines.append("")

    return "\n".join(lines).strip()


async def _single_shot_fallback(question: str, retrieve_tool, generate_tool) -> str:
    """
    Degraded-but-safe path for multi_hop_node when the decomposition step
    doesn't return parseable JSON: treat the whole question as one
    retrieval instead of raising. A hard crash here would take down the
    whole graph over what is very likely a prompt-following slip by a
    small local model, not a real error condition -- exactly the kind of
    thing Phase 5's failure analysis wants classified as a model failure,
    not a design failure, so the graph itself should survive it.
    """
    raw_chunks = await retrieve_tool.ainvoke({"query": question, "k": 5})
    chunks = unwrap_tool_result(raw_chunks)
    raw_answer = await generate_tool.ainvoke({"query": question, "chunks": chunks})
    return unwrap_tool_result(raw_answer)


# Generic, safe reply used whenever every generate_answer call in a turn
# failed (or the react loop's final message otherwise looks like a raw
# tool-call failure) -- see _extract_grounded_answer below. Never the raw
# exception/validation-error text: per this project's own hard rule (a
# person using the chat should see "this failed, try again", never a
# module path or a pydantic dump -- same reasoning agents/api.py's
# _invoke_turn docstring already gives for the 503 path), this is the
# ONE user-facing string this function is allowed to return when nothing
# usable came back from the tool.
_GENERATE_ANSWER_FAILED_MESSAGE = (
    "Sorry, something went wrong while putting together an answer. "
    "Please try asking again."
)

# Confirmed, not hypothetical: llama3.2 occasionally calls generate_answer
# a SECOND time after already calling it successfully once (despite
# RETRIEVAL_QA_SYSTEM_PROMPT's explicit "character-for-character
# unchanged" instruction telling it not to touch generate_answer's output
# at all after the first call) -- and when it does, it sometimes invents
# a malformed call, e.g. {"k": "5", "q": "..."} instead of the real
# {"query": ..., "chunks": ...} schema (echoing retrieve's own
# k/query-shaped signature instead of generate_answer's). create_react_
# agent's ToolNode catches that failure (a pydantic ValidationError) and
# turns it into a ToolMessage rather than raising -- which means it's
# name="generate_answer" and CONTENT-shaped just like a real answer, so a
# naive "grab the last ToolMessage named generate_answer" (the previous
# version of this function) returns the raw validation-error text as if
# it were the answer. Confirmed live: a person asking about zenithal
# lighting got back a real, correct, cited answer immediately followed by
# "4 validation errors for call[generate_answer] ... Missing required
# argument ...", verbatim, because THAT was the last matching ToolMessage.
#
# These patterns catch that shape (and the handful of other forms a
# swallowed tool-call failure takes across langchain/langgraph/pydantic
# versions) so _extract_grounded_answer can skip a failed call and keep
# looking for an earlier, real one, rather than surfacing raw internals
# to the person on the other end of the chat.
_TOOL_ERROR_PATTERNS = [
    re.compile(r"^\s*\d+\s+validation errors? for\b", re.IGNORECASE),
    re.compile(r"\bmissing required argument\b", re.IGNORECASE),
    re.compile(r"\bunexpected keyword argument\b", re.IGNORECASE),
    re.compile(r"\bpydantic\.errors\b", re.IGNORECASE),
    re.compile(r"errors\.pydantic\.dev", re.IGNORECASE),
    re.compile(r"^\s*Traceback \(most recent call last\)", re.IGNORECASE),
    re.compile(r"^\s*Error:.*Please fix your mistakes\.?\s*$", re.IGNORECASE | re.DOTALL),
]


def _stringify_message_content(content: object) -> str:
    """
    Normalize a message's .content to plain text before this module
    inspects or returns it.

    CONFIRMED ROOT CAUSE (reproduced against this project's own installed
    fastmcp/langgraph/langchain-core versions, not theorized): a
    malformed tool call FastMCP rejects at the schema-validation layer
    (e.g. llama3.2 calling generate_answer with {"k": "5", "query": ...},
    missing "chunks") does NOT raise on the client side at all -- FastMCP
    packages the pydantic validation-error text as an ORDINARY (non-error)
    MCP text-content block, and on this project's langgraph/langchain-core
    versions, create_react_agent's ToolNode leaves a langchain_mcp_adapters
    tool's result as that raw `[{"type": "text", "text": "...", "id":
    "..."}]` list -- it is NOT unwrapped to a plain string automatically,
    contrary to what this module (and mcp_client.py's unwrap_tool_result
    docstring) previously assumed. That is true for a SUCCESSFUL
    generate_answer call too, not just a failed one -- .content is a list
    either way.

    This matters because every caller downstream of a ToolMessage in this
    module used to do `isinstance(content, str)` as its very first check
    (see the previous version of _looks_like_tool_error) and return False
    -- "not an error" -- the moment that failed, since a list obviously
    isn't a str. That meant a malformed generate_answer call's raw
    pydantic dump was NEVER recognized as an error at all; it sailed
    straight through _extract_grounded_answer and got shown to the person
    verbatim, tagged as if it were retrieval_qa's real answer. Confirmed
    with a live repro: a scripted malformed tool call through this exact
    create_react_agent/tools setup produces a ToolMessage whose .content
    is `[{"type": "text", "text": "2 validation errors for call
    [generate_answer]\n...", "id": "..."}]`.

    guardrails.py's output_guard hit the identical shape independently
    (see that file's own _coerce_message_content_to_text docstring, "a
    confirmed live crash... observed on an Arabic-language question") and
    was fixed there; this is the same fix applied at the point this
    failure actually originates, so a malformed call never gets a chance
    to look like a legitimate answer in the first place.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return "" if content is None else str(content)


_HEDGE_PATTERNS = [
    re.compile(r"without (?:additional|more) (?:context|information)", re.IGNORECASE),
    re.compile(r"\bit(?:'s| is) difficult to (?:provide|say|determine|give)\b", re.IGNORECASE),
    re.compile(r"\bi (?:don't|do not) have (?:enough|sufficient) (?:information|context|detail)\b", re.IGNORECASE),
    re.compile(r"\b(?:i'm|i am) unable to (?:determine|provide|say|confirm)\b", re.IGNORECASE),
    re.compile(r"\bcannot (?:provide|give) a definitive\b", re.IGNORECASE),
    re.compile(r"\bhard to (?:say|tell) without\b", re.IGNORECASE),
    re.compile(r"\bunclear without\b", re.IGNORECASE),
    re.compile(r"\bimpossible to (?:say|determine|tell) (?:with|without)\b", re.IGNORECASE),
]


def _looks_like_hedge(text: object) -> bool:
    """
    True if `text` reads like generate_answer's underlying LLM hedging
    ("I don't have enough context to say for sure...") rather than
    actually using the ONE chunk it was given -- the personal-image
    branches in retrieval_qa_node/personal_docs_node/image_qa_node all
    call generate_tool with exactly one chunk: the uploaded image's own
    VLM caption (see _format_personal_image_chunk's own docstring --
    chunk["text"] IS that caption). A hedge here means the caption WAS
    available and DID get passed in, but the generation model still
    produced a vague non-answer instead of grounding on it -- confirmed
    live: an image whose Ollama-VLM caption plainly named its subject
    still got "without additional context... it is difficult to provide
    a definitive explanation" back from generate_answer, verbatim, shown
    to the person as the whole reply.

    Distinct from `_looks_like_tool_error`: that catches a swallowed
    tool-call FAILURE (nothing usable came back at all); this catches a
    tool call that technically SUCCEEDED but produced content strictly
    worse than the caption already sitting right there. Also distinct
    from supervisor.py's own `_looks_like_refusal`, which is deliberately
    narrow (fixed hard-refusal marker strings only -- see that module's
    own docstring for why it can't recognize this "partial or unclear"
    shape). image_qa is listed in supervisor.py's own
    `_DETERMINISTIC_NEVER_HEDGES` on the stated assumption that it "never
    generates open-ended, free-form prose from an LLM's own judgment" --
    true for its corpus-retrieval path, but NOT for this personal-image
    branch, which does call an LLM. This check is what keeps that
    assumption true in practice: a hedge here never becomes the final
    answer, so image_qa still can't SURFACE a hedging answer even though
    one of its branches can technically produce one internally.

    Deliberately pattern-based over a small, hand-picked set of common
    hedge openers rather than exhaustive -- same "cheap to miss, catches
    the common/confirmed case" tradeoff _REFUSAL_MARKERS already accepts
    (see that constant's own docstring) -- widen this list if your own
    model's hedging wording differs.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    return any(p.search(text) for p in _HEDGE_PATTERNS)


def _looks_like_tool_error(content: object) -> bool:
    """
    True if `content` (a ToolMessage's own .content -- str OR the raw
    MCP list-of-content-blocks shape, see _stringify_message_content)
    looks like a swallowed tool-call failure rather than a real
    generate_answer result -- see the module-level docstrings above for
    the confirmed failure this guards against. Deliberately pattern-based
    rather than relying only on ToolMessage.status == "error": that
    attribute isn't populated the same way across every
    langchain-core/langgraph version this project has run against, so the
    content itself is the one signal guaranteed to be there regardless.
    """
    text = _stringify_message_content(content)
    if not text.strip():
        return False
    return any(p.search(text) for p in _TOOL_ERROR_PATTERNS)


def _looks_like_degenerate_repeat(
    text: object, *, min_repeat_len: int = 20, min_repeats: int = 4
) -> bool:
    """
    True if `text` contains the same non-blank line, at least
    `min_repeat_len` characters long, repeated `min_repeats`-or-more
    times CONSECUTIVELY -- the classic small/local-model failure mode
    where generation gets stuck looping the same sentence instead of
    answering, until it hits its own max-token cutoff.

    CONFIRMED live-run failure this guards against, not a hypothetical
    one: painting_lookup_node's own synthesis call (see that function's
    own comment) landed on the local Ollama fallback (see
    llm_provider.py's own docstring on when that happens -- Groq being
    rate-limited/unreachable) for a long, Arabic-language, multi-chunk
    prompt, and degenerated into repeating the exact same
    "[filename] short sentence." block -- copied verbatim from the
    "[filename] chunk text" formatting PAINTING_LOOKUP_SYNTHESIZE_PROMPT_TEMPLATE's
    own corpus_content block uses -- more than 30 times in a row, cut
    off mid-word at the very end. The model didn't answer the question
    at all; it echoed the shape of its own prompt back on a loop.

    Deliberately checks CONSECUTIVE repeats of a meaningfully long,
    whole LINE, not just any repeated short word or phrase -- so normal
    prose that happens to reuse a short connector, or repeat a name
    inside two different sentences, never false-positives. Splitting on
    lines (rather than sentences or fixed-width windows) matches the
    actual observed failure shape exactly: each repeated unit was its
    own line, separated by blank lines, exactly like the
    corpus_content block's own "\n\n"-joined chunks.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    run_line: Optional[str] = None
    run_count = 0
    for line in lines:
        if line == run_line:
            run_count += 1
        else:
            run_line, run_count = line, 1
        if len(line) >= min_repeat_len and run_count >= min_repeats:
            return True
    return False


# Recognizes this project's OWN intended citation marker (see
# local_rag/generation/prompts.py's RAG_SYSTEM_PROMPT/BRANCH_SYSTEM_PROMPT)
# plus the ONE alternate style actually observed in a live run --
# OpenAI's own native browsing-citation glyph, 【N†source】 -- from a model
# that ignored the (at-the-time-softer) prompt instruction and defaulted
# to its own training-data habit instead. See _looks_like_cited_answer's
# own docstring for why detection needs to stay tolerant here even after
# that prompt was tightened.
_ALT_CITATION_RE = re.compile(r"【[^【】]{1,80}†[^【】]{1,80}】")


def _looks_like_cited_answer(text: object) -> bool:
    """
    True if `text` contains this project's own "[source: ...]" citation
    marker, OR the one alternate citation style a live run actually
    produced (see _ALT_CITATION_RE's own comment) -- used by
    retrieval_qa_node to decide whether an answer is a real, grounded,
    generate_answer-produced result worth auto-attaching a related image
    to, as opposed to a greeting reply or a "the corpus doesn't cover
    this" answer, neither of which ever carries a citation at all.

    CONFIRMED live-run failure this closes: generation/prompts.py's
    RAG_SYSTEM_PROMPT originally only gave "[source: ...]" as a soft
    example ("e.g. ..."), and after this project's own switch to Groq's
    openai/gpt-oss models (see local_rag/config.py's own comment), real
    answers came back citing as "some claim【1†page 17】" instead --
    OpenAI's own native browsing-citation format, not this project's own.
    A literal `"[source:" in answer.lower()` check (the ORIGINAL form of
    this check, before this function existed) never matches that, so
    retrieval_qa_node's own auto-image-attach silently never fired for
    any answer in that citation style, even though the answer was
    genuinely grounded and citation-bearing -- indistinguishable, to
    that check, from an uncited greeting reply.

    That prompt has SEPARATELY been tightened to insist on the intended
    format explicitly and to name-and-forbid this exact alternate style
    (see RAG_SYSTEM_PROMPT's own comment) -- this function is the second,
    independent layer: even if a future model swap reintroduces some
    OTHER citation habit neither of these two style checks recognizes,
    the failure mode is "auto-image-attach doesn't fire for that one
    answer" (a missed nice-to-have), never a crash or a wrong answer, so
    staying narrow here (two known, confirmed styles) rather than trying
    to pattern-match "anything that looks vaguely like a citation" is a
    deliberate, low-risk choice -- a broader regex risks matching
    bracketed text that ISN'T a citation at all (e.g. a genuine "[1]"
    footnote-style reference in retrieved chunk text quoted back verbatim
    inside an answer) and firing the image-attach logic on an answer that
    was never actually grounded.
    """
    if not isinstance(text, str) or not text:
        return False
    lowered = text.lower()
    if "[source:" in lowered:
        return True
    return bool(_ALT_CITATION_RE.search(text))


def _extract_grounded_answer(messages: list) -> Optional[str]:
    """
    Return generate_answer's own tool output directly, instead of trusting
    the ReAct loop's final AIMessage to have faithfully carried it.

    This exists because of a confirmed finding, not a hypothetical one:
    tightening RETRIEVAL_QA_SYSTEM_PROMPT with an explicit "your final
    reply must be that returned answer, character-for-character unchanged"
    instruction was tried first and empirically did NOT stop llama3.2 from
    re-narrating generate_answer's cited output in its own words on the
    very next turn -- silently dropping the inline citations
    RETRIEVAL_QA_SYSTEM_PROMPT requires. Traced by inspecting
    result["messages"]: the ToolMessage produced by calling generate_answer
    already contains the citations intact (ToolNode unwraps it for you --
    see mcp_client.py's unwrap_tool_result docstring); it's specifically
    the *next* AIMessage, where the model summarizes that tool result in
    its own words, that loses them. That makes this a design failure
    (per Phase 5's classification), not a prompt failure: no wording fix
    was going to help, because the citations were never at risk in the
    tool call, only in choosing to let the model re-narrate it afterward.

    Walks messages in reverse and returns the (normalized-to-plain-text,
    see _stringify_message_content) content of the LAST generate_answer
    ToolMessage that does NOT look like a swallowed tool-call failure
    (see _looks_like_tool_error) -- not simply the last one, full stop,
    since a stray second/malformed call after a real answer would
    otherwise win. If every generate_answer call in this turn failed, or
    the react loop's own final message is itself error-shaped (e.g.
    generate_answer was never reached at all because an earlier tool call
    in the loop failed), this returns None rather than that raw text --
    these kinds of messages must never reach the person on the other end
    of the chat (see agents/api.py's _invoke_turn for the same rule at
    the HTTP boundary). The caller (retrieval_qa_node) is the one that
    decides what a None means for the person -- normally one more,
    deterministic retrieve+generate attempt (see _single_shot_fallback)
    rather than an immediate apology, since a malformed call is a small
    local model's own tool-calling slip, not evidence the question itself
    is unanswerable. The real exception detail is printed to stderr
    either way, for whoever is running this server to see.

    Falls back to the last AIMessage's content only for the case
    RETRIEVAL_QA_SYSTEM_PROMPT explicitly allows: generate_answer was
    never called at all (e.g. retrieve() found nothing and the agent said
    so directly, in plain text, without ever producing a ToolMessage to
    extract from).
    """
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and msg.name == "generate_answer":
            if _looks_like_tool_error(msg.content):
                print(f"[specialists] retrieval_qa: discarding a failed "
                      f"generate_answer call, not showing it to the user: "
                      f"{_stringify_message_content(msg.content)!r}", file=sys.stderr)
                continue
            return _stringify_message_content(msg.content)

    fallback = _stringify_message_content(messages[-1].content)
    if _looks_like_tool_error(fallback):
        print(f"[specialists] retrieval_qa: final message looked like a "
              f"swallowed tool error, not showing it to the user: "
              f"{fallback!r}", file=sys.stderr)
        return None
    return fallback


# ---------------------------------------------------------------------
# Shared helpers for framing_quote_node
# ---------------------------------------------------------------------
# Same split _parse_color_request above already uses for color_palette:
# light, deterministic triage of the free-text request into structured
# arguments here; the real domain logic (frame-style/glazing/shipping-
# zone resolution, and every dollar figure) happens entirely on the
# TOOL side -- for this specialist specifically, that means across the
# network boundary into System B (framing_agent/, a separate Google ADK
# + FastAPI service -- see get_framing_quote's own docstring in
# mcp_server/server.py and mcp_server/framing_tools.py's module
# docstring for exactly why that boundary is drawn the way it is). This
# module never guesses a dimension, medium, or destination the user
# didn't actually state -- see _parse_framing_request's own "missing"
# list, the same "never invent a number/fact the input didn't provide"
# principle invoice_tools.build_invoice's own "skipped" list already
# applies to a missing price.

_DIMENSION_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(cm|centimeters?|centimetres?|in|inch|inches|\")?"
    r"\s*(?:x|×|by)\s*"
    r"(\d+(?:\.\d+)?)\s*(cm|centimeters?|centimetres?|in|inch|inches|\")?",
    re.IGNORECASE,
)

_CM_UNIT_WORDS = {"cm", "centimeter", "centimeters", "centimetre", "centimetres"}

# Longest/most specific phrases first so e.g. "oil on canvas" is
# captured whole rather than truncated to just "oil" -- same
# "longest match wins" reasoning _parse_color_request's own
# all_scheme_words sort already relies on.
_MEDIUM_KEYWORDS = (
    "oil on canvas", "acrylic on canvas", "watercolor", "watercolour",
    "gouache", "giclee print", "giclée print", "charcoal drawing",
    "ink drawing", "oil painting", "acrylic painting", "print",
    "photograph", "photo", "pastel", "sketch", "drawing", "oil",
    "acrylic", "canvas",
)

# A destination-recognition list on System A's own side -- deliberately
# NOT the same table as framing_agent/pricing.py's own _ZONE_BY_COUNTRY.
# That table lives entirely on System B's side of the network boundary;
# this list only has to recognize that a country NAME is present in the
# text well enough to pass it through as plain text. System B does the
# real shipping-zone lookup once it arrives there -- the two lists are
# allowed to drift (e.g. System B adding a country this list doesn't
# know to look for yet) without breaking anything here, since an
# unmatched destination just means this node asks the user to name one,
# not that a shipping estimate becomes wrong.
#
# EXPANDED after a confirmed live-run failure, not a hypothetical one:
# the original list here had ~20 entries -- mostly the Middle East plus
# a handful of Western countries -- and "japan" wasn't one of them. A
# real request naming Japan explicitly ("...shipped to japan") still
# came back "I'm still missing: a shipping destination country," which
# then cascaded the supervisor through nearly every other specialist
# (retrieval_qa, personal_docs, invoice, corpus_meta, product_search)
# before giving up -- see supervisor.py's own new framing_quote
# early-stop net for the other half of that fix. This list now aims for
# genuinely comprehensive (every commonly-referenced country, not just
# a curated regional sample) specifically so this class of gap doesn't
# recur for the next country nobody thought to add. Deliberately full
# names only, no bare 2-letter ISO codes (a bare "in"/"us"/"is" as a
# substring check would false-positive on a huge fraction of ordinary
# English sentences) -- the few short forms below (uk/usa/uae) are kept
# because they're common enough in casual writing to be worth the small
# residual risk, not because short forms are safe in general.
_DESTINATION_KEYWORDS = (
    # Middle East
    "united arab emirates", "uae", "saudi arabia", "lebanon", "syria",
    "jordan", "iraq", "iran", "israel", "palestine", "yemen", "oman",
    "qatar", "kuwait", "bahrain",
    # North Africa / wider Africa
    "egypt", "morocco", "algeria", "tunisia", "libya", "sudan",
    "nigeria", "kenya", "ethiopia", "ghana", "south africa", "tanzania",
    "uganda", "senegal", "ivory coast", "cameroon", "zimbabwe",
    "zambia", "rwanda", "mozambique",
    # Europe
    "united kingdom", "uk", "ireland", "france", "germany", "italy",
    "spain", "portugal", "netherlands", "belgium", "luxembourg",
    "switzerland", "austria", "sweden", "norway", "denmark", "finland",
    "iceland", "poland", "czech republic", "czechia", "slovakia",
    "hungary", "romania", "bulgaria", "greece", "cyprus", "turkey",
    "ukraine", "russia", "belarus", "estonia", "latvia", "lithuania",
    "croatia", "serbia", "slovenia", "bosnia", "albania", "moldova",
    "malta", "monaco", "montenegro", "north macedonia", "georgia",
    "armenia", "azerbaijan",
    # Americas
    "united states", "usa", "canada", "mexico", "brazil", "argentina",
    "chile", "colombia", "peru", "venezuela", "ecuador", "bolivia",
    "paraguay", "uruguay", "cuba", "jamaica", "haiti", "dominican republic",
    "costa rica", "panama", "guatemala", "honduras", "el salvador",
    "nicaragua", "trinidad", "bahamas", "barbados",
    # Asia
    "china", "japan", "south korea", "north korea", "india", "pakistan",
    "bangladesh", "sri lanka", "nepal", "afghanistan", "kazakhstan",
    "uzbekistan", "mongolia", "vietnam", "thailand", "cambodia", "laos",
    "myanmar", "malaysia", "singapore", "indonesia", "philippines",
    "taiwan", "hong kong", "brunei", "bhutan", "maldives",
    # Oceania
    "australia", "new zealand", "fiji", "papua new guinea",
)


def _extract_dimensions_cm(text: str) -> Optional[tuple[float, float]]:
    """
    (width_cm, height_cm) from the first "NxN"-shaped pair found in the
    text (e.g. "16x20", "40 x 50cm", "24in x 36in"), or None if no such
    pair is present at all -- never guesses a missing pair.

    Unit defaults to INCHES whenever neither number carries an explicit
    unit -- art/framing sizing convention (e.g. "16x20", "24x36") means
    inches far more often than centimeters when nothing is stated. An
    explicit "cm"/"centimeters" on EITHER number overrides that default
    for both (a mismatched pair like "16cm x 20in" is treated as fully
    cm here -- deliberately permissive rather than rejecting an
    ambiguous-but-plausible phrasing outright).
    """
    m = _DIMENSION_RE.search(text)
    if not m:
        return None
    w_raw, w_unit, h_raw, h_unit = m.groups()
    stated_unit = (w_unit or h_unit or "").strip().lower()
    use_cm = stated_unit in _CM_UNIT_WORDS
    width = float(w_raw)
    height = float(h_raw)
    if not use_cm:
        width *= 2.54
        height *= 2.54
    return round(width, 1), round(height, 1)


def _extract_medium(text: str) -> Optional[str]:
    lowered = text.lower()
    for kw in _MEDIUM_KEYWORDS:
        if kw in lowered:
            return kw
    return None


def _extract_destination(text: str) -> Optional[str]:
    lowered = text.lower()
    for kw in _DESTINATION_KEYWORDS:
        if kw in lowered:
            return kw
    return None


def _parse_framing_request(text: str) -> dict:
    """
    {"width_cm", "height_cm", "medium", "destination_country", "missing"}
    -- `missing` lists, in plain English, exactly which of dimensions /
    medium / destination the text didn't provide, so framing_quote_node
    can ask for precisely what's absent instead of a generic "I didn't
    understand." Empty `missing` means every field resolved and the
    request is ready to send to get_framing_quote as-is.
    """
    dims = _extract_dimensions_cm(text)
    medium = _extract_medium(text)
    destination = _extract_destination(text)

    missing = []
    if dims is None:
        missing.append('dimensions (e.g. "16x20 inches" or "40x50 cm")')
    if medium is None:
        missing.append('the medium (e.g. "oil on canvas", "watercolor")')
    if destination is None:
        missing.append("a shipping destination country")

    return {
        "width_cm": dims[0] if dims else None,
        "height_cm": dims[1] if dims else None,
        "medium": medium,
        "destination_country": destination,
        "missing": missing,
    }


async def build_specialists() -> dict[str, Specialist]:
    """
    Build all three Phase 2 specialist node functions against one shared
    MCP client and one shared corpus snapshot.

    Returns a dict keyed by the same route names the supervisor (Phase 3)
    will use as its Literal type and its known-agent-name allowlist:
        {"retrieval_qa": ..., "corpus_meta": ..., "multi_hop": ...}
    Keeping these names identical here and in the supervisor's routing
    schema is what makes the validated-routing check in Phase 3 a simple
    membership test against this dict's keys, rather than a second,
    separately-maintained list that can drift out of sync with this one.
    """
    client = build_client()
    tools_by_name = await load_tools_by_name(client)
    retrieve_tool = tools_by_name["retrieve"]
    generate_tool = tools_by_name["generate_answer"]

    corpus = await fetch_corpus_documents(client)
    document_list = _format_document_list(corpus)

    # temperature=0: every specialist here is doing a grounded-answer or
    # structured-decomposition task, not creative writing -- determinism
    # matters more than variety, and it makes the eval table's "actual
    # route" and "iteration count" columns reproducible between runs.
    #
    # Built as TWO instances now, one per difficulty tier (see this
    # module's own "Model routing by difficulty" section, above
    # build_specialists(), for the full rationale) -- both built once
    # here, at the same one-per-run granularity every LLM-backed node in
    # this project already uses, and picked between PER CALL by
    # `_llm_for()` below rather than one hardcoded instance shared
    # unconditionally by every node the way this used to work. `llm` is
    # kept as a name (aliased to the large tier) only where a call site
    # was already mid-refactor; every node below now calls
    # `_llm_for(question)` instead of referencing `llm` directly.
    # retrieval_qa is the one exception -- see retrieval_qa_agent_large's
    # own comment below for why its react agent always uses the large
    # tier, with no small-tier option at all.
    # get_chat_model (agents/llm_provider.py) -- not a raw ChatOllama --
    # so every specialist's reasoning call below now tries Groq's hosted
    # free tier FIRST and falls back to the exact same local Ollama
    # model/tier this project used before Groq was added, on any Groq
    # failure (missing GROQ_API_KEY, a network error, a rate limit). See
    # llm_provider.py's own module docstring for the full reasoning,
    # including how tool-calling (retrieval_qa_agent_large, just below)
    # survives the fallback unchanged.
    llm_large = get_chat_model("large", node="specialists")
    llm_small = get_chat_model("small", node="specialists")

    def _llm_for(text: Optional[str]):
        """Pick the pre-built chat model for `text`'s difficulty tier --
        see classify_request_difficulty's own docstring for the
        heuristic. Never builds a new model per call; only ever returns
        one of the two instances already built above."""
        return llm_large if classify_request_difficulty(text) == "complex" else llm_small

    # --- Specialist 1: retrieval-QA -----------------------------------
    # ALWAYS the large agent now -- retrieval_qa is no longer tiered by
    # difficulty at all, unlike every other reasoning call site in this
    # project. See retrieval_qa_agent_large's own comment just below for
    # the confirmed live-run crash this fixes.
    retrieval_qa_agent_large = create_react_agent(
        llm_large,
        tools=[retrieve_tool, generate_tool],
        prompt=RETRIEVAL_QA_SYSTEM_PROMPT,
    )
    # CONFIRMED live-run crash this closes: `create_react_agent` binds
    # its tools at build time (`tools=[retrieve_tool, generate_tool]`),
    # which means the underlying Ollama chat request always carries a
    # `tools` payload -- and _SMALL_REASONING_MODEL (phi3) does not
    # support that Ollama API parameter at all. A "simple"-classified
    # question (classify_request_difficulty's own heuristic -- short,
    # no complex-keyword, single "?") routed to retrieval_qa reliably
    # 503'd end to end with `ResponseError('registry.ollama.ai/library/
    # phi3:latest does not support tools')`, since phi3 is the ONLY
    # model currently configured for the small tier
    # (_SMALL_REASONING_MODEL, specialists.py) and there's no
    # tool-capable alternative in that tier to fall back to.
    #
    # This is categorically different from every OTHER difficulty-tiered
    # call site in this project (`_llm_for()` above, and every specialist
    # that calls it for a plain `.ainvoke()`): none of those bind tools
    # at all -- they're ordinary chat completions, sometimes constrained
    # to a JSON output shape via Ollama's `format=<schema>` parameter
    # (see supervisor.py's own DEFAULT_ROUTE_FORMAT docstring), which is
    # a different Ollama API parameter phi3 handles fine. Tool-calling
    # specifically is what phi3 can't do -- so retrieval_qa's react
    # agent, which genuinely cannot function without it, has no small
    # tier to offer at all; the cost/latency savings the small tier
    # exists for (see this module's "Model routing by difficulty"
    # section) simply aren't available here without configuring a
    # different, tool-capable small model in OLLAMA_GENERATION_MODELS
    # (local_rag/config.py) to replace phi3 in that role.
    #
    # A `retrieval_qa_agent_small` bound to `llm_small` used to exist
    # here, picked via a `_agent_for()` helper mirroring `_llm_for()`
    # above -- removed rather than kept as an unused dead code path,
    # since keeping it around invites someone re-wiring it back in later
    # without rediscovering why it was pulled in the first place.

    # Looked up here (not just below, where personal_docs_node needs them
    # too) so retrieval_qa_node -- defined next -- can close over them for
    # its own personal-image short-circuit. See that node's own comment
    # for why it needs this at all.
    search_personal_tool = tools_by_name.get("search_personal_documents")
    latest_personal_image_tool = tools_by_name.get("latest_personal_image")

    async def retrieval_qa_node(state: AgentState) -> dict:
        question = _last_human_text(state)
        thread_id = state.get("thread_id")

        # THE SAME structural short-circuit personal_docs_node and
        # image_qa_node already apply (see _best_personal_image_result's
        # own docstring: "called from both... so that guarantee holds
        # regardless of which of the two the supervisor's own routing
        # decision happens to pick") -- extended here to cover the THIRD
        # possible routing target that docstring didn't originally
        # account for. Confirmed live, not hypothetical: a follow-up like
        # "explain this to me" right after uploading an image got routed
        # to retrieval_qa, not personal_docs -- retrieval_qa has no
        # concept of "this thread's own upload" at all, so its react
        # agent tried to search the main CORPUS for a pronoun-only
        # question with nothing corpus-searchable in it, found nothing to
        # ground an answer in, and -- per a confirmed, separate small-
        # model reliability issue -- sometimes responded by calling
        # generate_answer a second time with malformed arguments (see
        # _looks_like_tool_error's own docstring) while trying to recover
        # from the empty retrieval. Checking for a personal-image match
        # FIRST means a misrouted "explain this"/"what is this" question
        # never reaches the react agent at all in that case, which closes
        # off that failure mode structurally rather than hoping the
        # supervisor's prompt-based routing rule holds every time.
        image_hit = await _best_personal_image_result(
            search_personal_tool, latest_personal_image_tool, thread_id, question
        )
        if image_hit is not None:
            image_block = _personal_image_display_block(image_hit)
            raw_image_answer = await generate_tool.ainvoke({"query": question, "chunks": [image_hit]})
            image_answer = unwrap_tool_result(raw_image_answer)
            if (
                not isinstance(image_answer, str)
                or not image_answer.strip()
                or _looks_like_tool_error(image_answer)
                or _looks_like_hedge(image_answer)
            ):
                # Covers the same malformed-tool-call shape
                # _extract_grounded_answer guards against (see its own
                # docstring) -- this call's args are always well-formed
                # (built here, not by the LLM), but defense in depth costs
                # nothing: never let a raw pydantic/tool-error dump stand
                # in for a real caption-grounded answer.
                content = _format_personal_image_chunk(image_hit)
            else:
                content = _image_answer_content(image_block, image_answer, question)
            return {"messages": [AIMessage(content=content, name="retrieval_qa")]}

        result = await retrieval_qa_agent_large.ainvoke(
            {"messages": [HumanMessage(content=question)]}
        )
        answer = _extract_grounded_answer(result["messages"])
        if answer is None:
            # Every generate_answer call the react loop made this turn was
            # error-shaped (see _extract_grounded_answer's own docstring
            # for the confirmed malformed-call failure this covers) --
            # recover with one deterministic retrieve+generate instead of
            # going straight to an apology: the question itself is very
            # likely still answerable, a small local model just issued a
            # bad tool call while trying to answer it. Same fallback shape
            # multi_hop_node's own decomposition-parse-failure already
            # uses (see _single_shot_fallback).
            try:
                answer = await _single_shot_fallback(question, retrieve_tool, generate_tool)
            except Exception as e:
                print(f"[specialists] retrieval_qa: single-shot fallback "
                      f"also failed: {e!r}", file=sys.stderr)
                answer = _GENERATE_ANSWER_FAILED_MESSAGE

        # Auto-attach a related corpus image, when a genuinely strong
        # match exists, to a real grounded content answer -- never to
        # the greeting fast-path above (RETRIEVAL_QA_SYSTEM_PROMPT
        # explicitly forbids a `[source:` citation on that reply) and
        # never to a "the corpus doesn't cover this" answer (that path
        # never calls generate_answer either, so it carries no citation
        # either -- see _extract_grounded_answer's own docstring). The
        # presence of a `[source:` citation is exactly how this tells
        # "real answer, worth illustrating" apart from either of those,
        # the same signal SUPERVISOR_SYSTEM_PROMPT's own greeting-FINISH
        # rule relies on (see prompts.py). retrieve_images_embedded_tool/
        # retrieve_images_tool and their formatters are defined further
        # down in this same build_specialists() call, in the image_qa
        # section below -- safe to reference here anyway, since Python
        # closures resolve a name against the enclosing scope at CALL
        # time, and this node is never actually invoked until well after
        # build_specialists() has finished assigning them.
        if isinstance(answer, str) and _looks_like_cited_answer(answer):
            image_tool = retrieve_images_embedded_tool or retrieve_images_tool
            if image_tool is not None:
                try:
                    raw_img = await image_tool.ainvoke({"query": question, "k": 1})
                    img_hits = unwrap_tool_result(raw_img)
                except Exception as e:  # noqa: BLE001 -- an image lookup failing must never break the text answer already in hand
                    print(f"[specialists] retrieval_qa: auto-image lookup failed, "
                          f"showing the text answer alone: {e!r}", file=sys.stderr)
                    img_hits = None
                if isinstance(img_hits, list) and img_hits and isinstance(img_hits[0], dict):
                    top = img_hits[0]
                    score = top.get("score", 0.0)
                    print(f"[specialists] retrieval_qa: auto-image top score={score!r} "
                          f"(threshold={_AUTO_IMAGE_MIN_SCORE}) for query {question!r}",
                          file=sys.stderr)
                    if isinstance(score, (int, float)) and score >= _AUTO_IMAGE_MIN_SCORE:
                        formatter = (
                            _format_image_result_embedded
                            if image_tool is retrieve_images_embedded_tool
                            else _format_image_result
                        )
                        answer = f"{formatter(top)}\n\n{answer}"

        return {"messages": [AIMessage(content=answer, name="retrieval_qa")]}

    # --- Specialist 1b: personal-docs ------------------------------------
    # Explicit retrieve-then-generate, same shape as multi_hop/painting_
    # lookup below -- not a react agent, since the flow is always exactly
    # "search THIS thread's own uploads once, generate once," never a
    # variable number of tool calls. Reuses generate_tool unchanged (the
    # same one retrieval_qa/multi_hop already use) -- it's generic over
    # any chunks list (see mcp_server/server.py's generate_answer
    # docstring), so no second generation prompt is needed just because
    # these chunks came from local_rag/personal_rag.py's "temp"
    # collection instead of the main corpus.
    #
    # thread_id comes from state["thread_id"] (see state.py's own
    # docstring for that field) -- NEVER parsed out of the question text
    # -- so this can only ever search the conversation it's actually
    # running in. A missing thread_id (state.py's default None, e.g. a
    # bare ask()/CLI call with no persisted thread at all) degrades the
    # same way a missing tool does: nothing to search, say so plainly,
    # never raise.

    async def personal_docs_node(state: AgentState) -> dict:
        question = _last_human_text(state)
        thread_id = state.get("thread_id")

        if search_personal_tool is None or not thread_id:
            return {"messages": [AIMessage(content=PERSONAL_DOCS_NO_RESULTS_MESSAGE, name="personal_docs")]}

        # Structural preference, checked FIRST: if the best-matching
        # thing this thread ever uploaded is an IMAGE, show that actual
        # image (never a corpus image that merely resembles it -- see
        # _best_personal_image_result's own docstring), paired with a
        # REAL generated answer to whatever was actually asked --
        # grounded in nothing but this image's own VLM caption (the
        # same anti-hallucination "only this chunk, nothing else" shape
        # generate_tool is already called with everywhere else in this
        # node), so the picture can never be swapped for a wrong one AND
        # a specific question ("explain this image", "what's the title
        # say") gets a real answer instead of the caption echoed back
        # unchanged. A confirmed live report showed why the unchanged
        # echo reads badly: getting back only the exact same caption
        # already shown once at upload time, verbatim, on a genuine
        # follow-up question, feels like the question was ignored.
        image_hit = await _best_personal_image_result(
            search_personal_tool, latest_personal_image_tool, thread_id, question
        )
        if image_hit is not None:
            image_block = _personal_image_display_block(image_hit)
            raw_image_answer = await generate_tool.ainvoke({"query": question, "chunks": [image_hit]})
            image_answer = unwrap_tool_result(raw_image_answer)
            if (
                not isinstance(image_answer, str)
                or not image_answer.strip()
                or _looks_like_tool_error(image_answer)
                or _looks_like_hedge(image_answer)
            ):
                # generate_tool degraded to something unusable (see its
                # own contract), or -- same defense-in-depth as the other
                # two image branches in this file, see their own comments
                # -- came back error-shaped. Fall back to the caption
                # itself rather than combining with an empty/broken
                # answer, or showing a raw tool-error dump.
                content = _format_personal_image_chunk(image_hit)
            else:
                content = _image_answer_content(image_block, image_answer, question)
            return {"messages": [AIMessage(content=content, name="personal_docs")]}

        raw_chunks = await search_personal_tool.ainvoke(
            {"thread_id": thread_id, "query": question, "k": 5}
        )
        chunks = unwrap_tool_result(raw_chunks) or []
        if not chunks:
            return {"messages": [AIMessage(content=PERSONAL_DOCS_NO_RESULTS_MESSAGE, name="personal_docs")]}

        raw_answer = await generate_tool.ainvoke({"query": question, "chunks": chunks})
        answer = unwrap_tool_result(raw_answer)
        return {"messages": [AIMessage(content=answer, name="personal_docs")]}

    # --- Specialist 2: corpus-meta -------------------------------------
    # No tools bound at all -- the document list is baked into the system
    # prompt exactly once, at build time, not re-fetched per call. That's
    # deliberate: this specialist answering from a snapshot rather than a
    # live lookup is an acceptable tradeoff for "what's in the corpus"
    # questions (unlike retrieve()'s BM25 snapshot, staleness here just
    # means a newly-ingested document is briefly invisible to this one
    # specialist, not a wrong answer about existing content).
    corpus_meta_system = CORPUS_META_SYSTEM_PROMPT.format(document_list=document_list)

    async def corpus_meta_node(state: AgentState) -> dict:
        question = _last_human_text(state)
        response = await _llm_for(question).ainvoke(
            [SystemMessage(content=corpus_meta_system), HumanMessage(content=question)]
        )
        return {"messages": [AIMessage(content=response.content, name="corpus_meta")]}

    # --- Specialist 3: multi-hop ---------------------------------------
    async def multi_hop_node(state: AgentState) -> dict:
        question = _last_human_text(state)

        # Step 1: decompose into exactly two sub-questions. A genuine
        # multi-hop question almost always trips _COMPLEX_REQUEST_KEYWORDS
        # on its own (this specialist only ever gets routed compound
        # "compare"/"both" style questions in the first place), so this
        # is expected to land on the LARGE tier in practice -- kept as a
        # call to _llm_for() rather than hardcoded to llm_large anyway,
        # for the same "one rule, everywhere" reason every other call
        # site in this function does the same.
        decompose_resp = await _llm_for(question).ainvoke(
            [
                SystemMessage(content=MULTI_HOP_DECOMPOSE_SYSTEM_PROMPT),
                HumanMessage(content=question),
            ]
        )
        try:
            sub_qs = json.loads(decompose_resp.content)
            sub_q1, sub_q2 = sub_qs["sub_query_1"], sub_qs["sub_query_2"]
        except (json.JSONDecodeError, KeyError, TypeError):
            # See _single_shot_fallback's docstring: a parse failure here
            # degrades to one retrieval instead of crashing the graph.
            answer = await _single_shot_fallback(question, retrieve_tool, generate_tool)
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "[multi-hop decomposition did not return valid JSON; "
                            "fell back to a single retrieval]\n\n" + answer
                        ),
                        name="multi_hop",
                    )
                ]
            }

        # Step 2: retrieve twice, explicitly -- two real, countable tool
        # calls, never left to an agent looping an unknown number of times.
        raw_chunks_1 = await retrieve_tool.ainvoke({"query": sub_q1, "k": 5})
        raw_chunks_2 = await retrieve_tool.ainvoke({"query": sub_q2, "k": 5})
        chunks_1 = unwrap_tool_result(raw_chunks_1)
        chunks_2 = unwrap_tool_result(raw_chunks_2)

        # Step 3: synthesize once, over both chunk sets combined. Passed
        # as the "query" to generate_answer so the underlying RAG prompt
        # template sees the full synthesis instructions, not just the
        # bare original question.
        synth_prompt = MULTI_HOP_SYNTHESIZE_PROMPT_TEMPLATE.format(
            question=question, sub_q1=sub_q1, sub_q2=sub_q2
        )
        raw_answer = await generate_tool.ainvoke(
            {"query": synth_prompt, "chunks": chunks_1 + chunks_2}
        )
        answer = unwrap_tool_result(raw_answer)

        return {"messages": [AIMessage(content=answer, name="multi_hop")]}

    # --- Specialist 4: image-qa -----------------------------------------
    # Zero LLM calls, by design -- see prompts.py's note above this
    # section. retrieve_images() already returns a caption per image (see
    # mcp_server/image_tools.py), so this node's only job is deterministic
    # markdown formatting of the tool's own output, the same
    # "structurally cannot hallucinate" property corpus_meta gets from
    # having no tools at all, applied here to having no LLM at all.
    retrieve_images_tool = tools_by_name.get("retrieve_images")

    # New, additive: mcp_server/server.py's retrieve_images_embedded tool
    # (mcp_server/image_tools.py's retrieve_images_with_data) returns each
    # image's actual bytes as base64 instead of a server-local path, so
    # this node can embed images directly with no `/images/{filename}`
    # static-file route needed at all. `.get()` returns None against an
    # older server.py that predates this tool, so the branch below falls
    # straight back to the original retrieve_images_tool path -- byte-for-
    # byte the same behavior as before this was added -- rather than
    # erroring on a missing tool.
    retrieve_images_embedded_tool = tools_by_name.get("retrieve_images_embedded")

    # Image-to-image "find corpus images similar to my upload" tools --
    # see mcp_server/server.py's own find_similar_images/
    # find_similar_images_embedded docstrings. `.get()` returns None
    # against an older server.py that predates these, same graceful
    # degrade retrieve_images_embedded_tool already follows for its own
    # newer-tool case.
    find_similar_images_tool = tools_by_name.get("find_similar_images")
    find_similar_images_embedded_tool = tools_by_name.get("find_similar_images_embedded")

    def _format_image_result(item: dict) -> str:
        # Deliberately duplicated from mcp_server/image_tools.py's
        # format_markdown_image rather than imported -- agents/ talks to
        # the pipeline exclusively through the MCP server (see this
        # module's own docstring), never by importing mcp_server/'s
        # internals directly, and this is a two-line formatting rule, not
        # logic worth crossing that boundary for.
        #
        # image_path, as returned by retrieve_images(), is a path on the
        # server's own local disk (e.g. wherever ingestion/ingest_pdf.py
        # wrote extracted images) -- a browser, on a different origin or
        # even just a different process, can never load that path
        # directly. `os.path.basename` reduces it to just the filename,
        # and agents/api.py's `GET /images/{filename}` endpoint serves
        # that same file back out over HTTP from IMAGE_DIR. Emitted as a
        # *relative* URL (not a full host:port) so it resolves correctly
        # against whichever origin actually served this markdown:
        # same-origin automatically for agents/static/chat.html (served
        # by this same FastAPI app), and rewritten onto the API's own
        # origin by the React frontend's markdown image component (see
        # frontend/src/components/MarkdownText.tsx), since that app is
        # served from Vite's separate origin instead.
        caption = _escape_markdown_caption(item.get("caption")) or "(no caption available)"
        path = item.get("image_path")
        if not path:
            return f"*(image unavailable — no path returned)*\n*{caption}*"
        image_url = f"/images/{os.path.basename(path)}"
        return f"![{caption}]({image_url})\n*{caption}*"

    def _format_image_result_embedded(item: dict) -> str:
        # Sibling to _format_image_result() above, for
        # retrieve_images_embedded() results: embeds the image bytes as
        # a `data:` URI instead of a `/images/{filename}` path. Falls
        # back to _format_image_result()'s own (unchanged) path-based
        # rendering when this particular item has no data_uri -- e.g.
        # mcp_server/image_tools.py's size cap or a missing file -- so a
        # partially-embeddable result set still renders every item
        # instead of a broken image tag for the ones that couldn't be
        # encoded.
        #
        # frontend/src/components/MarkdownText.tsx's RemoteImage only
        # rewrites an <img> src that starts with "/" onto the API's own
        # origin; a "data:image/...;base64,..." src never starts with
        # "/", so it passes straight through unrewritten and just
        # renders -- no frontend change needed for this to work in the
        # React UI, and it displays the same way in
        # agents/static/chat.html (same-origin) too.
        caption = _escape_markdown_caption(item.get("caption")) or "(no caption available)"
        data_uri = item.get("data_uri")
        if not data_uri:
            return _format_image_result(item)
        return f"![{caption}]({data_uri})\n*{caption}*"

    async def image_qa_node(state: AgentState) -> dict:
        question = _last_human_text(state)
        thread_id = state.get("thread_id")

        # "find images similar to what I uploaded" -- checked FIRST,
        # before the personal-upload short-circuit below, since that
        # branch's whole job is showing the person's OWN upload back;
        # this one instead searches the CORPUS for images that visually
        # resemble it, a different request entirely. Gated on thread_id
        # (nothing to compare against without one) and on
        # _looks_like_similar_image_request (see its own docstring) so
        # an ordinary "what is this?" about the same upload still falls
        # through to the existing behavior below, unchanged.
        if thread_id and _looks_like_similar_image_request(question):
            similar_tool = find_similar_images_embedded_tool or find_similar_images_tool
            if similar_tool is not None:
                try:
                    raw_similar = await similar_tool.ainvoke({"thread_id": thread_id, "k": 3})
                    similar_hits = unwrap_tool_result(raw_similar)
                except Exception as e:  # noqa: BLE001 -- a failed lookup degrades to the same "nothing found" message below, never breaks the turn
                    print(f"[specialists] image_qa: find_similar_images lookup failed: "
                          f"{e!r}", file=sys.stderr)
                    similar_hits = None
                valid_hits = (
                    [h for h in similar_hits if isinstance(h, dict)]
                    if isinstance(similar_hits, list)
                    else []
                )
                if valid_hits:
                    formatter = (
                        _format_image_result_embedded
                        if similar_tool is find_similar_images_embedded_tool
                        else _format_image_result
                    )
                    blocks = [formatter(item) for item in valid_hits]
                    answer = (
                        "Here's what I found in the corpus that visually resembles "
                        "the image you uploaded:\n\n" + "\n\n".join(blocks)
                    )
                    return {"messages": [AIMessage(content=answer, name="image_qa")]}
                # The tool exists and was actually tried -- this specific
                # request is fully handled either way, so return here
                # rather than falling through to the unrelated "show the
                # upload back" or generic text-query branches below,
                # which would silently answer a different question than
                # the one actually asked.
                return {
                    "messages": [
                        AIMessage(
                            content=(
                                "I couldn't find anything in the corpus that visually "
                                "resembles the image you uploaded -- either you haven't "
                                "uploaded an image into this conversation yet, nothing has "
                                "been ingested into the image collection with multimodal "
                                "support enabled, or nothing matched closely enough."
                            ),
                            name="image_qa",
                        )
                    ]
                }
            # similar_tool is None (older server.py without these two
            # tools) -- fall through to this node's existing behavior
            # below rather than blocking on a capability that doesn't
            # exist here.

        # Structural preference, checked FIRST, before ANY corpus-wide
        # retrieval: if this thread personally uploaded an image that's
        # relevant to the question, show THAT image -- never a corpus
        # image retrieve_images/retrieve_images_embedded merely judges
        # similar. See _best_personal_image_result's own docstring for
        # the confirmed live-run misroute (a personal-image follow-up
        # landing on image_qa instead of personal_docs) this guards
        # against regardless of which specialist the supervisor picked.
        # thread_id is None for callers with no persisted thread at all
        # (graph.py's own ask(), the CLI, the Phase 5 eval script) --
        # _best_personal_image_result already degrades to None for that
        # case, so this is a no-op there, exactly like personal_docs_node
        # already behaves for the same callers.
        #
        # Paired with a real generate_tool answer (same reasoning and
        # same fallback shape as personal_docs_node's own image_hit
        # branch -- see that one's comment) rather than just the caption
        # echoed back unchanged, so a specific question about this
        # thread's own image gets an actual answer here too, not just
        # in personal_docs_node.
        personal_hit = await _best_personal_image_result(
            search_personal_tool, latest_personal_image_tool, thread_id, question
        )
        if personal_hit is not None:
            image_block = _personal_image_display_block(personal_hit)
            raw_image_answer = await generate_tool.ainvoke({"query": question, "chunks": [personal_hit]})
            image_answer = unwrap_tool_result(raw_image_answer)
            if (
                not isinstance(image_answer, str)
                or not image_answer.strip()
                or _looks_like_tool_error(image_answer)
                or _looks_like_hedge(image_answer)
            ):
                # Covers the same malformed-tool-call shape
                # _extract_grounded_answer guards against (see its own
                # docstring) -- this call's args are always well-formed
                # (built here, not by the LLM), but defense in depth costs
                # nothing: never let a raw pydantic/tool-error dump stand
                # in for a real caption-grounded answer.
                content = _format_personal_image_chunk(personal_hit)
            else:
                content = _image_answer_content(image_block, image_answer, question)
            return {"messages": [AIMessage(content=content, name="image_qa")]}

        if retrieve_images_embedded_tool is not None:
            raw = await retrieve_images_embedded_tool.ainvoke({"query": question, "k": 3})
            results = unwrap_tool_result(raw)
            if not results:
                return {"messages": [AIMessage(content=IMAGE_QA_NO_RESULTS_MESSAGE, name="image_qa")]}
            blocks = [_format_image_result_embedded(item) for item in results]
            answer = "\n\n".join(blocks)
            return {"messages": [AIMessage(content=answer, name="image_qa")]}

        if retrieve_images_tool is None:
            # Server wasn't built with either image tool (e.g. an older
            # server.py) -- degrade to the same "say plainly" message a
            # genuinely empty result set would produce, rather than
            # raising KeyError on a missing dict entry.
            return {"messages": [AIMessage(content=IMAGE_QA_NO_RESULTS_MESSAGE, name="image_qa")]}

        raw = await retrieve_images_tool.ainvoke({"query": question, "k": 3})
        results = unwrap_tool_result(raw)
        if not results:
            return {"messages": [AIMessage(content=IMAGE_QA_NO_RESULTS_MESSAGE, name="image_qa")]}

        blocks = [_format_image_result(item) for item in results]
        answer = "\n\n".join(blocks)
        return {"messages": [AIMessage(content=answer, name="image_qa")]}

    # --- Specialist 5: painting-lookup -----------------------------------
    # Not a create_react_agent, for the same reason multi_hop isn't one:
    # this specialist's two source calls (one retrieve() against the
    # corpus, one search_painting_online() against the internet) always
    # happen exactly once each, in explicit Python, so its iteration count
    # is fixed and knowable in advance -- never however many times an
    # agent decides to loop between two tools that don't depend on each
    # other's output.
    search_painting_tool = tools_by_name.get("search_painting_online")

    def _plain_painting_lookup_fallback(question: str, corpus_content: str, web_summary: str) -> str:
        """
        Last-resort answer for painting_lookup_node -- only reached when
        the LLM synthesis step degenerated TWICE in a row (see
        _looks_like_degenerate_repeat's own docstring for the confirmed
        failure this exists for). Never shows the user the broken,
        repeated output, but also never silently returns nothing: plain
        string formatting directly over what was actually retrieved,
        same "the data owns the content, not a possibly-broken
        generation" preference this project applies elsewhere (e.g.
        framing_agent/server.py's own _template_explanation on System
        B's side of the network boundary).
        """
        parts = [
            f'I found information about "{question}", but had trouble summarizing it '
            "cleanly this time. Here's what was actually retrieved, unedited:"
        ]
        if corpus_content and corpus_content != "(nothing found in the corpus)":
            parts.append(f"**From the corpus:**\n{corpus_content}")
        if web_summary and web_summary != "(nothing found on the internet)":
            parts.append(f"**From the web:**\n{web_summary}")
        if len(parts) == 1:
            parts.append("Unfortunately, nothing was found in the corpus or online for this painting.")
        return "\n\n".join(parts)

    async def painting_lookup_node(state: AgentState) -> dict:
        question = _last_human_text(state)

        raw_chunks = await retrieve_tool.ainvoke({"query": question, "k": 5})
        chunks = unwrap_tool_result(raw_chunks) or []
        corpus_content = (
            "\n\n".join(f"[{c.get('metadata', {}).get('filename', '?')}] {c.get('text', '')}" for c in chunks)
            if chunks
            else "(nothing found in the corpus)"
        )

        web_result = {"summary": None, "sources": []}
        if search_painting_tool is not None:
            raw_web = await search_painting_tool.ainvoke({"painting_name": question})
            web_result = unwrap_tool_result(raw_web) or web_result

        # CONFIRMED live-run bug this closes: search_famous_painting
        # (mcp_server/web_tools.py) deliberately returns summary=None
        # alongside a non-empty sources list when Wikipedia's own summary
        # lookup came up empty but the general web search still found
        # real, allowlisted sources -- that function's OWN comment says
        # "caller decides how to phrase 'no summary, but see sources'."
        # This is that phrasing, finally implemented: a flat "(nothing
        # found on the internet)" whenever summary was falsy, regardless
        # of sources, told the synthesis LLM there was nothing online at
        # all even when three real Wikipedia/Britannica links were about
        # to be appended right below its own answer -- producing exactly
        # that contradiction in a live run ("no summary was found
        # online," followed immediately by three internet sources).
        sources = web_result.get("sources") or []
        if web_result.get("summary"):
            web_summary = web_result["summary"]
        elif sources:
            web_summary = (
                "(no clean summary could be extracted, but the internet search did find "
                "real sources on this -- see the source list, and feel free to say a "
                "summary wasn't available while still naming what the sources appear to be about "
                "based on their titles)"
            )
        else:
            web_summary = "(nothing found on the internet)"

        synth_prompt = PAINTING_LOOKUP_SYNTHESIZE_PROMPT_TEMPLATE.format(
            painting_name=question, corpus_content=corpus_content, web_summary=web_summary
        )
        synth_messages = [SystemMessage(content=synth_prompt), HumanMessage(content=question)]
        response = await _llm_for(question).ainvoke(synth_messages)
        answer = response.content

        if _looks_like_degenerate_repeat(answer):
            # CONFIRMED live-run failure -- see _looks_like_degenerate_repeat's
            # own docstring for the exact incident (a local-Ollama-fallback
            # synthesis call looping the same "[filename] sentence." block
            # 30+ times on an Arabic, multi-chunk prompt). Retried ONCE,
            # explicitly on llm_large -- deliberately bypassing _llm_for's
            # own tier pick, since the point is to get a DIFFERENT model
            # than whatever just degenerated: llm_large tries Groq first
            # (see llm_provider.py), a completely different, much stronger
            # model, before it would ever itself fall back to the same
            # local model that just failed.
            retry_response = await llm_large.ainvoke(synth_messages)
            retry_answer = retry_response.content
            if (
                isinstance(retry_answer, str)
                and retry_answer.strip()
                and not _looks_like_degenerate_repeat(retry_answer)
            ):
                answer = retry_answer
            else:
                # Both attempts degenerated (rare -- would mean the retry
                # landed on a broken model too, or Groq is unreachable and
                # fell back to the same local model again). Never show the
                # user a wall of repeated text -- degrade to a plain,
                # unsynthesized answer built directly from what was
                # actually retrieved instead.
                answer = _plain_painting_lookup_fallback(question, corpus_content, web_summary)

        if sources:
            source_lines = "\n".join(f"- [{s['title']}]({s['url']})" for s in sources if s.get("url"))
            answer = f"{answer}\n\n**Sources:**\n{source_lines}"

        return {"messages": [AIMessage(content=answer, name="painting_lookup")]}

    # --- Specialist 6: product-search -------------------------------------
    # One tool call (search_art_supplies), then Python -- not the LLM --
    # tags each candidate's tier and picks the top candidates PER TIER;
    # the LLM's single call is confined to writing tier-scoped comparison
    # paragraphs (PRODUCT_SEARCH_SYSTEM_PROMPT is explicit that it must
    # not invent a price or link, or cross-compare between tiers). Every
    # price/link actually shown to the user is rendered directly from the
    # tool's own output below, never from the model's text.
    #
    # Splits results into "beginner-friendly" and "professional-grade"
    # buckets, 5 of each where available (see _classify_tier /
    # _tier_candidates / _pick_top above) -- fetches a bigger raw pool
    # (max_results=12, up from the original single-list's 8) so there's
    # realistically enough candidates to fill both tiers rather than
    # starving one of them.
    search_supplies_tool = tools_by_name.get("search_art_supplies")

    async def product_search_node(state: AgentState) -> dict:
        question = _last_human_text(state)

        if search_supplies_tool is None:
            return {
                "messages": [
                    AIMessage(
                        content=_embed_product_data(
                            "I don't have a working internet-search tool available right now, "
                            "so I can't look up art supplies. Please try again once the "
                            "product-search tool is configured.",
                            [],
                        ),
                        name="product_search",
                    )
                ]
            }

        raw = await search_supplies_tool.ainvoke({"query": question, "max_results": 12})
        candidates = unwrap_tool_result(raw) or []

        if not candidates:
            return {
                "messages": [
                    AIMessage(
                        content=_embed_product_data(
                            "I searched the internet but couldn't find any art-supply listings "
                            "for that right now (the search backend may be unavailable). "
                            "I'm not going to guess at products or prices.",
                            [],
                        ),
                        name="product_search",
                    )
                ]
            }

        # Tag every candidate's tier. "unclassified" ones (no keyword cue
        # either way) are resolved against the pool's own median KNOWN
        # price -- cheaper-than-median -> beginner, at-or-above -> pro.
        # A candidate with no price AND no keyword cue defaults to
        # "beginner" (documented default, not a silent guess): a totally
        # unlabeled, unpriced listing is, if anything, more likely to be
        # a generic starter item than a specialty professional one.
        for item in candidates:
            item["tier"] = _classify_tier(item)

        known_prices = [c["price"] for c in candidates if c.get("price") is not None]
        median_price = statistics.median(known_prices) if known_prices else None
        for item in candidates:
            if item["tier"] != "unclassified":
                continue
            if median_price is not None and item.get("price") is not None:
                item["tier"] = "beginner" if item["price"] <= median_price else "professional"
            else:
                item["tier"] = "beginner"

        beginner_pool, professional_pool = _tier_candidates(candidates)
        beginner_picks = _pick_top(beginner_pool, 5)
        professional_picks = _pick_top(professional_pool, 5)

        for i, item in enumerate(beginner_picks, start=1):
            item["id"] = f"b{i}"
        for i, item in enumerate(professional_picks, start=1):
            item["id"] = f"g{i}"

        def _format_tier_listing(label: str, items: list[dict]) -> str:
            if not items:
                return f"{label}: (none found in this search)"
            lines = [f"{label}:"]
            for it in items:
                price_text = (
                    f"${it['price']:.2f}" if it.get("price") is not None else "price not found in snippet"
                )
                lines.append(
                    f"  - id={it['id']} | {it['title']} | {it['source']} | {price_text} "
                    f"| snippet: {it.get('snippet', '')[:200]}"
                )
            return "\n".join(lines)

        listing_text = (
            _format_tier_listing("BEGINNER-FRIENDLY CANDIDATES", beginner_picks)
            + "\n\n"
            + _format_tier_listing("PROFESSIONAL-GRADE CANDIDATES", professional_picks)
        )
        system_prompt = PRODUCT_SEARCH_SYSTEM_PROMPT.format(product_listing=listing_text)
        response = await _llm_for(question).ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=question)]
        )
        comparison_text = response.content

        def _render_tier_section(header: str, items: list[dict]) -> list[str]:
            lines = [header]
            if not items:
                lines.append("_None found for this search._")
                return lines
            for it in items:
                price_str = f"${it['price']:.2f}" if it.get("price") is not None else "price not listed"
                lines.append(
                    f"{it['id']}. **{it['title']}** — {price_str} — via {it['source']} — "
                    f"[View listing]({it['url']})"
                )
            return lines

        lines = [comparison_text, ""]
        lines += _render_tier_section("**Beginner-Friendly Picks:**", beginner_picks)
        lines.append("")
        lines += _render_tier_section("**Professional-Grade Picks:**", professional_picks)
        answer = "\n".join(lines)

        structured_items = [
            {
                "id": it["id"],
                "name": it["title"],
                "price": it.get("price"),
                "url": it["url"],
                "source": it["source"],
                "tier": it["tier"],
            }
            for it in (beginner_picks + professional_picks)
        ]
        answer = _embed_product_data(answer, structured_items)

        return {"messages": [AIMessage(content=answer, name="product_search")]}

    # --- Specialist 7: invoice ----------------------------------------------
    # Zero LLM calls -- see this module's module-level helpers
    # (_collect_product_catalog / _select_invoice_items) and
    # mcp_server/invoice_tools.py's own docstring for why: every number on
    # an invoice is computed in plain Python from structured data parsed
    # out of prior product_search messages, never from an LLM re-reading
    # its own past answers.
    generate_invoice_tool = tools_by_name.get("generate_invoice")

    async def invoice_node(state: AgentState) -> dict:
        request_text = _last_human_text(state)
        catalog = _collect_product_catalog(state["messages"])

        if not catalog:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "I don't see any product search results yet in this conversation, "
                            "so there's nothing to invoice. Ask me to look up some art supplies "
                            "first, then I can build an invoice for the ones you want."
                        ),
                        name="invoice",
                    )
                ]
            }

        # "Most recent batch" for the fallback case in
        # _select_invoice_items: the last product_search message's own
        # items, not the merged multi-search catalog. Shared with
        # supervisor.py's pre-LLM routing check via the same helper --
        # see _latest_product_search_batch's own docstring for why using
        # ONE shared function (not two independent backward walks that
        # could drift apart) matters here specifically.
        latest_batch = _latest_product_search_batch(state["messages"])

        selected, assumption_note = _select_invoice_items(request_text, latest_batch)

        if not selected:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "I found earlier product search results in this conversation, but "
                            "couldn't tell which items you want invoiced. Try naming them "
                            "directly, or say \"all of them\" to invoice everything found so far."
                        ),
                        name="invoice",
                    )
                ]
            }

        invoice_items = [
            {"name": it.get("name"), "price": it.get("price"), "quantity": 1, "url": it.get("url", "")}
            for it in selected
        ]

        if generate_invoice_tool is None:
            return {
                "messages": [
                    AIMessage(
                        content="The invoice-generation tool isn't available right now.",
                        name="invoice",
                    )
                ]
            }

        raw = await generate_invoice_tool.ainvoke({"items": invoice_items, "customer_note": ""})
        result = unwrap_tool_result(raw) or {}

        answer = result.get("invoice_markdown", "(invoice could not be generated)")
        if assumption_note:
            answer = f"{answer}\n\n{assumption_note}"
        # Deliberately NOT surfacing result["file_path"] here (it's still
        # saved server-side by build_invoice/_write_invoice_file -- this
        # only affects what the user sees in chat): a local filesystem
        # path is Windows-vs-server-machine-specific and not something
        # the person reading the invoice in a browser can do anything
        # with, so showing it added noise without adding value.

        return {"messages": [AIMessage(content=answer, name="invoice")]}

    # --- Specialist 8: color-palette -----------------------------------
    # Zero LLM calls, same as image_qa/invoice -- every hex/rgb value,
    # every scheme's hue math, and every "closest named color" lookup is
    # plain, reproducible arithmetic done in mcp_server/color_tools.py,
    # never an LLM's own guess at what "#3f7cac" or "a triadic scheme"
    # means. `_parse_color_request` above does only light, deterministic
    # text triage (explicit color vs. mood, which scheme if any); the
    # real color resolution -- including the mood-word associations in
    # both directions -- happens entirely on the tool side.
    color_palette_tool = tools_by_name.get("generate_color_palette")

    async def color_palette_node(state: AgentState) -> dict:
        question = _last_human_text(state)
        parsed = _parse_color_request(question)

        if color_palette_tool is None:
            return {
                "messages": [
                    AIMessage(
                        content="The color-palette tool isn't available right now.",
                        name="color_palette",
                    )
                ]
            }

        raw = await color_palette_tool.ainvoke(
            {
                # generate_color_palette's own MCP schema declares plain,
                # non-nullable `str = ""` parameters (see server.py's own
                # tool signature; the server-side wrapper already
                # normalizes an empty string back to None internally via
                # `color or None` before calling color_tools.generate_
                # palette) -- but _parse_color_request above returns
                # Optional[str], using Python None for "not provided".
                # CONFIRMED BUG this closes: sending that None straight
                # through .ainvoke() raises a pydantic ValidationError
                # ("Input should be a valid string") inside the MCP
                # tool-calling machinery itself, synchronously, with
                # NOTHING here to catch it -- which then propagates all
                # the way out of this node, out of graph.ainvoke(), into
                # api.py's blanket exception handler, which (correctly,
                # for what it's designed to catch) reports it to the
                # caller as a 503 "Ollama isn't running, or the MCP
                # server subprocess died" -- a diagnosis that was
                # completely wrong for what was actually a plain type
                # mismatch in this node's own tool-call arguments, not
                # an infrastructure failure at all. `or ""` converts
                # each None to the empty string the tool's schema
                # actually expects, matching the type-safe calling
                # convention every other specialist's own tool call in
                # this file already follows (e.g. product_search_node's
                # `{"query": question, "max_results": 12}` never passes
                # None for a required argument either) -- the fix here
                # is to stop being the one exception to that, not to add
                # a new defensive try/except this file doesn't use
                # anywhere else.
                "color": parsed["color"] or "",
                "mood": parsed["mood"] or "",
                "scheme": parsed["scheme"] or "",
            }
        )
        result = unwrap_tool_result(raw) or {}
        answer = _format_color_palette_answer(result)
        return {"messages": [AIMessage(content=answer, name="color_palette")]}

    # --- Specialist 9: framing-quote ------------------------------------
    # The ONE specialist in this file whose tool call crosses a real
    # network boundary into an entirely separate, independently
    # deployed service (System B -- framing_agent/, Google ADK +
    # FastAPI, its own container) rather than staying inside this
    # process or this project's own MCP server. get_framing_quote
    # (mcp_server/server.py) is still called the exact same way every
    # other tool in this file is called -- `.ainvoke({...})` against a
    # handle pulled from `tools_by_name` -- specifically SO that this
    # node doesn't need to know or care that the real work happens
    # across that boundary; the MCP layer already hides it, the same
    # way it hides retrieve()'s local ChromaDB call or
    # search_art_supplies' own internet call behind an identical
    # interface. See mcp_server/framing_tools.py's own module docstring
    # for what happens on the wire.
    framing_quote_tool = tools_by_name.get("get_framing_quote")

    async def framing_quote_node(state: AgentState) -> dict:
        question = _last_human_text(state)
        parsed = _parse_framing_request(question)

        if parsed["missing"]:
            missing_list = "; ".join(parsed["missing"])
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "I can get a framing and shipping quote, but I'm still "
                            f"missing: {missing_list}. Give me the artwork's size, "
                            "what it's made of, and where it's shipping to, and "
                            "I'll get a real quote."
                        ),
                        name="framing_quote",
                    )
                ]
            }

        if framing_quote_tool is None:
            return {
                "messages": [
                    AIMessage(
                        content="The framing & shipping quote tool isn't available right now.",
                        name="framing_quote",
                    )
                ]
            }

        raw = await framing_quote_tool.ainvoke(
            {
                "width_cm": parsed["width_cm"],
                "height_cm": parsed["height_cm"],
                "medium": parsed["medium"],
                "destination_country": parsed["destination_country"],
                "frame_style": "",
            }
        )
        result = unwrap_tool_result(raw) or {}

        if not result.get("available"):
            # System B was unreachable, errored, or timed out --
            # request_quote()'s own "error" string is already
            # human-readable and safe to show directly (see its
            # docstring: never a raw stack trace), so it's surfaced
            # as-is rather than wrapped in extra framing-node text that
            # would just repeat the same information.
            error = result.get("error") or (
                "The framing & shipping quote service isn't available right now."
            )
            return {"messages": [AIMessage(content=error, name="framing_quote")]}

        quote = result.get("quote") or {}
        explanation = result.get("explanation") or "(no explanation returned)"
        answer = explanation

        subtotal = quote.get("subtotal_usd")
        if subtotal is not None:
            answer += f"\n\n**Estimated total: ${subtotal:.2f}**"

        disclaimer = quote.get("disclaimer")
        if disclaimer:
            answer += f"\n\n_{disclaimer}_"

        return {"messages": [AIMessage(content=answer, name="framing_quote")]}

    # Dict order here is NOT cosmetic -- it's `ordered_specialist_names`
    # in supervisor.py's build_supervisor(), i.e. the exact order
    # `_next_untried_route`'s fallback walk tries specialists in whenever
    # the repeat-route guard has to override a stuck model. Originally
    # this was retrieval_qa, corpus_meta, multi_hop, image_qa,
    # painting_lookup, product_search, invoice (build order = discovery
    # order in this file) -- which put `invoice` LAST. A confirmed
    # live run (`python -m agents.graph "Tell me about the Mona Lisa"`)
    # showed the model repeat-picking `painting_lookup` on every visit,
    # so the guard walked every other specialist in this exact order and
    # force-FINISHed only after reaching `invoice` last -- whose answer
    # ("I don't see any product search results yet...") is a
    # near-universal refusal with nothing to do with the actual
    # question. supervisor.py's `_finalize_with_first_attempt` now fixes
    # WHICH answer gets reaffirmed as final in that situation (the
    # first specialist to answer, not the last one walked) -- but
    # reordering here, so a full walk is less likely to burn its last
    # slot on a near-guaranteed-empty specialist in the first place, is
    # a second, independent mitigation for the same failure: retrieval_qa
    # stays first (it's DEFAULT_FALLBACK_ROUTE, the safest first pick),
    # then the "often has nothing to say without real underlying data"
    # specialists (product_search, invoice) are moved up early rather
    # than left last, so a full walk's last slot lands on one of the
    # specialists most likely to have attempted a genuine answer
    # (multi_hop, image_qa, painting_lookup). color_palette joins that
    # same "usually has something to say" group as product_search/
    # invoice -- it's fully deterministic and degrades to a plain "I
    # couldn't recognize that" rather than a hard refusal -- so it sits
    # right alongside them rather than in the "most likely to have
    # attempted a genuine answer" tail.
    # personal_docs sits right after retrieval_qa: same "answer from
    # retrieved chunks" shape and the same DEFAULT_FALLBACK_ROUTE-adjacent
    # safety as retrieval_qa for the fallback walk described above, but
    # placed second (not first) since -- unlike retrieval_qa, which always
    # has the whole main corpus to fall back on -- personal_docs has
    # nothing to say at all in a thread nobody has uploaded a file into,
    # so it shouldn't be the very first thing a stuck-model walk tries.
    # framing_quote sits right next to product_search/invoice for the
    # same reason they were moved up: it's the specialist in this whole
    # file MOST likely to come back with "I'm missing X" rather than a
    # genuine answer (three separate free-text fields -- dimensions,
    # medium, destination -- all have to actually be present in the
    # message, a stricter bar than invoice's own single "is there a
    # product_search batch yet" check), so a full repeat-route-guard
    # walk should never be left to burn its LAST slot here.
    return {
        "retrieval_qa": retrieval_qa_node,
        "personal_docs": personal_docs_node,
        "corpus_meta": corpus_meta_node,
        "product_search": product_search_node,
        "invoice": invoice_node,
        "framing_quote": framing_quote_node,
        "color_palette": color_palette_node,
        "multi_hop": multi_hop_node,
        "image_qa": image_qa_node,
        "painting_lookup": painting_lookup_node,
    }
