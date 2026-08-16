"""
Smoke tests for the four new specialists added on top of the original
three: image_qa, painting_lookup, product_search, invoice.

Same philosophy and fake shapes as test_specialists_smoke.py (see that
file's own module docstring) -- fake MCP tools, a scripted fake chat
model, real specialists.py control flow, no live Ollama or MCP server.
Fakes are redefined locally rather than imported from
test_specialists_smoke.py, matching this project's existing convention
of each test file owning its own small fakes (see
test_guardrails_smoke.py's own ScriptedRouterLLM, which does the same
rather than importing test_supervisor_smoke.py's).

Two flavors of fake tool are used, deliberately, mirroring
test_specialists_smoke.py's own split:
  - "retrieve" / "generate_answer" must be REAL @tool-decorated
    callables, even in tests that barely touch them, because
    build_specialists() always binds them into a real
    create_react_agent (for retrieval_qa) whose ToolNode dispatches by
    name against real BaseTool objects -- a plain mock object isn't
    enough (see test_specialists_smoke.py's test_build_specialists_wiring
    for the same constraint).
  - The four NEW tools (retrieve_images, search_painting_online,
    search_art_supplies, generate_invoice) are never bound into a
    create_react_agent -- every new specialist calls them directly via
    .ainvoke() -- so a plain FakeMCPTool (a mock with just a `.name` and
    an async `.ainvoke`) is sufficient and lets tests assert call
    counts/args directly via unittest.mock's own recording.

Run with:
    python agents/test_new_specialists_smoke.py
    (or, from the project root: python -m agents.test_new_specialists_smoke)
"""

import asyncio
import json
import uuid
from typing import Any, List
from unittest.mock import AsyncMock, patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from agents import specialists


def _check(label: str, condition: bool):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    assert condition, label


# ---------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------


class ScriptedChatModel(BaseChatModel):
    responses: List[AIMessage] = []
    _i: int = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        msg = self.responses[self._i]
        self._i += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return self._generate(messages, stop, run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    @property
    def call_count(self) -> int:
        return self._i


def _mcp_text_result(payload: Any) -> list:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return [{"type": "text", "text": text, "id": str(uuid.uuid4())}]


class FakeMCPTool:
    """For the four NEW tools only -- see this module's docstring."""

    def __init__(self, name: str, result: Any):
        self.name = name
        self.ainvoke = AsyncMock(return_value=_mcp_text_result(result))


def _make_retrieve_tool(result: Any = None):
    """
    Real @tool-decorated "retrieve", required because build_specialists()
    binds it into a real create_react_agent (see this module's docstring)
    -- ToolNode's constructor normalizes every bound tool at BUILD time,
    which fails immediately for a plain FakeMCPTool object, so this one
    specific tool must always be a real BaseTool regardless of which
    specialist a given test actually exercises.

    Deliberately returns MCP-wire-format output (via _mcp_text_result),
    not the raw chunk list, even though this is a real, locally-invoked
    @tool: painting_lookup_node and multi_hop_node both call
    `retrieve_tool.ainvoke(...)` directly (bypassing create_react_agent's
    ToolNode) and immediately run the result through
    `unwrap_tool_result()`, exactly as they would against a REAL
    langchain-mcp-adapters tool (which always returns wire-wrapped output,
    since it actually crossed the MCP protocol boundary). A real @tool
    invoked in-process would otherwise skip that wrapping entirely,
    silently testing a code path production never takes.

    Returns (tool, calls) where `calls` records each invocation's kwargs,
    since a plain @tool function has no AsyncMock-style call recording of
    its own.
    """
    calls: list = []

    @tool
    def retrieve(query: str, k: int = 5) -> list:
        """Fake retrieve tool for new-specialist tests."""
        calls.append({"query": query, "k": k})
        return _mcp_text_result(result if result is not None else [])

    return retrieve, calls


def _make_generate_answer_tool(result: str = ""):
    calls: list = []

    @tool
    def generate_answer(query: str, chunks: list) -> str:
        """Fake generate_answer tool for new-specialist tests."""
        calls.append({"query": query, "chunks": chunks})
        return _mcp_text_result(result)

    return generate_answer, calls


FAKE_CORPUS = {"documents": [], "total_documents": 0, "total_chunks": 0}


async def _build(extra_tools: dict, llm_responses: list, retrieve_result: Any = None) -> tuple[dict, ScriptedChatModel, list]:
    """
    Build the full specialists dict (all seven) with a fake MCP client
    layer and a scripted LLM. `extra_tools` supplies whichever of the
    four new tools a given test cares about (retrieve_images,
    search_painting_online, search_art_supplies, generate_invoice);
    "retrieve" and "generate_answer" are always auto-included as real
    @tool callables so build_specialists() never fails to construct
    retrieval_qa's react agent, regardless of which specialist a given
    test actually exercises. Returns (node_map, llm, retrieve_calls).
    """
    retrieve_tool, retrieve_calls = _make_retrieve_tool(retrieve_result)
    generate_tool, _ = _make_generate_answer_tool("")
    tools_by_name = {"retrieve": retrieve_tool, "generate_answer": generate_tool, **extra_tools}

    llm = ScriptedChatModel(responses=llm_responses)
    with patch("agents.specialists.build_client", return_value=object()), \
         patch("agents.specialists.load_tools_by_name", new=AsyncMock(return_value=tools_by_name)), \
         patch("agents.specialists.fetch_corpus_documents", new=AsyncMock(return_value=FAKE_CORPUS)), \
         patch("agents.specialists.ChatOllama", return_value=llm):
        node_map = await specialists.build_specialists()
    return node_map, llm, retrieve_calls


def _state(question: str, extra_messages: list = None) -> dict:
    messages = list(extra_messages or [])
    messages.append(HumanMessage(content=question))
    return {"messages": messages, "route": None, "iteration_count": 0, "blocked": False, "injection_patterns": []}


# ---------------------------------------------------------------------
# image_qa
# ---------------------------------------------------------------------


async def test_image_qa_node_formats_results():
    print("\n=== image_qa_node: formats retrieved images with captions, zero LLM calls ===")
    fake_images = [
        {"image_path": "/data/raw/palette.png", "caption": "A wooden palette with mixed oil paints",
         "score": 0.91, "metadata": {}},
        {"image_path": "/data/raw/canvas.png", "caption": "A primed blank canvas on an easel",
         "score": 0.85, "metadata": {}},
    ]
    node_map, llm, _ = await _build({"retrieve_images": FakeMCPTool("retrieve_images", fake_images)}, [])

    result = await node_map["image_qa"](_state("show me a painter's palette"))
    answer = result["messages"][0].content

    _check("zero LLM calls (image_qa is fully deterministic)", llm.call_count == 0)
    # _format_image_result() deliberately reduces image_path down to just
    # its basename under a /images/ URL (see its own docstring) -- a
    # browser can't load the server's local /data/raw/... path directly.
    # This assertion was checking for the pre-reduction full path, which
    # this node has never actually emitted; fixed to match its real,
    # intended output rather than changing the node to match a stale test.
    _check("both images embedded", "/images/palette.png" in answer and "/images/canvas.png" in answer)
    _check("both captions present", "wooden palette" in answer and "primed blank canvas" in answer)
    _check("message is named image_qa", result["messages"][0].name == "image_qa")


async def test_image_qa_node_says_plainly_when_nothing_found():
    print("\n=== image_qa_node: no images found -> says so plainly, doesn't fabricate ===")
    node_map, llm, _ = await _build({"retrieve_images": FakeMCPTool("retrieve_images", [])}, [])

    result = await node_map["image_qa"](_state("show me a griffin eating a sandwich"))
    answer = result["messages"][0].content
    _check("zero LLM calls", llm.call_count == 0)
    _check("plain 'couldn't find' message returned", "couldn't find any relevant images" in answer)


async def test_image_qa_node_prefers_embedded_tool_when_available():
    print("\n=== image_qa_node: retrieve_images_embedded present -> uses base64 data URIs, not /images/ paths ===")
    fake_images_embedded = [
        {
            "image_path": "/data/raw/palette.png",
            "caption": "A wooden palette with mixed oil paints",
            "score": 0.91,
            "metadata": {},
            "image_base64": "QUJD",
            "mime_type": "image/png",
            "data_uri": "data:image/png;base64,QUJD",
        },
    ]
    # Both tools available at once (a real server exposes both) -- the
    # embedded one must win, and the plain retrieve_images tool must
    # never even be called.
    plain_tool = FakeMCPTool("retrieve_images", [{"image_path": "/should/not/be/used.png", "caption": "x", "score": 0.1, "metadata": {}}])
    node_map, llm, _ = await _build(
        {
            "retrieve_images": plain_tool,
            "retrieve_images_embedded": FakeMCPTool("retrieve_images_embedded", fake_images_embedded),
        },
        [],
    )

    result = await node_map["image_qa"](_state("show me a painter's palette"))
    answer = result["messages"][0].content

    _check("zero LLM calls", llm.call_count == 0)
    _check("data URI embedded in the answer", "data:image/png;base64,QUJD" in answer)
    _check("caption present", "wooden palette" in answer)
    _check("no /images/ path in the answer -- fully self-contained", "/images/" not in answer)
    _check("the plain retrieve_images tool was never called", plain_tool.ainvoke.await_count == 0)


async def test_image_qa_node_falls_back_to_path_when_embed_fails_for_one_item():
    print("\n=== image_qa_node: an item with no data_uri (encoding failed) falls back to path rendering for that item ===")
    fake_images_embedded = [
        {"image_path": "/data/raw/ok.png", "caption": "Encoded fine", "score": 0.9, "metadata": {},
         "image_base64": "QUJD", "mime_type": "image/png", "data_uri": "data:image/png;base64,QUJD"},
        {"image_path": "/data/raw/too_big.png", "caption": "Too large to embed", "score": 0.8, "metadata": {},
         "image_base64": None, "mime_type": None, "data_uri": None, "encoding_note": "over the size cap"},
    ]
    node_map, llm, _ = await _build(
        {"retrieve_images_embedded": FakeMCPTool("retrieve_images_embedded", fake_images_embedded)}, []
    )

    result = await node_map["image_qa"](_state("show me some references"))
    answer = result["messages"][0].content

    _check("zero LLM calls", llm.call_count == 0)
    _check("the embeddable item rendered as a data URI", "data:image/png;base64,QUJD" in answer)
    _check("the non-embeddable item fell back to its /images/ path", "/images/too_big.png" in answer)
    _check("both captions present", "Encoded fine" in answer and "Too large to embed" in answer)


async def test_image_qa_node_uses_plain_tool_when_embedded_tool_absent():
    print("\n=== image_qa_node: no retrieve_images_embedded tool -> old retrieve_images path is completely unchanged ===")
    fake_images = [
        {"image_path": "/data/raw/palette.png", "caption": "A wooden palette with mixed oil paints",
         "score": 0.91, "metadata": {}},
    ]
    node_map, llm, _ = await _build({"retrieve_images": FakeMCPTool("retrieve_images", fake_images)}, [])

    result = await node_map["image_qa"](_state("show me a painter's palette"))
    answer = result["messages"][0].content

    _check("zero LLM calls", llm.call_count == 0)
    _check("falls back to the original /images/ path rendering", "/images/palette.png" in answer)
    _check("caption present", "wooden palette" in answer)


async def test_image_qa_node_escapes_a_caption_with_a_stray_bracket():
    print("\n=== image_qa_node: a caption containing ']' no longer spills the data_uri as raw text ===")
    # Reproduces a confirmed live-run failure: a VLM-generated caption
    # (free-form text -- config.IMAGE_CAPTION_PROMPT has no punctuation
    # constraint) containing an unescaped "]" closed the image's
    # `![...]` alt-text span early, so the entire base64 data_uri that
    # followed rendered as raw visible text in the chat UI instead of
    # an image.
    fake_images_embedded = [
        {
            "image_path": "/data/raw/palette.png",
            "caption": "A palette [detail: mixed oils] on a wooden table",
            "score": 0.9,
            "metadata": {},
            "image_base64": "QUJD",
            "mime_type": "image/png",
            "data_uri": "data:image/png;base64,QUJD",
        },
    ]
    node_map, llm, _ = await _build(
        {"retrieve_images_embedded": FakeMCPTool("retrieve_images_embedded", fake_images_embedded)}, []
    )

    result = await node_map["image_qa"](_state("show me a painter's palette"))
    answer = result["messages"][0].content

    _check("zero LLM calls", llm.call_count == 0)
    _check("the opening bracket is escaped", "\\[detail" in answer)
    _check("the closing bracket is escaped", "oils\\]" in answer)
    _check("exactly one image tag opens -- the alt span didn't close early", answer.count("![") == 1)
    _check("the data_uri is still inside the parens as a real image destination, not spilled as text",
           "](data:image/png;base64,QUJD)" in answer)


# ---------------------------------------------------------------------
# painting_lookup
# ---------------------------------------------------------------------


async def test_painting_lookup_node_combines_corpus_and_web():
    print("\n=== painting_lookup_node: combines corpus + web into one answer with sources ===")
    fake_chunks = [{"text": "The treatise briefly mentions sfumato technique.",
                     "score": 0.7, "metadata": {"filename": "cennini.pdf"}}]
    fake_web = {
        "query": "Mona Lisa",
        "summary": "The Mona Lisa is a portrait by Leonardo da Vinci, famed for its sfumato technique.",
        "sources": [{"title": "Wikipedia: Mona Lisa", "url": "https://en.wikipedia.org/wiki/Mona_Lisa"}],
    }
    synth_answer = "The Mona Lisa, per the corpus and Wikipedia, is known for its sfumato technique."
    node_map, llm, retrieve_calls = await _build(
        {"search_painting_online": FakeMCPTool("search_painting_online", fake_web)},
        [AIMessage(content=synth_answer)],
        retrieve_result=fake_chunks,
    )

    result = await node_map["painting_lookup"](_state("Tell me about the Mona Lisa"))
    answer = result["messages"][0].content

    _check("exactly one LLM synthesis call", llm.call_count == 1)
    _check("retrieve was called exactly once", len(retrieve_calls) == 1)
    _check("LLM's synthesized text is in the final answer", synth_answer in answer)
    _check("the source link is appended (deterministically, not by the LLM)",
           "https://en.wikipedia.org/wiki/Mona_Lisa" in answer)
    _check("message is named painting_lookup", result["messages"][0].name == "painting_lookup")


async def test_painting_lookup_node_survives_missing_web_tool():
    print("\n=== painting_lookup_node: falls back gracefully if search_painting_online isn't available ===")
    fake_chunks = [{"text": "Some corpus content.", "score": 0.5, "metadata": {"filename": "x.pdf"}}]
    node_map, llm, _ = await _build({}, [AIMessage(content="Corpus-only answer.")], retrieve_result=fake_chunks)

    result = await node_map["painting_lookup"](_state("Tell me about an obscure painting"))
    _check("no crash when search_painting_online tool is missing", result["messages"][0].name == "painting_lookup")


# ---------------------------------------------------------------------
# product_search
# ---------------------------------------------------------------------


_FAKE_PRODUCTS_TIERED = [
    # 7 beginner-cue items (by keyword) -- more than the 5-per-tier cap,
    # deliberately, to prove capping actually happens.
    {"title": "Student Grade Watercolor Brush Set", "url": "https://www.amazon.com/dp/b1", "source": "amazon",
     "price": 9.99, "snippet": "Great starter kit for beginners, budget-friendly."},
    {"title": "Kids Paint Brush Value Pack", "url": "https://www.amazon.com/dp/b2", "source": "amazon",
     "price": 5.49, "snippet": "Perfect for kids and beginners."},
    {"title": "Basic Craft Brush Set", "url": "https://www.ebay.com/itm/b3", "source": "ebay",
     "price": None, "snippet": "No reviews yet, basic entry level set."},
    {"title": "Beginner Canvas Pack", "url": "https://www.amazon.com/dp/b4", "source": "amazon",
     "price": 15.00, "snippet": "Value pack for beginners learning to paint."},
    {"title": "Starter Easel Kit", "url": "https://www.ebay.com/itm/b5", "source": "ebay",
     "price": 22.00, "snippet": "Starter kit, great for students."},
    {"title": "Budget Acrylic Paint Set", "url": "https://www.amazon.com/dp/b6", "source": "amazon",
     "price": 12.99, "snippet": "Budget option, good value set."},
    {"title": "Entry Level Palette Knife Set", "url": "https://www.ebay.com/itm/b7", "source": "ebay",
     "price": 7.50, "snippet": "Entry-level tools for learning."},
    # 3 professional-cue items -- fewer than the cap, to prove a
    # not-fully-populated tier still renders correctly (not padded).
    {"title": "Winsor & Newton Series 7 Kolinsky Sable Brush", "url": "https://www.amazon.com/dp/g1",
     "source": "amazon", "price": 89.99, "snippet": "Professional grade, kolinsky sable, studio favorite."},
    {"title": "Archival Fine Art Canvas Roll", "url": "https://www.amazon.com/dp/g2", "source": "amazon",
     "price": 65.00, "snippet": "Conservation quality, archival, fine art use."},
    {"title": "Studio Master Easel", "url": "https://www.ebay.com/itm/g3", "source": "ebay",
     "price": 250.00, "snippet": "Professional studio easel, master craftsmanship."},
]


def test_classify_tier_keyword_cues():
    print("\n=== specialists._classify_tier: keyword cues resolve to the right tier ===")
    _check(
        "a 'kolinsky sable... professional grade' item is professional",
        specialists._classify_tier(
            {"title": "Kolinsky Sable Brush", "snippet": "Professional grade, studio favorite."}
        )
        == "professional",
    )
    _check(
        "a 'starter kit for beginners' item is beginner",
        specialists._classify_tier({"title": "Starter Kit", "snippet": "Great for beginners, budget-friendly."})
        == "beginner",
    )
    _check(
        "an item with neither keyword set is left unclassified (resolved later by the caller)",
        specialists._classify_tier({"title": "Plain Round Brush", "snippet": "A brush."}) == "unclassified",
    )


async def test_product_search_node_splits_into_tiers_capped_at_5_each():
    print("\n=== product_search_node: splits into beginner/professional, capped at 5 per tier ===")
    comparison = (
        "Beginner-friendly options: several budget starter kits are available, all under $25.\n\n"
        "Professional-grade options: the Winsor & Newton brush and archival canvas are both premium picks."
    )
    node_map, llm, _ = await _build(
        {"search_art_supplies": FakeMCPTool("search_art_supplies", _FAKE_PRODUCTS_TIERED)},
        [AIMessage(content=comparison)],
    )

    result = await node_map["product_search"](_state("what brushes and canvases should I buy"))
    answer = result["messages"][0].content

    _check("exactly one LLM call (the two comparison paragraphs)", llm.call_count == 1)
    _check("both tier headers are present", "Beginner-Friendly Picks:" in answer and "Professional-Grade Picks:" in answer)
    _check("the LLM's tiered comparison text is included", comparison in answer)

    parsed = specialists._parse_product_data(answer)
    beginner_parsed = [p for p in parsed if p["tier"] == "beginner"]
    professional_parsed = [p for p in parsed if p["tier"] == "professional"]
    _check(
        "beginner tier is capped at 5, even though 7 beginner-cue items were found",
        len(beginner_parsed) == 5,
    )
    _check(
        "professional tier includes all 3 it found (no padding to reach 5)",
        len(professional_parsed) == 3,
    )
    _check("total structured items is 5 + 3 = 8", len(parsed) == 8)
    _check(
        "beginner ids use the 'b' prefix, professional ids use the 'g' prefix",
        all(p["id"].startswith("b") for p in beginner_parsed)
        and all(p["id"].startswith("g") for p in professional_parsed),
    )
    _check(
        "every parsed item kept its real, allowlisted URL",
        all(p["url"].startswith("https://www.amazon.com") or p["url"].startswith("https://www.ebay.com") for p in parsed),
    )
    _check(
        "no item invented by the LLM leaked into the structured data -- every name traces back to the fixture",
        {p["name"] for p in parsed} <= {it["title"] for it in _FAKE_PRODUCTS_TIERED},
    )


async def test_product_search_node_unclassified_items_resolved_by_price_median():
    print("\n=== product_search_node: keyword-less items fall back to a price-vs-median split ===")
    # No keyword cues at all -- every item's tier has to come from the
    # price-relative-to-median tiebreak. Prices: 5, 10, 15, 50, 100 ->
    # median of 5 values is the middle one, 15. Items AT OR BELOW the
    # median -> beginner; items ABOVE it -> professional.
    unclassified_items = [
        {"title": "Plain Brush A", "url": "https://www.amazon.com/dp/u1", "source": "amazon",
         "price": 5.00, "snippet": "A brush."},
        {"title": "Plain Brush B", "url": "https://www.amazon.com/dp/u2", "source": "amazon",
         "price": 10.00, "snippet": "Another brush."},
        {"title": "Plain Brush C", "url": "https://www.ebay.com/itm/u3", "source": "ebay",
         "price": 15.00, "snippet": "Yet another brush."},
        {"title": "Plain Brush D", "url": "https://www.amazon.com/dp/u4", "source": "amazon",
         "price": 50.00, "snippet": "A pricier brush."},
        {"title": "Plain Brush E", "url": "https://www.ebay.com/itm/u5", "source": "ebay",
         "price": 100.00, "snippet": "The priciest brush."},
    ]
    node_map, llm, _ = await _build(
        {"search_art_supplies": FakeMCPTool("search_art_supplies", unclassified_items)},
        [AIMessage(content="Tiered by price since no quality cues were present in the snippets.")],
    )

    result = await node_map["product_search"](_state("brushes"))
    parsed = specialists._parse_product_data(result["messages"][0].content)
    tiers_by_name = {p["name"]: p["tier"] for p in parsed}

    _check(
        "$5, $10, and $15 items (at/below the $15 median) landed in beginner",
        tiers_by_name["Plain Brush A"] == "beginner"
        and tiers_by_name["Plain Brush B"] == "beginner"
        and tiers_by_name["Plain Brush C"] == "beginner",
    )
    _check(
        "$50 and $100 items (above the median) landed in professional",
        tiers_by_name["Plain Brush D"] == "professional" and tiers_by_name["Plain Brush E"] == "professional",
    )


async def test_product_search_node_says_plainly_when_one_tier_is_empty():
    print("\n=== product_search_node: an empty tier says so plainly, isn't padded with the other tier's items ===")
    only_beginner_items = [
        {"title": "Starter Brush Set", "url": "https://www.amazon.com/dp/only1", "source": "amazon",
         "price": 8.00, "snippet": "Great starter kit for beginners."},
        {"title": "Kids Value Pack", "url": "https://www.amazon.com/dp/only2", "source": "amazon",
         "price": 6.00, "snippet": "Budget value pack for kids."},
    ]
    node_map, llm, _ = await _build(
        {"search_art_supplies": FakeMCPTool("search_art_supplies", only_beginner_items)},
        [AIMessage(content="Only beginner-friendly options turned up for this search.")],
    )

    result = await node_map["product_search"](_state("cheap starter brushes"))
    answer = result["messages"][0].content
    parsed = specialists._parse_product_data(answer)

    _check("both items landed in the beginner tier", all(p["tier"] == "beginner" for p in parsed))
    _check("the professional section explicitly says none were found", "_None found for this search._" in answer)
    _check("no professional-tier structured items were fabricated to fill the gap",
           not any(p["tier"] == "professional" for p in parsed))


async def test_product_search_node_no_results_skips_the_llm_entirely():
    print("\n=== product_search_node: no search results -> says so, never calls the LLM ===")
    node_map, llm, _ = await _build({"search_art_supplies": FakeMCPTool("search_art_supplies", [])}, [])

    result = await node_map["product_search"](_state("best watercolor brushes"))
    answer = result["messages"][0].content
    _check("zero LLM calls when there's nothing to compare", llm.call_count == 0)
    _check("plain 'couldn't find' message, not a fabricated product",
           "couldn't find any art-supply listings" in answer)
    _check("PRODUCT_DATA footer still present (as an empty list, for a consistent parse contract)",
           specialists._parse_product_data(answer) == [])


# ---------------------------------------------------------------------
# invoice
# ---------------------------------------------------------------------


def _fake_product_search_message(items: list[dict]) -> AIMessage:
    body = "Some comparison text.\n\n**Top picks:**\n" + "\n".join(f"- {it['name']}" for it in items)
    return AIMessage(content=specialists._embed_product_data(body, items), name="product_search", id=str(uuid.uuid4()))


_CATALOG_ITEMS = [
    {"id": "p1", "name": "Winsor & Newton Series 7 Brush", "price": 24.99,
     "url": "https://www.amazon.com/dp/1", "source": "amazon"},
    {"id": "p2", "name": "Fredrix Stretched Canvas", "price": 12.50,
     "url": "https://www.amazon.com/dp/3", "source": "amazon"},
]


async def test_invoice_node_no_catalog_yet():
    print("\n=== invoice_node: no prior product_search in history -> says so, no tool call ===")
    node_map, llm, _ = await _build({"generate_invoice": FakeMCPTool("generate_invoice", {})}, [])

    result = await node_map["invoice"](_state("give me an invoice"))
    answer = result["messages"][0].content
    _check("zero LLM calls (invoice is fully deterministic)", llm.call_count == 0)
    _check("explains there's nothing to invoice yet", "don't see any product search results" in answer)


async def test_invoice_node_all_phrase_selects_everything():
    print("\n=== invoice_node: \"all of them\" selects every item in the latest batch ===")
    fake_invoice_result = {
        "invoice_markdown": "# Invoice\n\n**Subtotal: $37.49**", "subtotal": 37.49,
        "item_count": 2, "skipped": [], "file_path": "/tmp/invoice_x.md",
    }
    invoice_tool = FakeMCPTool("generate_invoice", fake_invoice_result)
    node_map, llm, _ = await _build({"generate_invoice": invoice_tool}, [])

    history = [_fake_product_search_message(_CATALOG_ITEMS)]
    result = await node_map["invoice"](_state("invoice everything you found", extra_messages=history))
    answer = result["messages"][0].content

    _check("zero LLM calls", llm.call_count == 0)
    call_args = invoice_tool.ainvoke.call_args.args[0]
    _check("both catalog items were sent to generate_invoice", len(call_args["items"]) == 2)
    _check("the invoice markdown made it into the final answer", "$37.49" in answer)
    _check("the saved file path is surfaced", "/tmp/invoice_x.md" in answer)


async def test_invoice_node_selects_by_name_match():
    print("\n=== invoice_node: matches specific item(s) named in the request ===")
    invoice_tool = FakeMCPTool("generate_invoice", {"invoice_markdown": "ok", "file_path": None})
    node_map, llm, _ = await _build({"generate_invoice": invoice_tool}, [])

    history = [_fake_product_search_message(_CATALOG_ITEMS)]
    result = await node_map["invoice"](_state("just invoice the brush please", extra_messages=history))

    call_args = invoice_tool.ainvoke.call_args.args[0]
    names = [it["name"] for it in call_args["items"]]
    _check("only the brush was selected, not the canvas", names == ["Winsor & Newton Series 7 Brush"])
    _check("final answer carries no assumption note (the match was unambiguous)",
           "I assumed" not in result["messages"][0].content)


async def test_invoice_node_falls_back_to_latest_batch_with_a_note():
    print("\n=== invoice_node: ambiguous request falls back to the MOST RECENT product_search batch ===")
    older_items = [{"id": "p1", "name": "Old Discontinued Easel", "price": 40.0,
                     "url": "https://www.amazon.com/dp/old", "source": "amazon"}]
    newer_items = _CATALOG_ITEMS
    invoice_tool = FakeMCPTool("generate_invoice", {"invoice_markdown": "ok", "file_path": None})
    node_map, llm, _ = await _build({"generate_invoice": invoice_tool}, [])

    history = [_fake_product_search_message(older_items), _fake_product_search_message(newer_items)]
    result = await node_map["invoice"](_state("okay, make me an invoice", extra_messages=history))

    call_args = invoice_tool.ainvoke.call_args.args[0]
    billed_names = {it["name"] for it in call_args["items"]}
    _check("fell back to the newer batch's two items, not the older single item",
           billed_names == {"Winsor & Newton Series 7 Brush", "Fredrix Stretched Canvas"})
    _check("an assumption note was surfaced to the user", "I assumed" in result["messages"][0].content)


async def test_invoice_node_scopes_selection_to_latest_batch_only():
    print("\n=== invoice_node: item selection is scoped to the MOST RECENT batch, "
          "never reaches back into an earlier one -- even for \"all of them\" or a "
          "name match that happens to overlap an older item ===")
    older_items = [{"id": "p1", "name": "Old Discontinued Easel", "price": 40.0,
                     "url": "https://www.amazon.com/dp/old", "source": "amazon"}]
    newer_items = _CATALOG_ITEMS
    invoice_tool = FakeMCPTool("generate_invoice", {"invoice_markdown": "ok", "file_path": None})
    node_map, llm, _ = await _build({"generate_invoice": invoice_tool}, [])

    history = [_fake_product_search_message(older_items), _fake_product_search_message(newer_items)]

    # "all of them" must mean "all of the MOST RECENT search", not "every
    # item this conversation has ever surfaced" -- the older, unrelated
    # easel from a previous search must not silently ride along.
    result = await node_map["invoice"](_state("invoice all of them", extra_messages=history))
    billed_names = {it["name"] for it in invoice_tool.ainvoke.call_args.args[0]["items"]}
    _check("\"all of them\" only pulled in the latest batch's two items",
           billed_names == {"Winsor & Newton Series 7 Brush", "Fredrix Stretched Canvas"})
    _check("the older, unrelated easel was NOT included", "Old Discontinued Easel" not in billed_names)


async def main():
    await test_image_qa_node_formats_results()
    await test_image_qa_node_says_plainly_when_nothing_found()
    await test_image_qa_node_prefers_embedded_tool_when_available()
    await test_image_qa_node_falls_back_to_path_when_embed_fails_for_one_item()
    await test_image_qa_node_uses_plain_tool_when_embedded_tool_absent()
    await test_image_qa_node_escapes_a_caption_with_a_stray_bracket()
    await test_painting_lookup_node_combines_corpus_and_web()
    await test_painting_lookup_node_survives_missing_web_tool()
    test_classify_tier_keyword_cues()
    await test_product_search_node_splits_into_tiers_capped_at_5_each()
    await test_product_search_node_unclassified_items_resolved_by_price_median()
    await test_product_search_node_says_plainly_when_one_tier_is_empty()
    await test_product_search_node_no_results_skips_the_llm_entirely()
    await test_invoice_node_no_catalog_yet()
    await test_invoice_node_all_phrase_selects_everything()
    await test_invoice_node_selects_by_name_match()
    await test_invoice_node_falls_back_to_latest_batch_with_a_note()
    await test_invoice_node_scopes_selection_to_latest_batch_only()
    print("\nAll new-specialist smoke tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
