"""
Smoke tests for color_palette_node (agents/specialists.py) and its
underlying mcp_server/color_tools.py.

Two layers, tested separately, same split test_new_specialists_smoke.py
already uses for its own tool-backed specialists:
  - color_tools.py is pure, dependency-free Python (parsing, hue math,
    scheme generation, mood matching) -- tested directly, no fakes
    needed, same reasoning invoice_tools.py's own arithmetic is testable
    standalone.
  - specialists._parse_color_request (the deterministic color-vs-mood
    text triage) is tested directly as a pure function, same pattern
    test_new_specialists_smoke.py already uses for
    specialists._classify_tier / specialists._parse_product_data.
  - color_palette_node itself is tested through the same fake-MCP
    harness test_new_specialists_smoke.py uses, with one difference:
    the fake tool's result IS the real color_tools.generate_palette()
    output (not a hand-typed canned dict) -- there's no reason to
    hand-fake a pure, deterministic function's output when the real one
    is just as fast and actually exercises the color math this
    specialist depends on.

Run with:
    python agents/test_color_palette_smoke.py
    (or, from the project root: python -m agents.test_color_palette_smoke)
"""

import asyncio
import base64
import json
import uuid
from typing import Any
from unittest.mock import AsyncMock, patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from agents import specialists
from mcp_server import color_tools


def _check(label: str, condition: bool):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    assert condition, label


# ---------------------------------------------------------------------
# color_tools.py -- pure function tests, no fakes needed
# ---------------------------------------------------------------------


def test_parse_color_text_hex_rgb_name():
    print("\n=== color_tools.parse_color_text: hex, rgb, and name inputs ===")
    _check("6-digit hex parses", color_tools.parse_color_text("#3f7cac") == (0x3F, 0x7C, 0xAC))
    _check("3-digit hex parses (expanded)", color_tools.parse_color_text("#fff") == (255, 255, 255))
    _check("rgb(...) form parses", color_tools.parse_color_text("rgb(63, 124, 172)") == (63, 124, 172))
    _check("bare triplet parses", color_tools.parse_color_text("63, 124, 172") == (63, 124, 172))
    _check(
        "known name parses",
        color_tools.parse_color_text("forest green") == color_tools.hex_to_rgb("#228b22"),
    )
    _check(
        "unrecognized text returns None",
        color_tools.parse_color_text("not a color at all xyz") is None,
    )


def test_hex_rgb_roundtrip():
    print("\n=== color_tools hex/rgb roundtrip ===")
    for hexv in ["#000000", "#ffffff", "#3f7cac", "#a1b2c3"]:
        r, g, b = color_tools.hex_to_rgb(hexv)
        _check(f"{hexv} roundtrips through rgb_to_hex", color_tools.rgb_to_hex(r, g, b) == hexv)


def test_schemes_shape_and_membership():
    print("\n=== color_tools scheme generators: correct member counts, hue relationships ===")
    h, s, l = color_tools.rgb_to_hsl(*color_tools.hex_to_rgb("#3f7cac"))

    mono = color_tools.scheme_monochromatic(h, s, l)
    _check("monochromatic has 5 members", len(mono) == 5)
    _check("monochromatic includes the exact base lightness", any(abs(m[2] - l) < 1e-9 for m in mono))
    _check("monochromatic keeps the same hue throughout", all(abs(m[0] - h) < 1e-6 for m in mono))

    ana = color_tools.scheme_analogous(h, s, l)
    _check("analogous has 3 members", len(ana) == 3)
    _check("analogous is centered on the base hue", ana[1][0] == h)

    comp = color_tools.scheme_complementary(h, s, l)
    _check("complementary has exactly 2 members", len(comp) == 2)
    _check(
        "complementary's second member is 180 degrees opposite",
        abs((comp[1][0] - comp[0][0]) % 360 - 180) < 1e-6,
    )

    tri = color_tools.scheme_triadic(h, s, l)
    _check("triadic has exactly 3 members", len(tri) == 3)
    hues = sorted(m[0] % 360 for m in tri)
    gaps = [(hues[(i + 1) % 3] - hues[i]) % 360 for i in range(3)]
    _check("triadic members are 120 degrees apart", all(abs(g - 120) < 1e-6 for g in gaps))


def test_scheme_name_normalization():
    print("\n=== color_tools.normalize_scheme_name: aliases and unknowns ===")
    _check("'mono' resolves to monochromatic", color_tools.normalize_scheme_name("mono") == "monochromatic")
    _check(
        "'complement' resolves to complementary",
        color_tools.normalize_scheme_name("complement") == "complementary",
    )
    _check("'triad' resolves to triadic", color_tools.normalize_scheme_name("triad") == "triadic")
    _check(
        "unrecognized scheme name resolves to None (caller returns all four)",
        color_tools.normalize_scheme_name("hexadic") is None,
    )
    _check("empty/None resolves to None", color_tools.normalize_scheme_name(None) is None)


def test_color_from_mood_matches_and_averages():
    print("\n=== color_tools.color_from_mood: keyword matching + circular hue averaging ===")
    calm = color_tools.color_from_mood("I want this to feel calm and peaceful")
    _check(
        "a clear mood match returns matched keywords",
        set(calm["matched_keywords"]) >= {"calm", "peaceful"},
    )

    nothing = color_tools.color_from_mood("asdkjaslkdj nonsense text")
    _check("no keyword match returns None", nothing is None)

    # Circular-mean sanity check: two hues symmetric around the 0/360
    # wraparound (e.g. "angry"=0 and "passion(ate)"=355) should average
    # to somewhere near red, not drift toward green the way a naive
    # arithmetic mean of the raw degree values would.
    near_red = color_tools.color_from_mood("angry and passionate")
    h, _, _ = color_tools.rgb_to_hsl(*near_red["rgb"])
    _check("averaging two near-0-degree hues stays near red, not green", h < 30 or h > 330)


def test_generate_palette_end_to_end():
    print("\n=== color_tools.generate_palette: full orchestration ===")
    result = color_tools.generate_palette(color="#3f7cac", scheme="triadic")
    _check("no error for a valid hex + valid scheme", result["error"] is None)
    _check("input_type is 'color'", result["input_type"] == "color")
    _check("only the requested scheme is present", set(result["schemes"].keys()) == {"triadic"})
    _check("base color hex matches input", result["base_color"]["hex"] == "#3f7cac")

    all_schemes = color_tools.generate_palette(color="gold")
    _check(
        "omitted scheme returns all four",
        set(all_schemes["schemes"].keys()) == {"monochromatic", "analogous", "complementary", "triadic"},
    )

    bad_color = color_tools.generate_palette(color="not a real color at all")
    _check("unrecognized color returns an error, not a guess", bad_color["error"] is not None)
    _check("error result has no base_color", bad_color["base_color"] is None)

    bad_mood = color_tools.generate_palette(mood="asdkjaslkdj nonsense")
    _check("unrecognized mood returns an error, not a guess", bad_mood["error"] is not None)

    neither = color_tools.generate_palette()
    _check("neither color nor mood given returns an error", neither["error"] is not None)

    both = color_tools.generate_palette(color="red", mood="calm")
    _check("when both are given, color wins", both["input_type"] == "color")


def test_swatch_is_a_valid_data_uri():
    print("\n=== color_tools.swatch_data_uri: shape frontend's allowlist expects ===")
    uri = color_tools.swatch_data_uri("#3f7cac")
    _check(
        "starts with the exact prefix MarkdownText.tsx's allowlist regex expects",
        uri.startswith("data:image/svg+xml;base64,"),
    )
    decoded = base64.b64decode(uri.split(",", 1)[1]).decode("utf-8")
    _check("decodes back to real SVG containing the fill color", "#3f7cac" in decoded and "<svg" in decoded)


# ---------------------------------------------------------------------
# specialists.py -- _parse_color_request, pure function
# ---------------------------------------------------------------------


def test_parse_color_request_explicit_color():
    print("\n=== specialists._parse_color_request: explicit color detection ===")
    parsed = specialists._parse_color_request("give me a palette based on #3f7cac")
    _check(
        "hex is extracted as color, mood is None",
        parsed == {"color": "#3f7cac", "mood": None, "scheme": None},
    )

    parsed2 = specialists._parse_color_request("I want a monochromatic scheme for forest green")
    _check(
        "compound name 'forest green' is preserved, not truncated to 'green'",
        parsed2["color"] == "forest green",
    )
    _check("scheme is detected alongside the color", parsed2["scheme"] == "monochromatic")

    parsed3 = specialists._parse_color_request("rgb(200, 50, 50) analogous please")
    _check("rgb triplet extracted as color", parsed3["color"] == "rgb(200, 50, 50)")
    _check("scheme detected", parsed3["scheme"] == "analogous")


def test_parse_color_request_mood_fallback():
    print("\n=== specialists._parse_color_request: falls back to mood when no explicit color ===")
    parsed = specialists._parse_color_request("what colors would make my painting feel calm and mysterious")
    _check("color is None", parsed["color"] is None)
    _check("mood carries the full request text", "calm" in parsed["mood"] and "mysterious" in parsed["mood"])

    parsed2 = specialists._parse_color_request("give me a triadic scheme for a bold, passionate painting")
    _check("scheme word is stripped out of the mood text, not left dangling", "triad" not in parsed2["mood"])
    _check("scheme is still detected", parsed2["scheme"] == "triadic")


def test_parse_color_request_priority_color_over_mood():
    print("\n=== specialists._parse_color_request: an explicit color always wins over mood wording ===")
    parsed = specialists._parse_color_request("I want a calm painting using cerulean blue")
    _check(
        "explicit color wins even though 'calm' (a mood word) is also present",
        parsed["color"] is not None and parsed["mood"] is None,
    )


# ---------------------------------------------------------------------
# color_palette_node -- through the same fake-MCP harness
# test_new_specialists_smoke.py uses
# ---------------------------------------------------------------------


def _mcp_text_result(payload: Any) -> list:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return [{"type": "text", "text": text, "id": str(uuid.uuid4())}]


class RealColorPaletteTool:
    """
    Unlike test_new_specialists_smoke.py's FakeMCPTool (which returns a
    hand-typed canned result), this fake's .ainvoke calls the REAL
    color_tools.generate_palette() -- there's no reason to hand-fake a
    pure, deterministic function's output when the real one is just as
    fast and actually exercises the color math this specialist depends
    on. Only the MCP wire-wrapping (a list of {"type","text","id"}
    dicts) is faked, matching what a real langchain-mcp-adapters tool
    call would return.
    """

    name = "generate_color_palette"

    async def ainvoke(self, kwargs: dict):
        result = color_tools.generate_palette(
            color=kwargs.get("color"), mood=kwargs.get("mood"), scheme=kwargs.get("scheme")
        )
        return _mcp_text_result(result)


class ScriptedChatModel(BaseChatModel):
    responses: list = []
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


FAKE_CORPUS = {"documents": [], "total_documents": 0, "total_chunks": 0}


def _make_retrieve_tool():
    @tool
    def retrieve(query: str, k: int = 5) -> list:
        """Fake retrieve tool, never invoked by color_palette_node."""
        return _mcp_text_result([])

    return retrieve


def _make_generate_answer_tool():
    @tool
    def generate_answer(query: str, chunks: list) -> str:
        """Fake generate_answer tool, never invoked by color_palette_node."""
        return _mcp_text_result("")

    return generate_answer


async def _build(include_color_tool: bool = True):
    tools_by_name = {"retrieve": _make_retrieve_tool(), "generate_answer": _make_generate_answer_tool()}
    if include_color_tool:
        tools_by_name["generate_color_palette"] = RealColorPaletteTool()

    llm = ScriptedChatModel(responses=[])
    with patch("agents.specialists.build_client", return_value=object()), \
         patch("agents.specialists.load_tools_by_name", new=AsyncMock(return_value=tools_by_name)), \
         patch("agents.specialists.fetch_corpus_documents", new=AsyncMock(return_value=FAKE_CORPUS)), \
         patch("agents.specialists.ChatOllama", return_value=llm):
        node_map = await specialists.build_specialists()
    return node_map, llm


def _state(question: str) -> dict:
    return {
        "messages": [HumanMessage(content=question)],
        "route": None,
        "iteration_count": 0,
        "blocked": False,
        "injection_patterns": [],
    }


async def test_color_palette_node_explicit_color():
    print("\n=== color_palette_node: explicit hex color, all schemes ===")
    node_map, llm = await _build()
    result = await node_map["color_palette"](_state("give me a palette based on #3f7cac"))
    answer = result["messages"][0].content

    _check("zero LLM calls (color_palette is fully deterministic)", llm.call_count == 0)
    _check("message is named color_palette", result["messages"][0].name == "color_palette")
    _check("hex code appears in the answer", "#3f7cac" in answer)
    _check("a swatch image is embedded", "data:image/svg+xml;base64," in answer)
    _check(
        "all four scheme headers appear",
        all(h in answer for h in ("Monochromatic", "Analogous", "Complementary", "Triadic")),
    )


async def test_color_palette_node_single_scheme():
    print("\n=== color_palette_node: single scheme requested ===")
    node_map, llm = await _build()
    result = await node_map["color_palette"](_state("triadic scheme for forest green please"))
    answer = result["messages"][0].content

    _check("only the requested scheme header appears", "Triadic" in answer)
    _check(
        "the other three scheme headers do NOT appear",
        not any(h in answer for h in ("Monochromatic:", "Analogous:", "Complementary:")),
    )
    _check("resolved to Forest Green, not just Green", "Forest Green" in answer)


async def test_color_palette_node_mood_reverse_direction():
    print("\n=== color_palette_node: mood -> color (reverse direction) ===")
    node_map, llm = await _build()
    result = await node_map["color_palette"](_state("what colors would make my painting feel calm and peaceful"))
    answer = result["messages"][0].content

    _check("names the matched mood keywords", "calm" in answer and "peaceful" in answer)
    _check("a base color swatch is shown", "data:image/svg+xml;base64," in answer)


async def test_color_palette_node_unrecognized_input_says_so_plainly():
    print("\n=== color_palette_node: nonsense input -> plain error, no fabrication ===")
    node_map, llm = await _build()
    result = await node_map["color_palette"](_state("asdkjaslkdj qqqzzz nonsense"))
    answer = result["messages"][0].content

    _check("no swatch is fabricated for unrecognized input", "data:image/svg+xml;base64," not in answer)
    _check(
        "explains it couldn't connect the text to a color",
        "couldn't connect" in answer or "couldn't recognize" in answer,
    )


async def test_color_palette_node_degrades_when_tool_missing():
    print("\n=== color_palette_node: MCP tool unavailable -> says so, doesn't crash ===")
    node_map, llm = await _build(include_color_tool=False)
    result = await node_map["color_palette"](_state("give me a palette for #3f7cac"))
    answer = result["messages"][0].content

    _check("zero LLM calls even in the degraded path", llm.call_count == 0)
    _check("says plainly the tool is unavailable", "isn't available" in answer)


async def main():
    test_parse_color_text_hex_rgb_name()
    test_hex_rgb_roundtrip()
    test_schemes_shape_and_membership()
    test_scheme_name_normalization()
    test_color_from_mood_matches_and_averages()
    test_generate_palette_end_to_end()
    test_swatch_is_a_valid_data_uri()
    test_parse_color_request_explicit_color()
    test_parse_color_request_mood_fallback()
    test_parse_color_request_priority_color_over_mood()
    await test_color_palette_node_explicit_color()
    await test_color_palette_node_single_scheme()
    await test_color_palette_node_mood_reverse_direction()
    await test_color_palette_node_unrecognized_input_says_so_plainly()
    await test_color_palette_node_degrades_when_tool_missing()
    print("\nAll color-palette smoke tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
