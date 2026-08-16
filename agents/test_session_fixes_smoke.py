"""
Smoke test for this session's fixes -- imports the REAL functions from
agents/specialists.py, agents/guardrails.py, and agents/api.py (not
reimplementations), and exercises them against fake MCP tools / fake
messages that reproduce the exact failure shapes seen in production
logs:

  - AttributeError("'str' object has no attribute 'get'") in
    _best_personal_image_result, when unwrap_tool_result() falls back to
    a raw string.
  - TypeError("expected string or bytes-like object, got 'list'") in
    _strip_internal_markup, when an AIMessage's .content is a list of
    content blocks instead of a plain string.
  - The upload-responsiveness fix (asyncio.to_thread) and the
    "explain this image" combined-answer fix are checked structurally
    (source inspection) rather than executed, since both need a live
    Ollama/MCP stack this environment doesn't have.

Run with:
    python -m agents.test_session_fixes_smoke
"""

import asyncio
import ast
import base64
import inspect
import json
import re
import sys
import tempfile
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

import agents.guardrails as guardrails
import agents.specialists as specialists

# agents/api.py transitively imports personal_rag -> the full local_rag
# pipeline stack (chromadb, sentence-transformers, PyMuPDF, ...) at
# MODULE IMPORT TIME, just to reach two small pure-Python functions
# (_strip_internal_markup / _coerce_message_content_to_text) this file
# actually needs to test. Rather than installing that entire heavy stack
# just to satisfy an unrelated import chain, this test extracts those
# two functions' REAL source (via ast, from the actual file on disk --
# not a hand-copied reimplementation) and execs them in an isolated
# namespace with only `re` available, which is all they use. The
# upload_document / delete_thread structural checks further down read
# source text directly from the file for the same reason -- no import
# of agents.api needed at all.
_API_PY_PATH = Path(__file__).with_name("api.py")
_API_SOURCE = _API_PY_PATH.read_text()


def _load_api_helpers():
    tree = ast.parse(_API_SOURCE)
    wanted = {"_SUPERVISOR_FORCED_FINISH_NOTE_RE", "_PRODUCT_DATA_FOOTER_RE",
              "_coerce_message_content_to_text", "_strip_internal_markup"}
    segments = []
    for node in tree.body:
        names = set()
        if isinstance(node, ast.FunctionDef):
            names = {node.name}
        elif isinstance(node, ast.Assign):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        if names & wanted:
            segments.append(ast.get_source_segment(_API_SOURCE, node))
    namespace = {"re": re}
    exec("\n\n".join(segments), namespace)
    return namespace


_api_helpers = _load_api_helpers()


def _check(label: str, condition: bool):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    assert condition, label


class FakeTool:
    """
    Minimal stand-in for a langchain-mcp-adapters BaseTool: only
    `.ainvoke(args)` matters here, returning whatever raw shape a real
    MCP round trip would -- either the normal
    `[{"type": "text", "text": json.dumps(payload)}]` wrapper
    unwrap_tool_result() expects, or a deliberately malformed shape to
    reproduce the confirmed crash.
    """

    def __init__(self, fn):
        self._fn = fn

    async def ainvoke(self, args):
        return self._fn(args)


def _mcp_wrap(payload) -> list:
    return [{"type": "text", "text": json.dumps(payload)}]


# ---------------------------------------------------------------------
# _best_personal_image_result / _format_personal_image_chunk /
# _personal_image_display_block -- agents/specialists.py
# ---------------------------------------------------------------------

async def test_best_personal_image_result_normal_hit():
    print("\n=== _best_personal_image_result: normal image-origin hit ===")
    search_tool = FakeTool(lambda args: _mcp_wrap([
        {"text": "a caption", "score": 0.9, "metadata": {"original_modality": "image", "image_path": None}},
        {"text": "some pdf text", "score": 0.8, "metadata": {}},
    ]))
    result = await specialists._best_personal_image_result(search_tool, None, "thread-1", "what is this?")
    _check("returns the image-origin chunk", result is not None and result["metadata"]["original_modality"] == "image")


async def test_best_personal_image_result_falls_back_to_latest():
    print("\n=== _best_personal_image_result: falls back to latest_personal_image ===")
    search_tool = FakeTool(lambda args: _mcp_wrap([]))  # nothing relevant found
    latest_tool = FakeTool(lambda args: _mcp_wrap(
        {"text": "recent caption", "score": 1.0, "metadata": {"original_modality": "image", "image_path": None}}
    ))
    result = await specialists._best_personal_image_result(search_tool, latest_tool, "thread-1", "what is this?")
    _check("falls back to the recency lookup", result is not None and result["text"] == "recent caption")


async def test_best_personal_image_result_survives_malformed_search_response():
    print("\n=== _best_personal_image_result: malformed search response (confirmed crash repro) ===")
    # Reproduces the exact production crash: unwrap_tool_result() falls
    # back to a raw STRING when its JSON parse fails. The OLD code
    # (`chunks = unwrap_tool_result(raw_chunks) or []`, then
    # `chunk.get(...)` with no type check) would iterate this string
    # character-by-character and crash with
    # AttributeError("'str' object has no attribute 'get'").
    search_tool = FakeTool(lambda args: [{"type": "text", "text": "not valid json {{{"}])
    latest_tool = FakeTool(lambda args: _mcp_wrap(None))
    try:
        result = await specialists._best_personal_image_result(search_tool, latest_tool, "thread-1", "q")
    except AttributeError as e:
        _check(f"did NOT crash (got AttributeError: {e})", False)
        return
    _check("degrades to None instead of crashing", result is None)


async def test_best_personal_image_result_survives_non_dict_items_in_list():
    print("\n=== _best_personal_image_result: list contains non-dict items ===")
    search_tool = FakeTool(lambda args: _mcp_wrap(["oops", 123, None]))
    latest_tool = FakeTool(lambda args: _mcp_wrap(None))
    result = await specialists._best_personal_image_result(search_tool, latest_tool, "thread-1", "q")
    _check("skips non-dict list items without crashing", result is None)


def test_format_personal_image_chunk_survives_non_dict():
    print("\n=== _format_personal_image_chunk: non-dict input (defense in depth) ===")
    result = specialists._format_personal_image_chunk("not a dict")
    _check("returns a plain fallback string instead of crashing", isinstance(result, str) and result)


def test_personal_image_display_block_and_data_uri_roundtrip():
    print("\n=== _personal_image_display_block: real file round trip ===")
    with tempfile.TemporaryDirectory() as d:
        img_path = Path(d) / "test.png"
        # A tiny valid-enough PNG header is not required here --
        # _read_local_image_data_uri only reads bytes and base64-encodes
        # them, it never decodes/validates image content.
        raw_bytes = b"\x89PNG\r\n\x1a\nfake-but-nonempty-bytes"
        img_path.write_bytes(raw_bytes)

        chunk = {"text": "a test caption", "metadata": {"image_path": str(img_path)}}
        block = specialists._personal_image_display_block(chunk)
        _check("produces a markdown image block", block.startswith("![a test caption]("))
        _check("embeds a data: URI", "data:image/png;base64," in block)

        expected_b64 = base64.b64encode(raw_bytes).decode("ascii")
        _check("base64 payload matches the file's real bytes", expected_b64 in block)

        full = specialists._format_personal_image_chunk(chunk)
        _check("_format_personal_image_chunk repeats the caption below the image", full.count("a test caption") == 2)

        # display_block (used when combining with a generated answer)
        # deliberately does NOT repeat the caption a second time.
        _check("_personal_image_display_block does not repeat the caption", block.count("a test caption") == 1)


def test_personal_image_display_block_missing_file_degrades_to_caption():
    print("\n=== _personal_image_display_block: missing file degrades gracefully ===")
    chunk = {"text": "orphaned caption", "metadata": {"image_path": "/no/such/file.png"}}
    block = specialists._personal_image_display_block(chunk)
    _check("falls back to caption-only text", block == "*orphaned caption*")


# ---------------------------------------------------------------------
# _coerce_message_content_to_text / _strip_internal_markup --
# agents/api.py, agents/guardrails.py
# ---------------------------------------------------------------------

def test_api_strip_internal_markup_survives_list_content():
    print("\n=== agents.api._strip_internal_markup: list-shaped content (confirmed crash repro) ===")
    # Exact shape observed in production: a create_react_agent's final
    # AIMessage with .content as a list of content blocks, on an
    # Arabic-language turn. The OLD function signature was
    # `_strip_internal_markup(content: str)` with a bare `.sub(...)`
    # call -- this would raise
    # TypeError("expected string or bytes-like object, got 'list'").
    list_content = [{"type": "text", "text": "هذا هو الجواب على سؤالك"}]
    strip_fn = _api_helpers["_strip_internal_markup"]
    try:
        result = strip_fn(list_content)
    except TypeError as e:
        _check(f"did NOT crash (got TypeError: {e})", False)
        return
    _check("normalizes list content to plain text", result == "هذا هو الجواب على سؤالك")


def test_api_strip_internal_markup_still_strips_forced_finish_note():
    print("\n=== agents.api._strip_internal_markup: still strips the supervisor note on plain strings ===")
    content = "[All specialists already tried this turn, none confirmed FINISH]\n\nThe real answer."
    strip_fn = _api_helpers["_strip_internal_markup"]
    result = strip_fn(content)
    _check("supervisor note removed, real answer kept", result == "The real answer.")


async def test_output_guard_normalizes_list_content_in_place():
    print("\n=== output_guard_node: normalizes list content for a real turn ===")
    state = {
        "messages": [
            HumanMessage(content="ما هو الطلاء الزيتي؟"),
            AIMessage(
                content=[{"type": "text", "text": "الطلاء الزيتي هو تقنية رسم."}],
                name="retrieval_qa",
                id="msg-1",
            ),
        ],
        "route": "FINISH",
        "iteration_count": 1,
        "blocked": False,
        "injection_patterns": [],
    }
    result = await guardrails.output_guard_node(state)
    _check("output_guard produced a replacement message", "messages" in result and len(result["messages"]) == 1)
    replacement = result["messages"][0]
    _check("replacement content is now a plain string", isinstance(replacement.content, str))
    _check("replacement keeps the same message id (in-place replace, not append)", replacement.id == "msg-1")
    _check("text content preserved correctly", replacement.content == "الطلاء الزيتي هو تقنية رسم.")

    # Confirm the now-normalized message would ALSO survive
    # agents.api._strip_internal_markup downstream without any special
    # handling -- the whole point of fixing it at the source.
    final = _api_helpers["_strip_internal_markup"](replacement.content)
    _check("downstream _strip_internal_markup handles it with zero special-casing", final == replacement.content)


async def test_output_guard_untouched_on_clean_string_turn():
    print("\n=== output_guard_node: no-op on an already-clean string turn ===")
    state = {
        "messages": [
            HumanMessage(content="What is glazing?"),
            AIMessage(content="Glazing is a thin transparent layer of paint.", name="retrieval_qa", id="msg-2"),
        ],
        "route": "FINISH",
        "iteration_count": 1,
        "blocked": False,
        "injection_patterns": [],
    }
    result = await guardrails.output_guard_node(state)
    _check("returns {} (no state update) for an already-clean turn", result == {})


# ---------------------------------------------------------------------
# Structural checks (can't execute without a live Ollama/MCP stack) --
# the responsiveness fix and the "real answer, not an echo" fix
# ---------------------------------------------------------------------

def _function_source_text(module_source: str, func_name: str) -> str:
    """Pull one top-level function's source text directly out of a
    module's raw source, via ast -- used for the structural checks
    below so they don't require importing agents.api at all (see this
    file's own top-of-file comment for why)."""
    tree = ast.parse(module_source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            return ast.get_source_segment(module_source, node)
    raise LookupError(f"{func_name} not found")


def test_upload_endpoint_offloads_blocking_ingest_to_a_thread():
    print("\n=== upload_document: ingest_upload is offloaded via asyncio.to_thread ===")
    source = _function_source_text(_API_SOURCE, "upload_document")
    _check(
        "calls personal_rag.ingest_upload through asyncio.to_thread, not directly",
        "asyncio.to_thread(\n            personal_rag.ingest_upload" in source
        or "asyncio.to_thread(personal_rag.ingest_upload" in source,
    )
    _check(
        "does not call personal_rag.ingest_upload directly (blocking the event loop)",
        "= personal_rag.ingest_upload(" not in source,
    )


def test_delete_thread_offloads_blocking_cleanup_to_a_thread():
    print("\n=== delete_thread: personal_rag cleanup is offloaded via asyncio.to_thread ===")
    source = _function_source_text(_API_SOURCE, "delete_thread")
    _check(
        "calls personal_rag.delete_thread_data through asyncio.to_thread",
        "asyncio.to_thread(personal_rag.delete_thread_data" in source,
    )


def test_personal_docs_and_image_qa_call_generate_tool_on_image_hit():
    print("\n=== personal_docs_node / image_qa_node: real answer, not a caption echo ===")
    source = inspect.getsource(specialists)
    # Both nodes' image_hit branches should call generate_tool.ainvoke
    # rather than immediately wrapping _format_personal_image_chunk's
    # output as the whole answer (the old, "echoes the caption verbatim"
    # behavior a live report flagged as feeling unresponsive to a real
    # question like "explain this image").
    occurrences = source.count("raw_image_answer = await generate_tool.ainvoke({\"query\": question, \"chunks\": [")
    _check("generate_tool is called on the image_hit branch in both nodes", occurrences == 2)


def run_all():
    sync_tests = [
        test_format_personal_image_chunk_survives_non_dict,
        test_personal_image_display_block_and_data_uri_roundtrip,
        test_personal_image_display_block_missing_file_degrades_to_caption,
        test_api_strip_internal_markup_survives_list_content,
        test_api_strip_internal_markup_still_strips_forced_finish_note,
        test_upload_endpoint_offloads_blocking_ingest_to_a_thread,
        test_delete_thread_offloads_blocking_cleanup_to_a_thread,
        test_personal_docs_and_image_qa_call_generate_tool_on_image_hit,
    ]
    async_tests = [
        test_best_personal_image_result_normal_hit,
        test_best_personal_image_result_falls_back_to_latest,
        test_best_personal_image_result_survives_malformed_search_response,
        test_best_personal_image_result_survives_non_dict_items_in_list,
        test_output_guard_normalizes_list_content_in_place,
        test_output_guard_untouched_on_clean_string_turn,
    ]

    failures = []
    for t in sync_tests:
        try:
            t()
        except AssertionError:
            failures.append(t.__name__)

    async def _run_async():
        for t in async_tests:
            try:
                await t()
            except AssertionError:
                failures.append(t.__name__)

    asyncio.run(_run_async())

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED: {len(failures)} test(s): {failures}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")


if __name__ == "__main__":
    run_all()
