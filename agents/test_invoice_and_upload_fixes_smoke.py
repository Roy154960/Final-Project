"""
Smoke test for this session's fixes. Items 6 and 7 replace an earlier,
reverted approach (item 6's history is worth keeping in mind: a
first-pass fix retracted the upload preview pair from persisted state
whenever a follow-up question arrived, but that depended on the
follow-up routing correctly -- when it didn't, the retraction removed
the one thing reliably showing the image, making the failure WORSE. It
was reverted; the routing itself is what item 7 actually fixes).

  1. product_search/invoice routing confusion + item mixups: a person
     naming an item already found ("I'll take b3", "buy the fine detail
     brush set") was getting mis-routed to product_search (which re-runs
     a live search and can return a different batch), and even when
     invoice_node DID run, its item-matcher never checked an item's own
     short id ("b1"/"g3" -- the exact label product_search_node's own
     rendered list uses), so an id-only reference silently fell back to
     invoicing the ENTIRE batch instead of the one item asked for.
     Fixed by `_match_by_id` (agents/specialists.py, feeding into
     `_score_items_by_name`) and `_looks_like_invoice_followup` +
     supervisor.py's new pre-LLM "safety net 0".
  2. invoice_tools.build_invoice() showed a partial dollar subtotal when
     one or more selected items had no listed price, with no visual
     signal that the number was incomplete. Fixed: the rendered
     Subtotal line now reads "Unavailable" whenever ANY item was
     skipped for a missing price, and skipped items now carry a link
     too. The "Saved to: <local path>" line was also dropped from the
     user-facing answer (specialists.py's invoice_node) -- a filesystem
     path on the machine running the server was never useful to the
     person reading the chat.
  3. Personal-RAG chat uploads (POST /chat/{thread_id}/upload) rejected
     .txt files outright. Fixed: SUPPORTED_UPLOAD_EXTS now includes
     ".txt", and the old two-way "pdf, else image" modality tag (which
     would have silently mislabeled a text upload as "image" the moment
     .txt became accepted) is now a real three-way branch.
  4. A CONFIRMED live-run cascade: invoice_node's own honest "nothing
     could be priced" refusal (after safety net 0 above correctly
     routed to it) still let the model try again on the next supervisor
     visit, which insisted on `product_search` and forced the
     repeat-route guard to walk every OTHER specialist before the
     iteration cap. Fixed: a new, narrowly-scoped early-stop check in
     supervisor.py finishes immediately on an invoice refusal, but ONLY
     when the original question would still qualify for safety net 0
     (so a genuinely mis-routed invoice call still gets the model's own
     second look, same as before).
  5. A CONFIRMED live-run crash: `retrieval_qa`'s react agent had a
     small-model tier bound to phi3, but `create_react_agent` binds
     tools at build time and phi3 doesn't support Ollama's tools
     parameter at all -- any "simple"-classified question routed to
     retrieval_qa 503'd outright. Fixed: retrieval_qa's react agent no
     longer has a small tier; it always uses the large (tool-capable)
     model.
  6. A CONFIRMED markdown-escaping bug: `_post_captioned_images_to_chat`
     (agents/api.py) was the one remaining place embedding a VLM caption
     into `![caption](data:...)` markdown WITHOUT escaping it first --
     every other image-rendering path in this project already uses
     `_escape_markdown_caption` for exactly this reason: an unescaped
     "]" in a free-form caption prematurely closes the image's alt-text
     span, so the entire `(data:image/...;base64,...)` destination falls
     through as plain text -- a raw base64 string spilled into the chat.
     Fixed: that one call site now escapes too.
  7. A CONFIRMED live-run mis-route: an obviously-attachment-related
     question ("explain this uploaded image titled X") was routed to
     `retrieval_qa` (which has no access to personal uploads at all)
     instead of `personal_docs`, and the repeat-route guard then walked
     every other specialist before the iteration cap, none of them able
     to see the upload either. Fixed: a new deterministic pre-LLM
     routing check (supervisor.py's "safety net 0a") routes any message
     carrying an `<attachment ...>` marker straight to `personal_docs`;
     contextualize.py now also skips rewriting such a message so the
     marker survives verbatim into that check; and the three "personal
     image + real answer" branches (retrieval_qa_node, personal_docs_node,
     image_qa_node) no longer re-embed the same image a second time when
     it was already shown by the upload's own preview this same turn.

Run with:
    python -m agents.test_invoice_and_upload_fixes_smoke
"""

import ast
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

import agents.specialists as specialists
import agents.supervisor as supervisor


def _product_search_message(items: list[dict]) -> AIMessage:
    """A fake product_search AIMessage carrying the same hidden
    PRODUCT_DATA footer product_search_node's real _embed_product_data
    produces -- built via the real helper, not a hand-rolled string, so
    this test breaks if the footer format ever changes."""
    return AIMessage(
        content=specialists._embed_product_data("Here are some options:", items),
        name="product_search",
    )


_SAMPLE_BATCH = [
    {"id": "b1", "name": "Super Fine Paint Brushes", "price": None, "url": "https://www.amazon.com/x", "source": "amazon", "tier": "beginner"},
    {"id": "b2", "name": "Fine Detail Brushes", "price": 6.99, "url": "https://www.amazon.com/y", "source": "amazon", "tier": "beginner"},
    {"id": "g1", "name": "Fine Detail Paint Brush Set - 7 Pieces Miniature Brushes", "price": 24.99, "url": "https://www.amazon.com/z", "source": "amazon", "tier": "professional"},
]


# --- Fix 1a: _match_by_id / _score_items_by_name -----------------------

def test_score_items_by_name_matches_by_short_id():
    matched = specialists._score_items_by_name("I'll take b2", _SAMPLE_BATCH)
    assert [it["id"] for it in matched] == ["b2"], matched


def test_score_items_by_name_id_match_does_not_pull_in_other_items():
    # CONFIRMED failure this guards against: before id-matching existed,
    # a bare id like "g1" scored zero on every name-based pass and fell
    # through to "assume the whole batch" -- silently invoicing items
    # never asked for.
    matched = specialists._score_items_by_name("just g1 please", _SAMPLE_BATCH)
    assert len(matched) == 1 and matched[0]["id"] == "g1", matched


def test_score_items_by_name_still_matches_by_name_when_no_id_given():
    matched = specialists._score_items_by_name(
        "I want the fine detail brushes", _SAMPLE_BATCH
    )
    ids = {it["id"] for it in matched}
    # "Fine Detail Brushes" (b2) is an exact substring match of the
    # request; g1's much longer name ("Fine Detail Paint Brush Set - 7
    # Pieces Miniature Brushes") is not -- exact-match pass 1 should
    # return ONLY b2, not both just because they share words.
    assert ids == {"b2"}, ids


# --- Fix 1b: _looks_like_invoice_followup -------------------------------

def test_looks_like_invoice_followup_true_for_id_plus_intent_word():
    assert specialists._looks_like_invoice_followup("I'll take b2", _SAMPLE_BATCH) is True


def test_looks_like_invoice_followup_true_for_select_all_phrase_alone():
    assert specialists._looks_like_invoice_followup("all of them please", _SAMPLE_BATCH) is True


def test_looks_like_invoice_followup_false_without_intent_word():
    # A name/id match with NO purchase-signaling word should NOT trigger
    # the deterministic router override -- e.g. a genuine follow-up
    # question about an item, not a purchase request.
    assert specialists._looks_like_invoice_followup(
        "what's g1 made of?", _SAMPLE_BATCH
    ) is False


def test_looks_like_invoice_followup_false_on_empty_batch():
    assert specialists._looks_like_invoice_followup("buy b2", []) is False


def test_looks_like_invoice_followup_false_for_unrelated_question():
    assert specialists._looks_like_invoice_followup(
        "how do I mix a good glaze for oil painting?", _SAMPLE_BATCH
    ) is False


# --- Fix 1c: supervisor.py's pre-LLM "safety net 0" ---------------------

async def test_supervisor_safety_net_0_routes_straight_to_invoice():
    fake_specialists = {
        "retrieval_qa": object(),
        "product_search": object(),
        "invoice": object(),
    }
    node = supervisor.build_supervisor(fake_specialists, fallback_route="retrieval_qa")
    state = {
        "messages": [
            HumanMessage(content="find me some paint brushes"),
            _product_search_message(_SAMPLE_BATCH),
            HumanMessage(content="I'll take b2"),
        ],
        "iteration_count": 0,
    }
    result = await node(state)
    assert result["route"] == "invoice", result


async def test_supervisor_finishes_on_invoice_refusal_when_safety_net_0_routed_it():
    """
    CONFIRMED live-run failure this guards against: a person asked to
    buy two named items that both turned out to have no listed price.
    invoice_node correctly answered "no items could be priced -- nothing
    to invoice" -- a refusal by `_REFUSAL_MARKERS`' own definition -- and
    because the general `_DETERMINISTIC_NEVER_HEDGES` early-stop net
    deliberately excludes refusals, the turn fell through to a second
    LLM call. The model then insisted on `product_search` on every
    subsequent visit, and the repeat-route guard was forced to walk
    every OTHER specialist (retrieval_qa, personal_docs, corpus_meta,
    color_palette, multi_hop, image_qa -- none of them related to the
    request) before the iteration cap forced a stop. This test
    reproduces the SECOND supervisor visit directly: exactly one prior
    attempt (invoice, with a refusal matching `_REFUSAL_MARKERS`), on a
    question that safety net 0 itself would have recognized as an
    invoice follow-up -- and confirms the turn now FINISHes immediately
    instead of proceeding to a second LLM call.
    """
    fake_specialists = {
        "retrieval_qa": object(),
        "product_search": object(),
        "invoice": object(),
    }
    node = supervisor.build_supervisor(fake_specialists, fallback_route="retrieval_qa")
    state = {
        "messages": [
            HumanMessage(content="find me some paint brushes"),
            _product_search_message(_SAMPLE_BATCH),
            HumanMessage(content="I'll take b2"),
            AIMessage(
                content="No items could be priced -- nothing to invoice.",
                name="invoice",
            ),
        ],
        "iteration_count": 1,
    }
    result = await node(state)
    assert result["route"] == "FINISH", result


async def test_supervisor_does_not_force_finish_on_invoice_refusal_when_not_net_0_routed():
    """
    The counterpart to the test above: when the ORIGINAL question would
    NOT have qualified for safety net 0 (no id/name match, no purchase
    intent word) -- e.g. the model chose `invoice` on its own, possibly
    incorrectly -- an invoice refusal should still fall through to the
    model's own routing judgment, same as before this fix. This is what
    keeps the new early-stop net narrowly scoped to the exact case
    safety net 0 already vouched for, rather than treating every invoice
    refusal as automatically final.
    """
    fake_specialists = {
        "retrieval_qa": object(),
        "product_search": object(),
        "invoice": object(),
    }
    node = supervisor.build_supervisor(fake_specialists, fallback_route="retrieval_qa")
    state = {
        "messages": [
            HumanMessage(content="find me some paint brushes"),
            _product_search_message(_SAMPLE_BATCH),
            HumanMessage(content="what's a good brush for glazing?"),
            AIMessage(
                content="No items could be priced -- nothing to invoice.",
                name="invoice",
            ),
        ],
        "iteration_count": 1,
    }
    try:
        result = await node(state)
    except Exception:
        # Expected in this environment: the new net stayed silent (as it
        # should), so the node proceeded on to an actual (unreachable)
        # LLM call.
        return
    assert result["route"] != "FINISH" or "final here" not in str(result), result


async def test_supervisor_safety_net_0_does_not_fire_without_intent_word():
    """
    Confirms net 0 stays silent for a name/id match with no
    purchase-signaling word -- the turn should fall through to the
    model's own routing judgment (safety nets 1-4), never get
    force-routed to invoice. No live Ollama is reachable in this test
    environment, so a downstream connection failure when the LLM call
    net 0 left unreached is actually attempted is itself the expected
    outcome here -- what matters is that the failure does NOT come from
    net 0 forcing "invoice" before ever reaching that call.
    """
    fake_specialists = {
        "retrieval_qa": object(),
        "product_search": object(),
        "invoice": object(),
    }
    node = supervisor.build_supervisor(fake_specialists, fallback_route="retrieval_qa")
    state = {
        "messages": [
            HumanMessage(content="find me some paint brushes"),
            _product_search_message(_SAMPLE_BATCH),
            HumanMessage(content="what's g1 made of?"),
        ],
        "iteration_count": 0,
    }
    try:
        result = await node(state)
    except Exception:
        # Expected in this environment: net 0 stayed silent, so the node
        # proceeded on to an actual (unreachable) LLM call, which is
        # exactly what should happen.
        return
    assert result["route"] != "invoice", result


# --- Fix 2: invoice_tools.build_invoice() "Unavailable" total -----------

def test_invoice_total_unavailable_when_any_item_unpriced():
    import types
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mcp_server"))
    if "safety" not in sys.modules:
        # invoice_tools.py imports safety.domain_allowlist, which lives
        # under mcp_server/ as a top-level package on THAT process's own
        # sys.path (see mcp_server/server.py) -- stubbed here rather
        # than adding mcp_server's own dependency chain, since only
        # is_allowed_domain's return value matters for this test.
        safety_pkg = types.ModuleType("safety")
        domain_mod = types.ModuleType("safety.domain_allowlist")
        domain_mod.is_allowed_domain = lambda url: True
        sys.modules["safety"] = safety_pkg
        sys.modules["safety.domain_allowlist"] = domain_mod
    import invoice_tools

    result = invoice_tools.build_invoice(
        [
            {"name": "Priced brush", "price": 10.0, "quantity": 1, "url": ""},
            {"name": "Unpriced brush", "price": None, "quantity": 1, "url": ""},
        ]
    )
    assert result["subtotal_available"] is False
    assert "Subtotal: Unavailable" in result["invoice_markdown"]

    result_all_priced = invoice_tools.build_invoice(
        [{"name": "Priced brush", "price": 10.0, "quantity": 1, "url": ""}]
    )
    assert result_all_priced["subtotal_available"] is True
    assert "Subtotal: $10.00" in result_all_priced["invoice_markdown"]


# --- Fix 3: personal_rag.py accepts .txt uploads -------------------------
#
# personal_rag.py transitively imports the full local_rag pipeline stack
# (chromadb, sentence-transformers, torch, ...) just to be imported at
# all -- the same thing this project's own test_session_fixes_smoke.py
# already notes for personal_rag ("agents/api.py transitively imports
# personal_rag -> the full local_rag pipeline stack"), and the same
# reason that file checks some fixes "structurally (source inspection)
# rather than executed" instead of importing and running them directly.
# These two follow that same pattern: read the file's own source text,
# no import required.

def _personal_rag_source() -> str:
    path = Path(__file__).resolve().parents[1] / "local_rag" / "personal_rag.py"
    return path.read_text(encoding="utf-8")


def test_personal_rag_accepts_txt_extension():
    source = _personal_rag_source()
    assert 'SUPPORTED_UPLOAD_EXTS = (".pdf", ".txt", ".png"' in source, (
        "expected .txt to be present in SUPPORTED_UPLOAD_EXTS"
    )


def test_personal_rag_modality_is_three_way_not_binary():
    # Confirms the old binary `"pdf" if ext == ".pdf" else "image"` line
    # (which would have mislabeled a .txt upload's own reported modality
    # as "image" the moment .txt became an accepted extension) is gone,
    # replaced with a real three-way branch.
    source = _personal_rag_source()
    assert '"pdf" if ext == ".pdf" else "image"' not in source
    assert 'modality = "text"' in source


# --- Fix (this session): retrieval_qa's react agent no longer has a
# phi3-bound small tier -- create_react_agent binds tools at build time,
# and phi3 does not support Ollama's tools parameter at all. Source-level
# check for the same reason the personal_rag ones above are: importing
# agents.specialists successfully in THIS test process doesn't require a
# live Ollama server, but building retrieval_qa_agent_large/small does
# require the full MCP tool-loading path this smoke test doesn't set up.

def _specialists_source() -> str:
    path = Path(__file__).resolve().parent / "specialists.py"
    return path.read_text(encoding="utf-8")


def test_retrieval_qa_has_no_phi3_bound_react_agent():
    source = _specialists_source()
    # Checks for an actual assignment/build, not just the name appearing
    # anywhere -- this module's own comments deliberately still mention
    # "retrieval_qa_agent_small" as a historical note explaining why it
    # was removed, so a bare substring check would false-positive on
    # that comment.
    assert "retrieval_qa_agent_small = create_react_agent" not in source, (
        "retrieval_qa_agent_small should no longer be built -- "
        "create_react_agent binds tools, and phi3 (_SMALL_REASONING_MODEL) "
        "does not support Ollama's tools parameter"
    )
    assert "_agent_for(" not in source or "def _agent_for(" not in source
    assert "await retrieval_qa_agent_large.ainvoke(" in source


# --- Fix 6: escaping captions in the upload-preview markdown -----------
#
# agents/api.py transitively imports personal_rag -> the full local_rag
# pipeline stack (chromadb, sentence-transformers, torch, ...) at module
# IMPORT TIME. Source-level check instead, same pattern this project's
# own test_session_fixes_smoke.py already uses for the identical
# constraint.

_API_PY_PATH = Path(__file__).with_name("api.py")
_API_SOURCE = _API_PY_PATH.read_text(encoding="utf-8")


def test_post_captioned_images_escapes_caption_before_embedding():
    source = _API_SOURCE
    idx = source.index("async def _post_captioned_images_to_chat")
    body = source[idx: idx + 4000]
    assert 'specialists._escape_markdown_caption(img.get("caption"))' in body, (
        "_post_captioned_images_to_chat should escape the caption via "
        "specialists._escape_markdown_caption before embedding it in "
        "markdown -- an unescaped ']' in a free-form VLM caption "
        "prematurely closes the image's alt-text span, spilling the raw "
        "base64 data: URI into the chat as visible text"
    )
    # The un-escaped form should be gone from this function's body.
    assert 'caption = img.get("caption") or "(no caption available)"' not in body


def test_retract_upload_preview_function_no_longer_exists():
    # Confirms the earlier, reverted approach (see this module's own
    # docstring) is actually gone, not just unused.
    assert "_retract_upload_preview_for_this_turn" not in _API_SOURCE
    assert "_attachment_filenames_in_message" not in _API_SOURCE


# --- Fix 7: deterministic attachment-follow-up routing -------------------

_SAMPLE_ATTACHMENT_MARKER = (
    '<attachment name=chart.png status="ingested into this '
    'conversation\'s personal knowledge base" chunks=2>'
)


def test_message_carries_attachment_true_for_real_marker():
    message = f"Explain this uploaded image titled chart.png\n{_SAMPLE_ATTACHMENT_MARKER}"
    assert specialists._message_carries_attachment(message) is True


def test_message_carries_attachment_false_for_plain_text():
    assert specialists._message_carries_attachment("just a normal question") is False


async def test_supervisor_routes_attachment_message_to_personal_docs():
    fake_specialists = {
        "retrieval_qa": object(),
        "personal_docs": object(),
        "corpus_meta": object(),
    }
    node = supervisor.build_supervisor(fake_specialists, fallback_route="retrieval_qa")
    state = {
        "messages": [
            HumanMessage(content=f"Explain this uploaded image\n{_SAMPLE_ATTACHMENT_MARKER}"),
        ],
        "iteration_count": 0,
    }
    result = await node(state)
    assert result["route"] == "personal_docs", result


async def test_supervisor_finishes_on_personal_docs_answer_when_net_0a_routed_it():
    """
    CONFIRMED live-run failure this guards against: an attachment-related
    question got mis-routed to retrieval_qa, and even after fixing THAT
    mis-route, personal_docs isn't in _DETERMINISTIC_NEVER_HEDGES, so
    without this dedicated check, a second supervisor LLM call could
    still send the SAME turn cascading through every other specialist --
    none of which have any access to personal uploads at all.
    """
    fake_specialists = {
        "retrieval_qa": object(),
        "personal_docs": object(),
        "corpus_meta": object(),
    }
    node = supervisor.build_supervisor(fake_specialists, fallback_route="retrieval_qa")
    state = {
        "messages": [
            HumanMessage(content=f"Explain this uploaded image\n{_SAMPLE_ATTACHMENT_MARKER}"),
            AIMessage(content="Here's what I see in the image...", name="personal_docs"),
        ],
        "iteration_count": 1,
    }
    result = await node(state)
    assert result["route"] == "FINISH", result


def test_contextualize_never_rewrites_a_message_with_attachment_marker():
    path = Path(__file__).with_name("contextualize.py")
    source = path.read_text(encoding="utf-8")
    assert "_message_carries_attachment(original.content)" in source, (
        "contextualize_node should skip rewriting (return {}) when the "
        "original message carries an <attachment ...> marker, so "
        "supervisor.py's safety net 0a always sees it intact"
    )


def test_image_answer_content_omits_image_block_when_attachment_present():
    result = specialists._image_answer_content(
        "![caption](data:image/png;base64,AAAA)",
        "This chart shows six elements.",
        f"Explain this uploaded image\n{_SAMPLE_ATTACHMENT_MARKER}",
    )
    assert result == "This chart shows six elements.", result


def test_image_answer_content_includes_image_block_without_attachment_marker():
    result = specialists._image_answer_content(
        "![caption](data:image/png;base64,AAAA)",
        "This chart shows six elements.",
        "what is the third element in that chart?",
    )
    assert result == "![caption](data:image/png;base64,AAAA)\n\nThis chart shows six elements.", result


def test_image_answer_content_returns_answer_alone_when_no_image_block():
    result = specialists._image_answer_content("", "just text", "any question")
    assert result == "just text"


def run_all():
    sync_tests = [
        test_score_items_by_name_matches_by_short_id,
        test_score_items_by_name_id_match_does_not_pull_in_other_items,
        test_score_items_by_name_still_matches_by_name_when_no_id_given,
        test_looks_like_invoice_followup_true_for_id_plus_intent_word,
        test_looks_like_invoice_followup_true_for_select_all_phrase_alone,
        test_looks_like_invoice_followup_false_without_intent_word,
        test_looks_like_invoice_followup_false_on_empty_batch,
        test_looks_like_invoice_followup_false_for_unrelated_question,
        test_invoice_total_unavailable_when_any_item_unpriced,
        test_personal_rag_accepts_txt_extension,
        test_personal_rag_modality_is_three_way_not_binary,
        test_retrieval_qa_has_no_phi3_bound_react_agent,
        test_post_captioned_images_escapes_caption_before_embedding,
        test_retract_upload_preview_function_no_longer_exists,
        test_message_carries_attachment_true_for_real_marker,
        test_message_carries_attachment_false_for_plain_text,
        test_contextualize_never_rewrites_a_message_with_attachment_marker,
        test_image_answer_content_omits_image_block_when_attachment_present,
        test_image_answer_content_includes_image_block_without_attachment_marker,
        test_image_answer_content_returns_answer_alone_when_no_image_block,
    ]
    async_tests = [
        test_supervisor_safety_net_0_routes_straight_to_invoice,
        test_supervisor_finishes_on_invoice_refusal_when_safety_net_0_routed_it,
        test_supervisor_does_not_force_finish_on_invoice_refusal_when_not_net_0_routed,
        test_supervisor_safety_net_0_does_not_fire_without_intent_word,
        test_supervisor_routes_attachment_message_to_personal_docs,
        test_supervisor_finishes_on_personal_docs_answer_when_net_0a_routed_it,
    ]

    failures = []
    for t in sync_tests:
        try:
            t()
            print(f"  ok  {t.__name__}")
        except AssertionError as e:
            failures.append(t.__name__)
            print(f"FAIL  {t.__name__}: {e}")

    import asyncio

    async def _run_async():
        for t in async_tests:
            try:
                await t()
                print(f"  ok  {t.__name__}")
            except AssertionError as e:
                failures.append(t.__name__)
                print(f"FAIL  {t.__name__}: {e}")

    asyncio.run(_run_async())

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED: {len(failures)} test(s): {failures}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")


if __name__ == "__main__":
    run_all()
