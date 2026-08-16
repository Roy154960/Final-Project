"""
Smoke tests for the three new tool modules backing this server's newer
tools (retrieve_images, search_painting_online, search_art_supplies,
generate_invoice): mcp_server/web_tools.py, image_tools.py,
invoice_tools.py.

Same philosophy as the rest of this project's smoke tests -- no live
network, no real Ollama, no real CLIP weights. Network calls are faked
by monkeypatching `ddgs.DDGS`/`requests.get` at the point this module's
functions call them, so the actual parsing/filtering/error-handling logic
inside web_tools.py runs for real; only the external service itself is
stubbed out.

Run with:
    python mcp_server/test_new_tools_smoke.py
    (or, from the project root: python -m mcp_server.test_new_tools_smoke)
"""

import base64
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Same sys.path pattern server.py itself uses, so `import web_tools` etc.
# (bare, sibling-module imports) resolve when this file is run directly.
_MCP_SERVER_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MCP_SERVER_DIR.parent
for p in (_MCP_SERVER_DIR, _PROJECT_ROOT / "local_rag", _PROJECT_ROOT):
    if str(p) not in sys.path and p.exists():
        sys.path.insert(0, str(p))

import image_tools  # noqa: E402
import invoice_tools  # noqa: E402
import web_tools  # noqa: E402


def _check(label: str, condition: bool):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    assert condition, label


# ---------------------------------------------------------------------
# web_tools.py
# ---------------------------------------------------------------------


def test_clean_painting_query_strips_question_wrappers():
    print("\n=== web_tools._clean_painting_query: strips common question phrasing ===")
    cases = [
        ("Tell me about the Mona Lisa", "the Mona Lisa"),
        ("Who painted Starry Night?", "Starry Night"),
        ("What is Guernica?", "Guernica"),
        ("Mona Lisa", "Mona Lisa"),  # already clean -- passes through unchanged
    ]
    for raw, expected in cases:
        _check(f"{raw!r} -> {expected!r}", web_tools._clean_painting_query(raw) == expected)
    _check(
        "an input that's ONLY a wrapper (nothing left after stripping) falls back to the "
        "original rather than becoming an empty query",
        web_tools._clean_painting_query("tell me about") == "tell me about",
    )


def test_search_famous_painting_uses_the_cleaned_query_not_the_raw_question():
    print("\n=== web_tools.search_famous_painting: passes the CLEANED name to every lookup ===")
    # Reproduces the confirmed live-run failure: painting_lookup_node
    # passed the full question "Tell me about the Mona Lisa" straight
    # through, and Wikipedia's search API resolved it to "Mona Lisa
    # Smile" (the film) instead of the painting -- the extra
    # question-wrapper words were noise the relevance ranking had no way
    # to know wasn't part of the subject. This test doesn't (can't, in
    # this sandbox) verify Wikipedia's actual search ranking; it verifies
    # the FIX mechanism itself: every lookup this function makes uses the
    # cleaned query, not the raw, noisy one.
    calls = {"summary_titles": [], "best_title_queries": [], "ddgs_queries": []}

    def _fake_summary(title):
        calls["summary_titles"].append(title)
        if title == "the Mona Lisa":
            return None  # exact-title lookup correctly fails for a non-exact title
        return {"title": "Mona Lisa", "extract": "A portrait by Leonardo da Vinci.", "url": "https://en.wikipedia.org/wiki/Mona_Lisa"}

    def _fake_best_title(query):
        calls["best_title_queries"].append(query)
        return "Mona Lisa"

    def _fake_ddgs(query, max_results):
        calls["ddgs_queries"].append(query)
        return []

    with patch("web_tools.wikipedia_summary", side_effect=_fake_summary), \
         patch("web_tools.wikipedia_best_title", side_effect=_fake_best_title), \
         patch("web_tools._ddgs_text", side_effect=_fake_ddgs):
        result = web_tools.search_famous_painting("Tell me about the Mona Lisa")

    _check(
        "the direct summary attempt used the CLEANED name, not the raw question",
        calls["summary_titles"][0] == "the Mona Lisa",
    )
    _check(
        "'tell me about' never appears in any lookup this function made",
        all("tell me about" not in t.lower() for t in calls["summary_titles"])
        and all("tell me about" not in q.lower() for q in calls["best_title_queries"])
        and all("tell me about" not in q.lower() for q in calls["ddgs_queries"]),
    )
    _check(
        "the fallback search-API call also used the cleaned name",
        calls["best_title_queries"] == ["the Mona Lisa"],
    )
    _check(
        "the resolved title's summary made it into the final result",
        result["summary"] is not None and "da Vinci" in result["summary"],
    )
    _check("the 'query' field still reports the ORIGINAL question, for transparency", result["query"] == "Tell me about the Mona Lisa")


def test_price_regex_extracts_from_snippet():
    print("\n=== web_tools: price regex pulls a $NN.NN out of a snippet ===")
    m = web_tools._PRICE_RE.search("Best-selling brush set, only $24.99 today, free shipping!")
    _check("price matched", m is not None)
    _check("price value is correct", m.group(1) == "24.99")


def test_price_regex_ignores_snippet_with_no_price():
    print("\n=== web_tools: price regex finds nothing when there's no $ amount ===")
    m = web_tools._PRICE_RE.search("A lovely set of brushes, customer favorite.")
    _check("no match when no price-shaped text is present", m is None)


def test_search_famous_painting_degrades_gracefully_with_no_network():
    print("\n=== web_tools.search_famous_painting: no network -> empty-but-valid result, no raise ===")
    import requests

    with patch("web_tools.requests.get", side_effect=requests.ConnectionError("no network in this test")), \
         patch("web_tools._ddgs_text", return_value=[]):
        result = web_tools.search_famous_painting("The Starry Night")
    _check("query is echoed back", result["query"] == "The Starry Night")
    _check("summary is None, not a crash", result["summary"] is None)
    _check("sources is an empty list, not a crash", result["sources"] == [])


def test_search_famous_painting_filters_web_hits_to_allowlist():
    print("\n=== web_tools.search_famous_painting: non-allowlisted web hits are dropped ===")
    fake_summary_resp = MagicMock(status_code=200)
    fake_summary_resp.json.return_value = {
        "title": "Mona Lisa",
        "extract": "A portrait painting by Leonardo da Vinci.",
        "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Mona_Lisa"}},
    }
    fake_web_hits = [
        {"title": "Louvre official page", "href": "https://www.louvre.fr/en/oeuvre-notices/mona-lisa"},
        {"title": "Sketchy blog", "href": "https://totally-unvetted-art-blog.example/mona-lisa"},
    ]
    with patch("web_tools.requests.get", return_value=fake_summary_resp), \
         patch("web_tools._ddgs_text", return_value=fake_web_hits):
        result = web_tools.search_famous_painting("Mona Lisa")

    urls = {s["url"] for s in result["sources"]}
    _check("summary came through from the (faked) Wikipedia response", "da Vinci" in result["summary"])
    _check("the Wikipedia source itself is included", "https://en.wikipedia.org/wiki/Mona_Lisa" in urls)
    _check("the allowlisted Louvre link is included", "https://www.louvre.fr/en/oeuvre-notices/mona-lisa" in urls)
    _check(
        "the non-allowlisted blog link is NOT included",
        "https://totally-unvetted-art-blog.example/mona-lisa" not in urls,
    )


def test_search_art_supplies_filters_to_amazon_and_ebay_only():
    print("\n=== web_tools.search_art_supplies: results restricted to the retail allowlist ===")

    def _fake_ddgs_text(query, max_results):
        if "site:amazon.com" in query:
            return [
                {"title": "Sable brush set", "href": "https://www.amazon.com/dp/example1",
                 "body": "Top rated, only $19.99, best seller in art supplies."},
            ]
        if "site:ebay.com" in query:
            return [
                {"title": "Vintage easel", "href": "https://www.ebay.com/itm/example2", "body": "Used, $45.00."},
                {"title": "Suspicious reseller", "href": "https://sketchy-dropship.example/x", "body": "$1.00 !!!"},
            ]
        return []

    with patch("web_tools._ddgs_text", side_effect=_fake_ddgs_text):
        results = web_tools.search_art_supplies("watercolor brush", max_results=5)

    urls = {r["url"] for r in results}
    _check("amazon result included", "https://www.amazon.com/dp/example1" in urls)
    _check("ebay result included", "https://www.ebay.com/itm/example2" in urls)
    _check("non-allowlisted dropship result excluded", "https://sketchy-dropship.example/x" not in urls)
    priced = {r["url"]: r["price"] for r in results}
    _check("price parsed correctly for the amazon result", priced["https://www.amazon.com/dp/example1"] == 19.99)
    _check("price parsed correctly for the ebay result", priced["https://www.ebay.com/itm/example2"] == 45.00)


def test_search_art_supplies_empty_when_ddgs_unavailable():
    print("\n=== web_tools.search_art_supplies: ddgs unavailable -> [] not a crash ===")
    with patch("web_tools._ddgs_text", return_value=[]):
        results = web_tools.search_art_supplies("canvas", max_results=5)
    _check("empty result, no raise", results == [])


# ---------------------------------------------------------------------
# image_tools.py
# ---------------------------------------------------------------------


def test_retrieve_images_returns_empty_when_stack_unavailable():
    print("\n=== image_tools.retrieve_images_with_captions: no CLIP/image store -> [] ===")
    with patch("image_tools._ensure_loaded", return_value=False):
        results = image_tools.retrieve_images_with_captions("a palette", k=3)
    _check("empty list, no raise", results == [])


def test_format_markdown_image_shape():
    print("\n=== image_tools.format_markdown_image: renders markdown image + caption line ===")
    rendered = image_tools.format_markdown_image(
        {"caption": "A wooden artist's palette with mixed oil paints", "image_path": "/data/raw/palette.png"}
    )
    _check("markdown image syntax present", rendered.startswith("!["))
    _check("caption text present", "wooden artist's palette" in rendered)
    _check("path present", "/data/raw/palette.png" in rendered)


def test_format_markdown_image_handles_missing_fields():
    print("\n=== image_tools.format_markdown_image: missing caption/path degrades, doesn't crash ===")
    rendered = image_tools.format_markdown_image({})
    _check("placeholder caption used", "(no caption available)" in rendered)
    _check("placeholder path used", "(no path)" in rendered)


# ---------------------------------------------------------------------
# image_tools.py -- base64-embedded retrieval (new, additive code)
# ---------------------------------------------------------------------


def test_encode_image_base64_returns_none_for_missing_file():
    print("\n=== image_tools._encode_image_base64: missing file -> None, no raise ===")
    result = image_tools._encode_image_base64("/no/such/file/anywhere.png")
    _check("None returned", result is None)


def test_encode_image_base64_returns_none_for_empty_path():
    print("\n=== image_tools._encode_image_base64: empty path -> None, no raise ===")
    _check("None for empty string", image_tools._encode_image_base64("") is None)
    _check("None for None", image_tools._encode_image_base64(None) is None)


def test_encode_image_base64_round_trips_real_bytes():
    print("\n=== image_tools._encode_image_base64: encodes real bytes losslessly, guesses MIME type ===")
    raw_bytes = b"not-really-a-png-but-any-bytes-work-for-this-test\x89PNG\r\n"
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(raw_bytes)
        tmp_path = f.name
    try:
        result = image_tools._encode_image_base64(tmp_path)
        _check("result is not None", result is not None)
        _check("decoded base64 matches the original bytes exactly", base64.b64decode(result["base64"]) == raw_bytes)
        _check("mime_type guessed from .png extension", result["mime_type"] == "image/png")
        _check("size_bytes matches the file's actual size", result["size_bytes"] == len(raw_bytes))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_encode_image_base64_respects_size_cap():
    print("\n=== image_tools._encode_image_base64: a file over max_bytes -> None, not truncated ===")
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(b"x" * 1000)
        tmp_path = f.name
    try:
        result = image_tools._encode_image_base64(tmp_path, max_bytes=500)
        _check("file larger than the cap is rejected, not silently truncated", result is None)
        # Same file, generous cap -> encodes fine. Confirms the cap itself
        # (not some other property of the file) was what rejected it above.
        result_ok = image_tools._encode_image_base64(tmp_path, max_bytes=5000)
        _check("the same file under a bigger cap encodes fine", result_ok is not None)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_retrieve_images_with_data_returns_empty_when_stack_unavailable():
    print("\n=== image_tools.retrieve_images_with_data: no CLIP/image store -> [], same as retrieve_images_with_captions ===")
    with patch("image_tools._ensure_loaded", return_value=False):
        results = image_tools.retrieve_images_with_data("a palette", k=3)
    _check("empty list, no raise", results == [])


def test_retrieve_images_with_data_embeds_real_bytes_alongside_existing_fields():
    print("\n=== image_tools.retrieve_images_with_data: adds image_base64/mime_type/data_uri without dropping existing fields ===")
    raw_bytes = b"\x89PNG\r\n\x1a\nfake-but-real-bytes-for-a-round-trip-test"
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(raw_bytes)
        tmp_path = f.name
    try:
        fake_item = {"image_path": tmp_path, "caption": "A wooden palette", "score": 0.9, "metadata": {"filename": "x.pdf"}}
        with patch("image_tools.retrieve_images_with_captions", return_value=[dict(fake_item)]):
            results = image_tools.retrieve_images_with_data("a palette", k=1)
        _check("one result", len(results) == 1)
        item = results[0]
        _check("original image_path preserved", item["image_path"] == tmp_path)
        _check("original caption preserved", item["caption"] == "A wooden palette")
        _check("original score preserved", item["score"] == 0.9)
        _check("original metadata preserved", item["metadata"] == {"filename": "x.pdf"})
        _check("image_base64 decodes back to the original bytes", base64.b64decode(item["image_base64"]) == raw_bytes)
        _check("mime_type is image/png", item["mime_type"] == "image/png")
        _check(
            "data_uri is a well-formed data: URI containing the same base64 payload",
            item["data_uri"] == f"data:image/png;base64,{item['image_base64']}",
        )
        _check("no encoding_note on a successful embed", "encoding_note" not in item)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_retrieve_images_with_data_degrades_gracefully_for_missing_file():
    print("\n=== image_tools.retrieve_images_with_data: missing file -> image_base64 None + explanatory note, other fields intact ===")
    fake_item = {"image_path": "/no/such/file.png", "caption": "A missing image", "score": 0.5, "metadata": {}}
    with patch("image_tools.retrieve_images_with_captions", return_value=[dict(fake_item)]):
        results = image_tools.retrieve_images_with_data("anything", k=1)
    _check("one result, not dropped", len(results) == 1)
    item = results[0]
    _check("image_base64 is None", item["image_base64"] is None)
    _check("mime_type is None", item["mime_type"] is None)
    _check("data_uri is None", item["data_uri"] is None)
    _check("encoding_note explains why", "encoding_note" in item and len(item["encoding_note"]) > 0)
    _check("caption/score/metadata still intact", item["caption"] == "A missing image" and item["score"] == 0.5)


def test_format_markdown_image_embedded_uses_data_uri():
    print("\n=== image_tools.format_markdown_image_embedded: renders a data: URI image, not a path ===")
    rendered = image_tools.format_markdown_image_embedded(
        {"caption": "A wooden palette", "data_uri": "data:image/png;base64,QUJD"}
    )
    _check("markdown image syntax present", rendered.startswith("!["))
    _check("caption present", "wooden palette" in rendered)
    _check("data URI present, not a filesystem path", "data:image/png;base64,QUJD" in rendered)


def test_format_markdown_image_embedded_falls_back_without_data_uri():
    print("\n=== image_tools.format_markdown_image_embedded: no data_uri -> falls back to path-based rendering ===")
    item = {"caption": "A wooden palette", "image_path": "/data/raw/palette.png", "data_uri": None}
    rendered = image_tools.format_markdown_image_embedded(item)
    _check("falls back to the same output format_markdown_image itself would produce",
           rendered == image_tools.format_markdown_image(item))
    _check("fallback still contains the path", "/data/raw/palette.png" in rendered)


def test_escape_markdown_caption_neutralizes_a_stray_closing_bracket():
    print("\n=== image_tools._escape_markdown_caption: a stray ']' no longer breaks the alt-text span ===")
    # Reproduces a confirmed live-run failure: a VLM caption (free-form
    # text, config.IMAGE_CAPTION_PROMPT has no punctuation constraint)
    # containing an unescaped "]" prematurely closed the image's
    # `![...]` alt-text span in the rendered markdown -- everything
    # after it, including the entire base64 data_uri, then fell through
    # as plain paragraph text in the chat UI instead of being parsed as
    # an image.
    caption = "A palette [detail: mixed oils] on a wooden table"
    escaped = image_tools._escape_markdown_caption(caption)
    _check("the stray ']' is now escaped, not a bare bracket", "\\]" in escaped and "]" not in escaped.replace("\\]", ""))
    _check("the stray '[' is also escaped", "\\[" in escaped)
    _check("the rest of the caption text is unchanged", "detail: mixed oils" in escaped)

    rendered = image_tools.format_markdown_image_embedded(
        {"caption": caption, "data_uri": "data:image/png;base64,QUJD"}
    )
    _check(
        "the full data_uri survives intact inside the parens -- the alt span no longer closes early",
        rendered.endswith("data:image/png;base64,QUJD)\n*" + escaped + "*"),
    )
    _check("only ONE '![' opens the image (the escaped one inside alt text doesn't start a new image)",
           rendered.count("![") == 1)


def test_escape_markdown_caption_collapses_newlines():
    print("\n=== image_tools._escape_markdown_caption: a multi-line caption is collapsed to one line ===")
    caption = "A wooden palette\nwith mixed oil paints\n\non a table"
    escaped = image_tools._escape_markdown_caption(caption)
    _check("no newlines remain", "\n" not in escaped)
    _check("words are preserved, single-spaced", escaped == "A wooden palette with mixed oil paints on a table")


def test_escape_markdown_caption_leaves_empty_or_none_untouched():
    print("\n=== image_tools._escape_markdown_caption: empty/None pass through so caller fallback logic still works ===")
    _check("None stays None", image_tools._escape_markdown_caption(None) is None)
    _check("empty string stays empty string", image_tools._escape_markdown_caption("") == "")


def test_format_markdown_image_still_shows_placeholder_for_missing_caption():
    print("\n=== image_tools.format_markdown_image: escaping doesn't disturb the hardcoded placeholder text ===")
    rendered = image_tools.format_markdown_image({"image_path": "/data/raw/x.png"})
    _check("placeholder caption is exactly '(no caption available)', untouched by escaping",
           "(no caption available)" in rendered)


# ---------------------------------------------------------------------
# invoice_tools.py
# ---------------------------------------------------------------------


def test_build_invoice_computes_correct_subtotal():
    print("\n=== invoice_tools.build_invoice: subtotal math is correct ===")
    with patch("invoice_tools._write_invoice_file", return_value=None):
        result = invoice_tools.build_invoice(
            [
                {"name": "Brush", "price": 10.0, "quantity": 2, "url": "https://www.amazon.com/dp/x"},
                {"name": "Canvas", "price": 5.5, "quantity": 1, "url": "https://www.ebay.com/itm/y"},
            ]
        )
    _check("subtotal is 25.50", result["subtotal"] == 25.5)
    _check("item_count sums quantities (2 + 1 = 3)", result["item_count"] == 3)
    _check("two line items rendered", len(result["line_items"]) == 2)
    _check("nothing skipped", result["skipped"] == [])
    _check("markdown invoice contains the subtotal", "$25.50" in result["invoice_markdown"])


def test_build_invoice_skips_unpriceable_items_instead_of_zeroing_them():
    print("\n=== invoice_tools.build_invoice: an item with no price is skipped, not billed as $0 ===")
    with patch("invoice_tools._write_invoice_file", return_value=None):
        result = invoice_tools.build_invoice(
            [
                {"name": "Priced brush", "price": 10.0, "quantity": 1, "url": ""},
                {"name": "Mystery item", "price": None, "quantity": 1, "url": ""},
                {"name": "Bad quantity item", "price": 5.0, "quantity": 0, "url": ""},
            ]
        )
    _check("subtotal only reflects the one valid item", result["subtotal"] == 10.0)
    _check("exactly one line item made it through", len(result["line_items"]) == 1)
    _check("two items were skipped (missing price, zero quantity)", len(result["skipped"]) == 2)


def test_build_invoice_drops_non_allowlisted_url_but_keeps_the_line_item():
    print("\n=== invoice_tools.build_invoice: a non-allowlisted URL is dropped, item is still billed ===")
    with patch("invoice_tools._write_invoice_file", return_value=None):
        result = invoice_tools.build_invoice(
            [{"name": "Brush", "price": 10.0, "quantity": 1, "url": "https://sketchy-dropship.example/x"}]
        )
    _check("item is still priced and included", result["subtotal"] == 10.0)
    _check("the non-allowlisted URL was stripped from the line item", result["line_items"][0]["url"] == "")
    _check("domain_ok is False", result["line_items"][0]["domain_ok"] is False)


def test_build_invoice_never_raises_on_malformed_input():
    print("\n=== invoice_tools.build_invoice: malformed items list degrades instead of raising ===")
    with patch("invoice_tools._write_invoice_file", return_value=None):
        result = invoice_tools.build_invoice([{"name": "no price key at all"}, {}, {"price": "not a number"}])
    _check("subtotal is 0.0", result["subtotal"] == 0.0)
    _check("every malformed item ended up in skipped", len(result["skipped"]) == 3)
    _check("invoice_markdown still renders something sane", "No items could be priced" in result["invoice_markdown"])


def main():
    test_clean_painting_query_strips_question_wrappers()
    test_search_famous_painting_uses_the_cleaned_query_not_the_raw_question()
    test_price_regex_extracts_from_snippet()
    test_price_regex_ignores_snippet_with_no_price()
    test_search_famous_painting_degrades_gracefully_with_no_network()
    test_search_famous_painting_filters_web_hits_to_allowlist()
    test_search_art_supplies_filters_to_amazon_and_ebay_only()
    test_search_art_supplies_empty_when_ddgs_unavailable()
    test_retrieve_images_returns_empty_when_stack_unavailable()
    test_format_markdown_image_shape()
    test_format_markdown_image_handles_missing_fields()
    test_encode_image_base64_returns_none_for_missing_file()
    test_encode_image_base64_returns_none_for_empty_path()
    test_encode_image_base64_round_trips_real_bytes()
    test_encode_image_base64_respects_size_cap()
    test_retrieve_images_with_data_returns_empty_when_stack_unavailable()
    test_retrieve_images_with_data_embeds_real_bytes_alongside_existing_fields()
    test_retrieve_images_with_data_degrades_gracefully_for_missing_file()
    test_format_markdown_image_embedded_uses_data_uri()
    test_format_markdown_image_embedded_falls_back_without_data_uri()
    test_escape_markdown_caption_neutralizes_a_stray_closing_bracket()
    test_escape_markdown_caption_collapses_newlines()
    test_escape_markdown_caption_leaves_empty_or_none_untouched()
    test_format_markdown_image_still_shows_placeholder_for_missing_caption()
    test_build_invoice_computes_correct_subtotal()
    test_build_invoice_skips_unpriceable_items_instead_of_zeroing_them()
    test_build_invoice_drops_non_allowlisted_url_but_keeps_the_line_item()
    test_build_invoice_never_raises_on_malformed_input()
    print("\nAll new-tool smoke tests passed.")


if __name__ == "__main__":
    main()
