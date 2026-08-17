"""
Internet-search backing for two new MCP tools: `search_painting_online`
and `search_art_supplies` (see server.py for the tool wrappers around
these functions).

Kept in its own module, not inlined in server.py, for the same reason
server.py's own module docstring gives for wiring against
retrieval/hybrid_retriever.py rather than reimplementing it there: this
is a distinct concern (going out to the live internet) from the rest of
server.py's job (serving the local retrieval pipeline), and it has its
own failure modes -- no network, a rate-limited search backend, a
malformed response -- that deserve to be handled in one place rather
than scattered through tool bodies.

Free, no paid API key, matching config.py's own "every model/backend
used anywhere in this project is FREE and runs locally" framing, extended
here to "and every external service this module calls is free and
keyless" -- specifically:
  - Wikipedia's REST API (no key, generous rate limits, well-documented)
    for `search_painting_online`'s primary source.
  - `ddgs` (the maintained successor to the old `duckduckgo_search`
    package) for general web search, used both to find non-Wikipedia
    art sources and to find product listings for `search_art_supplies`.
    No key, but also no SLA -- see `_ddgs_text`'s docstring for how a
    failure here degrades instead of raising.

Every public function in this module returns a (possibly empty) list or
dict and never raises on a network failure -- the same "empty list is a
valid answer, not an error" contract retrieve() already documents in
server.py, extended to "the internet was unreachable" as one more reason
the result can legitimately come back empty.
"""

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Optional

import requests

try:
    from safety.domain_allowlist import filter_allowed, is_allowed_domain
except ImportError:
    # Normal usage (imported by mcp_server/server.py) already has
    # local_rag/ on sys.path by the time this import runs -- server.py's
    # own _find_pipeline_root() does that BEFORE importing web_tools as
    # a bare sibling module (see server.py's own comments on this). This
    # except only fires when web_tools.py is run/imported some OTHER way
    # that skipped that setup -- most commonly `python -m
    # mcp_server.web_tools` or `python mcp_server/web_tools.py`, run
    # directly for the manual diagnostic check this module's own
    # __main__ block below supports. Confirmed live-run failure this
    # fixes: running that diagnostic directly raised `ModuleNotFoundError:
    # No module named 'safety'`, since executing this file/module
    # directly never goes through server.py's bootstrap at all.
    #
    # Same sys.path pattern test_new_tools_smoke.py's own header comment
    # already uses, so a manual check and the smoke tests resolve
    # imports identically -- not a new convention invented here.
    _MCP_SERVER_DIR = Path(__file__).resolve().parent
    _PROJECT_ROOT = _MCP_SERVER_DIR.parent
    for _p in (_MCP_SERVER_DIR, _PROJECT_ROOT / "local_rag", _PROJECT_ROOT):
        if str(_p) not in sys.path and _p.exists():
            sys.path.insert(0, str(_p))
    from safety.domain_allowlist import filter_allowed, is_allowed_domain

_WIKIPEDIA_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
_WIKIPEDIA_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
_REQUEST_TIMEOUT_SECONDS = 8
_USER_AGENT = "multi-agent-pipeline-coursework/1.0 (educational project; no commercial use)"

# Every `ddgs` (9.x) text-search engine except "wikipedia" and
# "grokipedia" -- see _ddgs_text's own docstring below for why those two
# are excluded rather than left to ddgs's own "auto" default. Hardcoded
# as a plain string (not read off `ddgs.engines.ENGINES` at runtime) so
# this doesn't depend on a non-public internal module staying stable
# across `ddgs` releases -- this project's own requirements.txt only
# pins `ddgs>=9.0.0`, so the installed version can drift upward. If a
# future `ddgs` renames/adds/removes engines, worst case this list goes
# stale and `ddgs` falls back to whichever of these names it still
# recognizes (logging a warning for the rest, never raising) -- see
# `DDGS._get_engines`'s own "invalid_keys" handling.
_DDGS_TEXT_BACKENDS = "duckduckgo,bing,brave,mojeek,yahoo,startpage,yandex,google"

_PRICE_RE = re.compile(r"\$\s?(\d{1,4}(?:,\d{3})*(?:\.\d{1,2})?)")

# Fallback price extraction for fetch_listing_price() below, tried in this
# order against a listing page's raw HTML once JSON-LD (_price_from_jsonld)
# comes up empty: an itemprop="price" meta/span (either attribute order),
# an Open Graph product:price:amount meta tag, or a raw "price":"NN.NN"
# embedded in inline JSON/JS (common in Amazon/eBay's own hydration data).
# First match wins -- this is a regex sweep, not a real HTML parser (this
# project has no bs4/lxml dependency; see mcp_server/requirements.txt's own
# "everything free and local, minimal deps" framing), so it is deliberately
# narrow rather than exhaustive.
_JSONLD_SCRIPT_RE = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.IGNORECASE | re.DOTALL,
)
_META_PRICE_RE = re.compile(
    r'itemprop=["\']price["\'][^>]*content=["\']([\d,]+\.?\d*)["\']'
    r'|content=["\']([\d,]+\.?\d*)["\'][^>]*itemprop=["\']price["\']'
    r'|property=["\']product:price:amount["\'][^>]*content=["\']([\d,]+\.?\d*)["\']'
    r'|"price"\s*:\s*"?(\d[\d,]*\.\d{1,2})"?',
    re.IGNORECASE,
)

# Confirmed live-run failure this exists to fix: a real run of
# `python -m agents.graph "Tell me about the Mona Lisa"` had
# painting_lookup_node pass the ENTIRE question straight into
# search_famous_painting("Tell me about the Mona Lisa"). The direct
# Wikipedia summary lookup 404s on that (it's not a real page title), so
# it fell through to wikipedia_best_title()'s full-text search API --
# which matched the whole sentence more strongly against "Mona Lisa
# Smile" (the 2003 film) than against the "Mona Lisa" painting article,
# since the extra question-wrapper words ("tell me about") are noise the
# search-relevance ranking has no way to know isn't part of the subject.
# Stripping a small, fixed set of common question-phrasing prefixes
# before either lookup removes that noise at the source, rather than
# trying to out-guess Wikipedia's own relevance ranking after the fact.
#
# "(the\s+)?" trailer ADDED after a second confirmed live-run failure:
# "explain about the mona lisa" only matched the bare "explain"
# alternative below, stripping just that one word and leaving "about the
# mona lisa" behind as the "cleaned" query -- still noisy enough to
# degrade the Wikipedia lookup, just less catastrophically than leaving
# the whole sentence in. "describe"/"explain" now also optionally
# swallow a following "about (the)?" the same way "tell me about"
# already fully consumes its own "about".
_QUESTION_WRAPPER_RE = re.compile(
    r"^\s*(please\s+)?"
    r"(tell me (more\s+)?about|"
    r"who (painted|created|made|drew|is the artist behind)|"
    r"what (is|are|was)|what's|"
    r"(describe|explain)(\s+about)?|"
    r"give me (information|details) (on|about)|"
    r"can you tell me about|"
    r"i want to know about)\s+(the\s+)?",
    re.IGNORECASE,
)


# Unicode range for Arabic script (same ranges eval_language.py's own
# _detect_language already uses for the identical purpose there -- see
# that function's docstring). Used ONLY by _extract_latin_title_fallback
# below, not for any language-identification decision elsewhere in this
# file.
_ARABIC_SCRIPT_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")
_LATIN_RUN_RE = re.compile(r"[A-Za-z][A-Za-z0-9''\-\s]{2,}[A-Za-z0-9]")


def _extract_latin_title_fallback(text: str) -> Optional[str]:
    """
    Last-resort extraction for a NON-English instruction wrapped around
    an otherwise-untouched Latin-script painting name -- e.g. Arabic
    "اشرح لي عن mona lisa" ("explain to me about mona lisa"),
    where the instruction words are Arabic but the painting's own name
    was typed in Latin script, unchanged, exactly as the person wrote
    it.

    CONFIRMED live-run failure this closes: _QUESTION_WRAPPER_RE above
    is English-only by design (a small, fixed, hand-maintained phrase
    list -- see that constant's own docstring on why it's deliberately
    narrow, not general NLU) -- it does not match, and cannot strip, an
    Arabic (or any non-Latin-script) instruction prefix at all. A real
    run of "اشرح لي عن mona lisa" passed that ENTIRE string, Arabic
    instruction and all, into the Wikipedia summary lookup, which
    predictably found nothing -- while a completely separate, much
    fuzzier general web search on the same polluted string still
    happened to surface real, relevant sources (Wikipedia, Britannica),
    producing the exact "no summary was found... [three sources listed
    right below]" contradiction specialists.py's own painting_lookup_node
    fix addresses on the OTHER side of this same failure.

    Rather than hand-maintaining Arabic (and every other language's)
    equivalent of _QUESTION_WRAPPER_RE's own phrase list -- an
    ever-growing, never-complete translation burden -- this takes the
    much more general approach the transcript itself already
    demonstrates works: a painting title given inside an otherwise
    non-Latin-script sentence is very often left in its own original
    script, unchanged, exactly because it's a proper noun. So: if the
    text contains BOTH Arabic-range characters AND a Latin-script run of
    3+ letters, and stripping wrapper words (the normal path above)
    didn't change anything, return just the LONGEST Latin-script run
    instead -- that's almost always the actual title someone meant.

    Returns None (meaning: no better candidate than the original text)
    if the input has no Arabic-range characters at all, or no
    Latin-script run of at least 3 letters -- so this never fires for a
    plain English (or plain Arabic) query, only the specific mixed-script
    case it exists for.
    """
    if not _ARABIC_SCRIPT_RE.search(text):
        return None
    runs = _LATIN_RUN_RE.findall(text)
    if not runs:
        return None
    longest = max(runs, key=len).strip()
    return longest or None


def _clean_painting_query(painting_name: str) -> str:
    """
    Strip a fixed set of common question-phrasing wrappers (and trailing
    punctuation) off `painting_name` before handing it to either
    Wikipedia lookup below -- see this module's comment above
    `_QUESTION_WRAPPER_RE` for the confirmed live-run failure this fixes,
    and `_extract_latin_title_fallback`'s own docstring for the second,
    mixed-script failure this now also closes.

    Deliberately narrow (a fixed prefix list, not general NLU) and
    applied ONLY inside search_famous_painting -- specialists.py's
    painting_lookup_node still passes the corpus retrieve() call the
    FULL original question, since hybrid/BM25 retrieval benefits from
    the extra context words rather than being confused by them the way
    a title-matching lookup is; only the internet lookup needs this.

    Falls back to the original, untouched string if stripping the
    wrapper would leave nothing behind (e.g. the input was just "tell me
    about" with no subject) -- an empty query is worse than a noisy one.
    """
    cleaned = _QUESTION_WRAPPER_RE.sub("", painting_name.strip())
    cleaned = cleaned.strip().rstrip("?!.").strip()
    if cleaned and cleaned.lower() != painting_name.strip().lower():
        return cleaned
    # The English wrapper-strip above didn't change anything -- either
    # there was no English wrapper to strip, or (per
    # _extract_latin_title_fallback's own docstring) the wrapper is in a
    # different script entirely. Try the mixed-script fallback before
    # giving up and returning the original untouched.
    latin_fallback = _extract_latin_title_fallback(painting_name)
    return latin_fallback or cleaned or painting_name.strip()


def _log(msg: str) -> None:
    print(f"[web_tools] {msg}", file=sys.stderr)


def _ddgs_text(query: str, max_results: int) -> list[dict]:
    """
    Thin wrapper around `ddgs.DDGS().text(...)`. Imported lazily inside
    the function (not at module top) so that a machine without `ddgs`
    installed can still import this module and use
    `wikipedia_summary`/`search_painting_online`'s Wikipedia-only path --
    the two concerns (Wikipedia lookups, general web search) are
    independently optional, not an all-or-nothing dependency.

    Passes an explicit `backend` rather than relying on `ddgs`'s own
    "auto" default. Confirmed by reading `ddgs.ddgs.DDGS._get_engines`'s
    own source (`ddgs` 9.14.4): for `category="text"`, "auto" always
    puts "wikipedia" and "grokipedia" FIRST in the engine list, ahead of
    every general web-search engine -- and neither can ever return a
    result for either query shape this module needs (a `site:amazon.com`
    / `site:ebay.com`-restricted product search, or a general painting-
    source search). With `max_results` kept small (this project's own
    per-site fetch is only 3-10), `ddgs`'s own concurrency cap
    (`ceil(max_results/10)+1` workers) can be as low as 2 -- meaning
    BOTH of the very first attempts on a call can land on engines that
    were structurally guaranteed to return nothing, before a real
    general-web engine is ever tried. `_DDGS_TEXT_BACKENDS` below is
    every registered text engine except those two, so no attempt is
    wasted on one that can't possibly help here.

    This does NOT fix a genuinely unreachable/blocked search backend --
    that's still a real failure mode (a network-level block, a captcha
    wall, a search engine actively rate-limiting scraped requests, all
    real and increasingly common with unofficial libraries like this
    one) and still degrades to [] exactly as before. It only stops
    burning the search's limited concurrency budget on two engines that
    could never have succeeded for this module's queries regardless.

    Returns [] on ANY failure (missing package, network error, rate
    limit, malformed response) rather than raising -- a live run of this
    project already treats "the corpus is empty" as a legitimate empty
    retrieve() result, and "the search backend is unreachable right now"
    deserves the exact same non-fatal treatment, logged to stderr for
    diagnosis rather than silently swallowed.

    CONFIRMED live-run failure this also guards against, not just a
    theorized one: `DDGS(timeout=...)` (default 5s) only bounds how long
    `ddgs.text()`'s own internal `wait(futures, timeout=self._timeout,
    return_when="FIRST_EXCEPTION")` waits before it STOPS WAITING and
    moves on -- it does not cancel or abandon a search-engine thread that
    is still running past that point. `ddgs.text()` dispatches each
    backend as a thread via `with ThreadPoolExecutor(...) as executor:`,
    and a bare `with` block's `__exit__` calls `executor.shutdown(wait=
    True)` unconditionally on the way out -- which blocks until EVERY
    submitted thread finishes, including a straggler stuck on a rate-
    limited or network-stalled backend, no matter how long that takes.
    A single call into this function was observed live holding an entire
    /chat turn open for the full 600s server-side turn timeout (see
    agents/api.py's TURN_TIMEOUT_SECONDS) with nothing else in the graph
    still running -- traced directly to this call, the only one in this
    module with no timeout of its OWN independent of `ddgs`'s internal
    (and, per the above, unreliable) one.

    Fixed the same way every OTHER network call in this module already
    is (see `_REQUEST_TIMEOUT_SECONDS` on the plain `requests.get` calls
    above): a hard, EXTERNAL deadline this function enforces itself,
    that does not depend on `ddgs` -- or whatever search backend it
    happens to be talking to that day -- ever honoring its own. The
    actual `ddgs.text()` call runs in its own throwaway single-thread
    executor; `future.result(timeout=_REQUEST_TIMEOUT_SECONDS)` is what
    actually bounds THIS function's own wall-clock time, not `DDGS`'s
    constructor argument. On timeout, the abandoned worker thread is
    left to finish (or never finish) on its own -- Python has no safe,
    general way to kill a running thread -- but `executor.shutdown(wait=
    False)` in the `finally` below means THIS call never blocks waiting
    for it, which is the only thing that actually matters for the
    person waiting on their chat turn.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        _log("`ddgs` is not installed (pip install ddgs) -- general web search unavailable")
        return []

    def _run_search() -> list[dict]:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results, backend=_DDGS_TEXT_BACKENDS))

    # max_workers=1: this executor exists solely to give one blocking
    # call a hard deadline, not to run anything concurrently -- same
    # "smallest tool for the job" reasoning as every other single-use
    # executor already in this codebase (e.g. api.py's own use of
    # asyncio.wait_for for its own, higher-level deadline).
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ddgs-text")
    try:
        future = executor.submit(_run_search)
        return future.result(timeout=_REQUEST_TIMEOUT_SECONDS)
    except FutureTimeoutError:
        _log(
            f"general web search TIMED OUT after {_REQUEST_TIMEOUT_SECONDS}s for "
            f"query {query!r} -- abandoning it and returning [] rather than "
            f"blocking the caller any further (see this function's own "
            f"docstring for why DDGS's own timeout doesn't reliably bound this)"
        )
        return []
    except Exception as e:  # noqa: BLE001 -- any backend failure degrades to [], see docstring
        _log(f"general web search failed for query {query!r}: {e}")
        return []
    finally:
        # wait=False: do NOT block here waiting for an already-abandoned
        # (timed-out) worker thread to finish -- that would silently
        # reintroduce the exact hang this whole rewrite exists to close.
        # Harmless on the normal, non-timeout path too: the thread has
        # already finished by the time future.result() returned above.
        executor.shutdown(wait=False)


def wikipedia_summary(title: str) -> Optional[dict]:
    """
    Fetch Wikipedia's own short summary for an exact (or close) page
    title via the REST summary endpoint. Returns None on a 404 (no such
    page) or any request failure -- never raises.

    Returns {"title": str, "extract": str, "url": str} on success. The
    extract is Wikipedia's own lede-paragraph summary, already short
    (typically 2-4 sentences) -- not further truncated here, since
    trimming mid-sentence would make it read as broken rather than
    "small," and the caller (search_famous_painting) is the one that
    knows how much room it actually has.
    """
    url = _WIKIPEDIA_SUMMARY_URL.format(title=requests.utils.quote(title))
    try:
        resp = requests.get(
            url, headers={"User-Agent": _USER_AGENT}, timeout=_REQUEST_TIMEOUT_SECONDS
        )
    except requests.RequestException as e:
        _log(f"wikipedia_summary request failed for {title!r}: {e}")
        return None

    if resp.status_code != 200:
        return None

    try:
        data = resp.json()
    except ValueError:
        return None

    extract = data.get("extract", "").strip()
    if not extract:
        return None

    page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
    return {"title": data.get("title", title), "extract": extract, "url": page_url}


def wikipedia_best_title(query: str) -> Optional[str]:
    """
    Resolve a loosely-phrased query (e.g. "starry night van gogh") to
    Wikipedia's own best-matching page title, via the plain search API --
    used as a fallback when wikipedia_summary(query) 404s on the raw
    query text (most famous-painting titles work directly, e.g. "Mona
    Lisa" or "The Starry Night", but not every phrasing will).

    Returns None if the search API fails or returns no hits.
    """
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": 1,
    }
    try:
        resp = requests.get(
            _WIKIPEDIA_SEARCH_URL,
            params=params,
            headers={"User-Agent": _USER_AGENT},
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        hits = resp.json().get("query", {}).get("search", [])
    except (requests.RequestException, ValueError) as e:
        _log(f"wikipedia_best_title search failed for {query!r}: {e}")
        return None

    return hits[0]["title"] if hits else None


def _price_from_jsonld(html: str) -> Optional[float]:
    """
    Best-effort price extraction from any `application/ld+json` block on
    the page -- schema.org Product/Offer markup, which most modern
    Amazon/eBay listing pages embed for search-engine consumption, and
    the most reliable source this module can read without a paid
    product API. Tries every JSON-LD block on the page, not just the
    first, since a listing page can carry several (breadcrumbs, reviews,
    product) and the price isn't guaranteed to be in the first one.

    Never raises: a block that isn't valid JSON, or valid JSON with no
    "offers"/"price" anywhere in it, is skipped, not fatal -- same
    "empty/None is a legitimate answer" contract as everything else in
    this module.
    """
    for block in _JSONLD_SCRIPT_RE.findall(html):
        try:
            data = json.loads(block.strip())
        except json.JSONDecodeError:
            continue
        for candidate in data if isinstance(data, list) else [data]:
            if not isinstance(candidate, dict):
                continue
            offers = candidate.get("offers")
            if isinstance(offers, list):
                offers = offers[0] if offers else None
            if isinstance(offers, dict):
                price = offers.get("price") or offers.get("lowPrice")
                if price is not None:
                    try:
                        return float(str(price).replace(",", ""))
                    except ValueError:
                        continue
    return None


def _price_from_meta(html: str) -> Optional[float]:
    """
    Fallback for pages without (parseable) JSON-LD -- see
    `_META_PRICE_RE`'s own comment for exactly which HTML shapes this
    tries, in order. First match wins.
    """
    match = _META_PRICE_RE.search(html)
    if not match:
        return None
    raw = next((g for g in match.groups() if g), None)
    if raw is None:
        return None
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def fetch_listing_price(url: str) -> Optional[float]:
    """
    Best-effort REAL price for one listing `url`, by fetching the actual
    page and parsing it -- the accurate alternative to
    search_art_supplies' old snippet-derived `_PRICE_RE` match, which
    only worked when the search engine happened to echo a price string
    into its own result snippet (rare for Amazon/eBay in practice, which
    is why product_search/invoice were seeing `price: None` on almost
    every candidate). Tries JSON-LD first (`_price_from_jsonld`, the more
    structured and reliable source when present), then a meta/inline-JSON
    regex sweep (`_price_from_meta`).

    Uses a real browser User-Agent, not `_USER_AGENT` -- Amazon and eBay
    both routinely serve a stripped-down or bot-detection page to
    non-browser user agents, which would make this function look broken
    for a reason that has nothing to do with the parsing logic itself.

    Still an honest best-effort, not a guarantee, for the same reason
    `search_art_supplies`' own docstring already states: a retailer can
    change its markup, rate-limit, or serve a CAPTCHA at any time, and
    this function degrades to None (never raises) in every one of those
    cases rather than crashing the caller. A real price should always be
    confirmed on the live page before anyone buys anything from a link
    this tool returns.
    """
    try:
        resp = requests.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                )
            },
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        _log(f"fetch_listing_price failed for {url}: {e}")
        return None

    html = resp.text
    return _price_from_jsonld(html) or _price_from_meta(html)


def search_famous_painting(painting_name: str) -> dict:
    """
    The function behind the `search_painting_online` MCP tool.

    Looks up `painting_name` on Wikipedia (after stripping common
    question-phrasing wrappers like "tell me about" -- see
    `_clean_painting_query`'s docstring for the confirmed live-run
    mis-resolution this fixes; trying the cleaned name first, then
    falling back to Wikipedia's own search API to resolve a loose
    phrasing to the right page title), and supplements it with up to two
    more links from a general web search -- filtered down to
    `safety.domain_allowlist.ALLOWED_DOMAINS` (reputable museum /
    encyclopedic sources only) BEFORE they're returned, so this
    function's own output can never hand a specialist's LLM an
    untrusted domain to relay to the user. See guardrails.py's
    output_guard for the second, independent check on the way out.

    Returns:
      {
        "query": the original painting_name,
        "summary": a short summary string, or None if nothing was found
            anywhere (Wikipedia missing AND web search empty/unavailable),
        "sources": list of {"title": str, "url": str}, always allowlist-
            filtered, possibly empty,
      }

    Never raises -- a fully-empty result (summary=None, sources=[]) is a
    legitimate answer meaning "the internet had nothing usable for this
    query right now," which the calling specialist (painting_lookup) is
    expected to say plainly rather than treat as a crash.
    """
    query_name = _clean_painting_query(painting_name)

    summary_data = wikipedia_summary(query_name)
    if summary_data is None:
        resolved_title = wikipedia_best_title(query_name)
        if resolved_title:
            summary_data = wikipedia_summary(resolved_title)

    sources: list[dict] = []
    summary_text: Optional[str] = None

    if summary_data is not None:
        summary_text = summary_data["extract"]
        if summary_data.get("url"):
            sources.append({"title": f"Wikipedia: {summary_data['title']}", "url": summary_data["url"]})

    # Supplement with up to 2 more allowlisted sources from a general web
    # search, regardless of whether Wikipedia itself succeeded -- a
    # second museum/reference source is useful even when Wikipedia also
    # answered, and becomes the ONLY source if Wikipedia had nothing.
    # Uses query_name (cleaned), same reasoning as the Wikipedia lookups
    # above -- the raw question's wrapper words are just as capable of
    # skewing a general web search toward an unrelated result.
    web_hits = _ddgs_text(f"{query_name} painting", max_results=6)
    web_hits = [{"title": h.get("title", ""), "url": h.get("href", "")} for h in web_hits]
    web_hits = filter_allowed(web_hits, url_key="url")
    for hit in web_hits:
        if len(sources) >= 3:
            break
        if hit["url"] not in {s["url"] for s in sources}:
            sources.append(hit)

    if summary_text is None and web_hits:
        # Wikipedia had nothing usable, but a reputable web result did --
        # note that explicitly rather than silently returning summary=None
        # next to a non-empty sources list, which would read as a bug.
        summary_text = None  # left None on purpose; caller decides how to phrase "no summary, but see sources"

    return {"query": painting_name, "summary": summary_text, "sources": sources}


def search_art_supplies(
    query: str,
    max_results: int = 5,
    fetch_real_prices: bool = True,
    max_price_fetches: int = 8,
) -> list[dict]:
    """
    The function behind the `search_art_supplies` MCP tool.

    Searches the general web, restricted to known retailers via two
    separate `site:` queries (Amazon, eBay) rather than one combined
    query -- `ddgs`'s own `site:a.com OR site:b.com` OR-syntax support is
    inconsistent across backends, while two plain site-restricted queries
    are simple and reliable. Results are deduplicated by URL, then
    allowlist-filtered as a second, independent check (the `site:` filter
    is a search-engine hint, not a guarantee -- the allowlist check is
    what's actually structurally enforced).

    Over-fetches `max_results * 2` raw candidates per retailer (up to
    `max_results * 4` total before capping) so the caller (the
    product_search specialist) has real material to rank down to 5 from
    -- the same "over-fetch before narrowing" reasoning server.py's own
    retrieve() already documents for its reranking step.

    Returns a list of dicts, each:
      {"title": str, "url": str, "source": "amazon" | "ebay",
       "price": float | None, "snippet": str}

    `price` is resolved in two stages:
      1. A regex extraction from the search snippet (`snippet`) when a
         "$NN.NN"-shaped substring is present -- fast, no extra request,
         but in practice Amazon/eBay search snippets rarely contain a
         price string, so this alone was leaving `price: None` on almost
         every candidate.
      2. If stage 1 came up empty AND `fetch_real_prices` is True (the
         default), `fetch_listing_price()` fetches the actual listing
         page and parses a real price out of it (JSON-LD product
         markup, then a meta-tag/inline-JSON fallback -- see that
         function's own docstring). Capped at `max_price_fetches` calls
         per `search_art_supplies` invocation, sequential, so this adds
         up to roughly `max_price_fetches * _REQUEST_TIMEOUT_SECONDS`
         worst-case latency -- deliberately bounded rather than fetching
         every candidate's page, since product_search_node can pass
         `max_results` up to a dozen or more.

    Even with `fetch_real_prices=True`, `price` is still NOT a
    guaranteed-accurate current price (a page's markup can change, or the
    fetch can fail/be blocked) -- a listing's real price should always be
    confirmed on the actual page before anyone buys anything from a link
    this tool returns. Pass `fetch_real_prices=False` to fall back to the
    old snippet-only behavior (faster, no extra requests, more `None`s)
    if the extra network calls aren't wanted for a given run.

    A production version of this tool would use Amazon's Product
    Advertising API or eBay's Browse API (both require developer
    registration/keys), which this project's "no paid APIs, everything
    free and local" constraint rules out -- `fetch_listing_price` is the
    free/keyless approximation of that.

    Returns [] if `ddgs` is unavailable or every search fails -- never
    raises. The caller (product_search specialist) is expected to say
    plainly that it couldn't reach the internet rather than fabricate
    product data, the same "say plainly, don't guess" rule
    RETRIEVAL_QA_SYSTEM_PROMPT already applies to the corpus.
    """
    per_site = max(max_results, 3)
    raw: list[dict] = []
    for site, source_name in (("amazon.com", "amazon"), ("ebay.com", "ebay")):
        hits = _ddgs_text(f"{query} site:{site}", max_results=per_site)
        for h in hits:
            raw.append(
                {
                    "title": h.get("title", ""),
                    "url": h.get("href", ""),
                    "source": source_name,
                    "snippet": h.get("body", ""),
                }
            )

    seen_urls: set[str] = set()
    deduped: list[dict] = []
    for item in raw:
        if item["url"] and item["url"] not in seen_urls:
            seen_urls.add(item["url"])
            deduped.append(item)

    allowlisted = [it for it in deduped if is_allowed_domain(it["url"])]

    results = []
    price_fetches_used = 0
    for item in allowlisted[: max_results * 2]:
        price_match = _PRICE_RE.search(item["snippet"])
        price = float(price_match.group(1).replace(",", "")) if price_match else None

        if price is None and fetch_real_prices and price_fetches_used < max_price_fetches:
            price_fetches_used += 1
            price = fetch_listing_price(item["url"])

        results.append({**item, "price": price})

    return results


if __name__ == "__main__":
    print("This module is meant to be imported by mcp_server/server.py.")
    print("Quick manual check (needs internet + `ddgs` installed):")
    print(search_famous_painting("Mona Lisa"))
    print(search_art_supplies("sable watercolor brush", max_results=3))
