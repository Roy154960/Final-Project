"""
Safety enhancement - link/domain allowlisting.

Added alongside prompt_injection.py and pii_redaction.py when the agent
system grew tools that can put a URL in front of the user for the first
time (painting sources, art-supply product links, invoice line items).
Every other safety module in this folder existed because ingested
documents or generated answers are untrusted; a clickable link the user
might actually follow is a new, higher-stakes flavor of the same problem
-- a hallucinated or malicious URL here isn't just a wrong answer, it's
something the user could click.

Two use sites, deliberately not one:

  1. SOURCE-side filtering, in mcp_server/web_tools.py and
     mcp_server/image_tools.py: search results are filtered down to this
     allowlist *before* they're ever handed to a specialist's LLM, so a
     tool's own output can't introduce an untrusted domain in the first
     place.
  2. SINK-side re-checking, in agents/guardrails.py's output_guard: every
     markdown link in the final answer is scanned one more time before
     it reaches the user, in case a model paraphrases, rewrites, or
     otherwise reintroduces a URL that didn't come straight from a tool
     call. This is the same "structural guardrail, not just prompt
     wording" preference already applied elsewhere in this project
     (corpus_meta's missing tools, _extract_grounded_answer's direct
     tool-output extraction, supervisor.py's four validated-routing
     safety nets) -- applied here to link safety instead of answer
     correctness or routing.

The allowlist below is deliberately small and curated, not a general
"is this a reputable site" classifier -- see the module-level TODO note
for the honest limitation this implies, worth stating directly in a
report rather than papering over: a domain-name allowlist says nothing
about whether a specific page is accurate, in stock, or fairly priced.
It only guarantees the *domain* is one of a short list picked in
advance, which is a much weaker claim than "reputable."
"""

import re
from urllib.parse import urlparse

# Reference / encyclopedic sources -- used by web_tools.search_famous_painting.
_REFERENCE_DOMAINS = {
    "wikipedia.org",
    "en.wikipedia.org",
    "commons.wikimedia.org",
    "www.wikiart.org",
    "wikiart.org",
    "www.metmuseum.org",
    "www.nationalgallery.org.uk",
    "www.nga.gov",
    "www.louvre.fr",
    "www.rijksmuseum.nl",
    "www.britannica.com",
    "www.smithsonianmag.com",
    "www.moma.org",
}

# Retail sources -- used by web_tools.search_art_supplies / invoice links.
_RETAIL_DOMAINS = {
    "www.amazon.com",
    "amazon.com",
    "www.ebay.com",
    "ebay.com",
    "www.blickart.com",
    "www.dickblick.com",
    "www.jacksonsart.com",
    "www.cheapjoes.com",
    "www.utrecht.com",
}

ALLOWED_DOMAINS: set[str] = _REFERENCE_DOMAINS | _RETAIL_DOMAINS

# TODO (documented limitation, not an oversight -- see this module's own
# docstring): a real product/review pipeline would check domain
# reputation dynamically (age, TLS issuer, a review-aggregation API)
# instead of a hand-maintained set. That's a reasonable Part-2 stretch
# goal; this project stays local-only / no-paid-API, so a static,
# hand-curated allowlist is the honest tradeoff here, not a placeholder
# that was meant to be finished and wasn't.

_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")


def _hostname(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
    except (ValueError, AttributeError):
        return ""
    return host.split(":")[0]  # strip a port, if any


def is_allowed_domain(url: str) -> bool:
    """
    True if `url`'s hostname is exactly in ALLOWED_DOMAINS, or is a
    subdomain of an entry in it (e.g. "shop.amazon.com" is allowed
    because "amazon.com" is; "amazon.com.evil.example" is NOT allowed --
    endswith("." + d) only matches a true subdomain, not a lookalike
    suffix, since "evil.example" would need to literally end in
    ".amazon.com" for that check to pass).
    """
    host = _hostname(url)
    if not host:
        return False
    return host in ALLOWED_DOMAINS or any(host.endswith("." + d) for d in ALLOWED_DOMAINS)


def filter_allowed(items: list[dict], url_key: str = "url") -> list[dict]:
    """
    Keep only dicts whose [url_key] resolves to an allowlisted domain.
    Used by web_tools.py to filter raw search-engine results before they
    ever become tool output a specialist's LLM can see or repeat.
    """
    return [it for it in items if it.get(url_key) and is_allowed_domain(it[url_key])]


def strip_disallowed_links(text: str) -> tuple[str, int]:
    """
    Rewrite every markdown link `[label](url)` whose url is NOT
    allowlisted into plain `label` text -- drop the clickable target,
    keep the surrounding sentence readable, rather than deleting the
    whole sentence around it. Returns (new_text, links_removed).

    This is the sink-side check output_guard runs on every final answer.
    In the normal case it removes nothing, because every link a
    specialist could have produced already came from an allowlist-
    filtered tool result -- this only fires if a model paraphrased a URL
    into existence on its own (fabricated it, or altered one it was
    given), which is exactly the failure mode source-side filtering
    alone cannot catch.
    """
    removed = 0

    def _sub(match: re.Match) -> str:
        nonlocal removed
        label, url = match.group(1), match.group(2)
        if is_allowed_domain(url):
            return match.group(0)
        removed += 1
        return label

    new_text = _MD_LINK_RE.sub(_sub, text)
    return new_text, removed


if __name__ == "__main__":
    safe = "See [the Louvre](https://www.louvre.fr/en/oeuvre-notices/mona-lisa) for more."
    unsafe = "Buy it [here](https://totally-legit-brushes.example/deal) instead."
    print("safe:", is_allowed_domain("https://www.louvre.fr/en/oeuvre-notices/mona-lisa"))
    print("unsafe:", is_allowed_domain("https://totally-legit-brushes.example/deal"))
    print(strip_disallowed_links(safe))
    print(strip_disallowed_links(unsafe))
