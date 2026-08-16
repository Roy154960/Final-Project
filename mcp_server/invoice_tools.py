"""
Invoice construction backing for the new `generate_invoice` MCP tool
(see server.py for the tool wrapper, agents/specialists.py's invoice_node
for the specialist that calls it).

Deliberately NOT an LLM call anywhere in this module. Every number in an
invoice (a line total, the subtotal, the item count) is computed here in
plain Python from structured input, the same "structural guardrail over
prompt wording" preference this project already applies repeatedly --
corpus_meta's missing tools, _extract_grounded_answer's direct
tool-output extraction, supervisor.py's four validated-routing safety
nets -- now applied to arithmetic a small local LLM has no business being
trusted with. A model can decide WHICH items belong on the invoice
(agents/specialists.py's invoice_node does that part, parsing the
conversation history); it never decides what they add up to.

Also deliberately re-checks every item's URL against
safety.domain_allowlist on the way in, independent of whatever filtering
web_tools.search_art_supplies already did when the item was first found
-- an item reaching this function came from a specialist's message
history, one hop removed from the original tool call, so re-checking
here is cheap insurance against a link having been altered somewhere in
between.
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from safety.domain_allowlist import is_allowed_domain  # noqa: E402


def _log(msg: str) -> None:
    print(f"[invoice_tools] {msg}", file=sys.stderr)


def _invoices_dir() -> Path:
    """
    Resolve the folder invoices are written to, lazily (not at import
    time) so importing this module never has a side effect of creating a
    directory.

    Deliberately NOT config.DATA_DIR/"invoices" anymore. DATA_DIR is the
    RAG pipeline's own corpus/ingest folder tree -- an invoice is a
    generated receipt, not corpus content, and writing it in there mixed
    unrelated artifacts into the same folder a future `pipeline.py
    ingest` pass walks over, with the same filesystem noise
    server.py's own _stdout_to_stderr() plumbing was built to avoid
    causing at the protocol level. Invoices now live in their own
    directory, entirely outside the RAG data folder: mcp_server/
    generated_invoices/ by default (a sibling of this file, not
    config.DATA_DIR), overridable via the INVOICE_OUTPUT_DIR environment
    variable for anyone who wants them written somewhere else entirely.
    """
    override = os.environ.get("INVOICE_OUTPUT_DIR")
    d = Path(override) if override else Path(__file__).resolve().parent / "generated_invoices"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_invoice(items: list[dict], customer_note: str = "") -> dict:
    """
    The function behind the `generate_invoice` MCP tool.

    Args:
        items: list of dicts, each expected to have:
            "name": str (required)
            "price": float (required -- see `skipped` below for what
                happens if it's missing or not a number)
            "quantity": int, defaults to 1 if absent
            "url": str, optional
        customer_note: optional free-text line included in the rendered
            invoice as-is (e.g. "for the mural project") -- NOT
            interpreted, evaluated, or used in any calculation; purely
            cosmetic text on the printed invoice.

    An item with a missing or non-numeric price is excluded from the
    subtotal and reported in `skipped` with a reason, rather than
    silently treated as a $0 line item -- a $0 line would understate the
    total with no signal anything was wrong, which is a worse failure
    mode than an honest "1 item couldn't be priced and isn't included
    below" note the caller can surface to the user.

    Returns:
      {
        "line_items": [
          {"name", "unit_price", "quantity", "line_total", "url",
           "domain_ok"}, ...
        ],
        "subtotal": float,           # sum of the PRICED line items only
                                      # -- see "subtotal_available" below
                                      # for whether this is the REAL
                                      # total or a partial figure
        "subtotal_available": bool,  # False whenever `skipped` is
                                      # non-empty, i.e. at least one
                                      # requested item had no usable
                                      # price -- "subtotal" above is then
                                      # only a partial sum, and
                                      # invoice_markdown's own Subtotal
                                      # line reads "Unavailable" rather
                                      # than that partial dollar figure,
                                      # so a person skimming just the
                                      # rendered markdown is never shown
                                      # a number that looks like a total
                                      # but silently excludes items
        "item_count": int,          # sum of quantities actually priced
        "skipped": [{"name", "reason", "url"}, ...],  # "url" is "" if
                                      # none was given or it failed the
                                      # domain allowlist -- same check
                                      # priced line items get
        "generated_at": ISO-8601 UTC timestamp string,
        "invoice_markdown": str,    # ready to show the user directly
        "file_path": str | None,    # where the markdown was saved, if
                                      # the write succeeded; None if it
                                      # failed (never raises either way)
      }

    Never raises: a malformed `items` list degrades to an invoice with
    everything in `skipped` and a $0.00 subtotal, not a crash -- the
    calling specialist can then tell the user plainly that nothing could
    be priced, the same "say plainly, don't crash" pattern used
    throughout this project's guardrail and fallback code.
    """
    line_items = []
    skipped = []
    subtotal = 0.0
    item_count = 0

    for raw in items or []:
        name = str(raw.get("name", "")).strip() or "(unnamed item)"
        price = raw.get("price")
        quantity = raw.get("quantity", 1)
        url = raw.get("url", "") or ""

        # Domain-allowlist check moved ahead of the price try/except so it
        # applies uniformly to EVERY item's link -- including one that
        # ends up in `skipped` for a missing price. A link is still worth
        # showing next to a skipped item (the person can go check the
        # price themselves), but it still needs to be an allowlisted
        # domain either way; there's no reason the safety check here
        # should only apply to items that happened to have a price.
        domain_ok = bool(url) and is_allowed_domain(url)
        if url and not domain_ok:
            _log(f"dropping non-allowlisted URL for item {name!r}: {url}")
            url = ""

        try:
            unit_price = float(price)
            quantity = int(quantity)
            if unit_price < 0 or quantity < 1:
                raise ValueError("negative price or non-positive quantity")
        except (TypeError, ValueError):
            skipped.append({"name": name, "reason": f"invalid or missing price ({price!r})", "url": url})
            continue

        line_total = round(unit_price * quantity, 2)
        subtotal += line_total
        item_count += quantity

        line_items.append(
            {
                "name": name,
                "unit_price": round(unit_price, 2),
                "quantity": quantity,
                "line_total": line_total,
                "url": url,
                "domain_ok": domain_ok,
            }
        )

    subtotal = round(subtotal, 2)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    markdown = _render_markdown(line_items, skipped, subtotal, customer_note, generated_at)

    file_path = _write_invoice_file(markdown, generated_at)

    return {
        "line_items": line_items,
        "subtotal": subtotal,
        "subtotal_available": not skipped,
        "item_count": item_count,
        "skipped": skipped,
        "generated_at": generated_at,
        "invoice_markdown": markdown,
        "file_path": file_path,
    }


def _render_markdown(
    line_items: list[dict],
    skipped: list[dict],
    subtotal: float,
    customer_note: str,
    generated_at: str,
) -> str:
    lines = ["# Invoice", "", f"Generated: {generated_at}", ""]
    if customer_note:
        lines += [f"Note: {customer_note}", ""]

    if line_items:
        lines.append("| Item | Qty | Unit Price | Line Total | Link |")
        lines.append("|---|---|---|---|---|")
        for it in line_items:
            link = f"[View listing]({it['url']})" if it["url"] else "(no link)"
            lines.append(
                f"| {it['name']} | {it['quantity']} | ${it['unit_price']:.2f} "
                f"| ${it['line_total']:.2f} | {link} |"
            )
        lines.append("")
        # A dollar subtotal is only ever shown when EVERY requested item
        # could actually be priced. If `skipped` is non-empty, `subtotal`
        # here is only the sum of the items that happened to have a
        # price -- showing that number under the label "Subtotal" reads
        # as "this is the total cost," which it explicitly isn't (it
        # silently excludes whatever's in `skipped`). That's a worse
        # failure than an honest "Unavailable": a person skimming a
        # partial dollar figure has no way to tell it's incomplete
        # without separately reading the "Not included" section below,
        # and a screenshot or copy-paste of just the subtotal line loses
        # that context entirely. Same "say plainly, don't silently
        # understate" principle this module's own docstring and
        # build_invoice's own `skipped` design already apply to
        # individual line items, now applied to the total line too.
        if skipped:
            item_word = "item" if len(skipped) == 1 else "items"
            verb = "has" if len(skipped) == 1 else "have"
            lines.append(
                f"**Subtotal: Unavailable** ({len(skipped)} {item_word} below "
                f"{verb} no listed price, so a total can't be computed)"
            )
        else:
            lines.append(f"**Subtotal: ${subtotal:.2f}**")
    else:
        lines.append("_No items could be priced -- nothing to invoice._")

    if skipped:
        lines += ["", "**Not included** (couldn't be priced):"]
        for s in skipped:
            link = f" -- [View listing]({s['url']})" if s.get("url") else ""
            lines.append(f"- {s['name']}: {s['reason']}{link}")

    return "\n".join(lines)


def _write_invoice_file(markdown: str, generated_at: str) -> str | None:
    """
    Best-effort save of the rendered invoice to data/invoices/ as a
    timestamped .md file, so "create an invoice" produces an actual
    artifact on disk, not just chat text that scrolls away. Returns None
    (not a raised exception) on any filesystem error -- the invoice's
    in-memory markdown is still returned to the caller either way, so a
    write failure degrades the deliverable, it doesn't break the tool.
    """
    try:
        safe_stamp = generated_at.replace(":", "-")
        path = _invoices_dir() / f"invoice_{safe_stamp}.md"
        path.write_text(markdown, encoding="utf-8")
        return str(path)
    except OSError as e:
        _log(f"failed to write invoice file: {e}")
        return None


if __name__ == "__main__":
    demo = build_invoice(
        [
            {"name": "Winsor & Newton Series 7 brush", "price": 24.99, "quantity": 2,
             "url": "https://www.amazon.com/dp/example"},
            {"name": "Fredrix 16x20 canvas", "price": 12.5, "quantity": 3,
             "url": "https://www.ebay.com/itm/example"},
            {"name": "mystery item with no price", "price": None, "quantity": 1, "url": ""},
        ],
        customer_note="Demo invoice from invoice_tools.py's own smoke check.",
    )
    print(demo["invoice_markdown"])
    print()
    print(f"subtotal={demo['subtotal']} item_count={demo['item_count']} skipped={demo['skipped']}")
    print(f"saved to: {demo['file_path']}")
