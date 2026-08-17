"""
framing_agent/pricing.py

The actual arithmetic behind System B's /quote endpoint. Deliberately
pure Python, zero LLM calls anywhere in this module -- the same
"structural guardrail over prompt wording" preference System A's own
mcp_server/invoice_tools.py and mcp_server/color_tools.py already apply
to their own numbers: an agent (in either system) can decide WHAT to
quote, never WHAT it adds up to. agent.py wraps compute_quote() as a
tool an LLM calls for the numbers and writes an explanation around; it
never re-derives them itself.

This is illustrative coursework pricing, not a real framing shop's rate
card -- every number below is a made-up placeholder, clearly labelled
as such in compute_quote()'s own returned "disclaimer" field, the same
"always treat as an estimate, not a quote" honesty
mcp_server/web_tools.py's search_art_supplies already applies to a
scraped price.

No imports from System A anywhere in this file (or anywhere in this
package) -- System B is a genuinely independent service. It knows
nothing about LangGraph, InMind's corpus, or InMind's own Python
modules; the only contract between the two systems is the plain HTTP
JSON shape compute_quote() returns, documented in server.py's /quote
route.
"""

from datetime import datetime, timezone
from typing import Optional

# ---------------------------------------------------------------------
# Frame styles -- USD per linear cm of frame PERIMETER (2*(w+h)).
# ---------------------------------------------------------------------
FRAME_STYLES: dict[str, float] = {
    "basic wood": 0.18,
    "modern metal": 0.30,
    "classic ornate": 0.42,
}
DEFAULT_FRAME_STYLE = "basic wood"

# ---------------------------------------------------------------------
# Glazing (the glass/acrylic sheet over the artwork) -- USD per cm^2 of
# AREA (width * height). "none" is for canvas-style work that's
# typically framed without glass at all.
# ---------------------------------------------------------------------
GLAZING_OPTIONS: dict[str, float] = {
    "none": 0.0,
    "standard glass": 0.018,
    "uv-protective acrylic": 0.03,
}

# Paper-based media conventionally get glazed (they're not
# lacquered/varnished the way an oil or acrylic canvas usually is);
# canvas-style media conventionally aren't. This is a DEFAULT, not a
# hard rule -- compute_quote()'s own `glazing_override` parameter lets
# a caller override it either direction, same "the model can decide
# WHICH, never WHAT it adds up to" split this module's own docstring
# describes for every other choice here.
_PAPER_BASED_MEDIUM_KEYWORDS = (
    "watercolor", "watercolour", "gouache", "drawing", "sketch", "pastel",
    "charcoal", "ink", "print", "lithograph", "photograph", "photo",
    "paper",
)

# ---------------------------------------------------------------------
# Shipping -- a destination resolves to one of three zones, each with
# its own flat base rate (USD) plus a per-kg rate (USD/kg) on top of an
# estimated package weight. Deliberately coarse (three zones, not a
# real carrier-rate lookup) -- see compute_quote()'s own
# "destination_recognized" field for the honest fallback when a
# destination isn't in this list at all.
# ---------------------------------------------------------------------
_ZONE_BY_COUNTRY: dict[str, str] = {
    "lebanon": "domestic",
    "syria": "regional", "jordan": "regional", "iraq": "regional",
    "egypt": "regional", "turkey": "regional", "cyprus": "regional",
    "uae": "regional", "united arab emirates": "regional",
    "saudi arabia": "regional", "qatar": "regional", "kuwait": "regional",
    "bahrain": "regional", "oman": "regional",
    "france": "international", "germany": "international",
    "italy": "international", "spain": "international",
    "uk": "international", "united kingdom": "international",
    "usa": "international", "united states": "international",
    "canada": "international", "australia": "international",
}
_ZONE_BASE_RATE_USD: dict[str, float] = {
    "domestic": 15.0, "regional": 45.0, "international": 95.0,
}
_ZONE_PER_KG_RATE_USD: dict[str, float] = {
    "domestic": 2.0, "regional": 6.0, "international": 14.0,
}
_DEFAULT_ZONE_FOR_UNKNOWN_DESTINATION = "international"

# Estimated surface density (kg per m^2 of artwork area), used ONLY to
# turn a physical size into a shipping-weight estimate -- glazed work
# is heavier (glass/acrylic sheet + sturdier packaging) than an
# unglazed canvas. _BASE_PACKAGE_WEIGHT_KG accounts for the box/padding
# itself, present regardless of size.
_BASE_PACKAGE_WEIGHT_KG = 0.8
_DENSITY_GLAZED_KG_PER_M2 = 4.5
_DENSITY_UNGLAZED_KG_PER_M2 = 2.0


def _resolve_frame_style(requested: Optional[str]) -> tuple[str, bool]:
    """Case-insensitive, substring-tolerant match against FRAME_STYLES
    (e.g. "wood" or "wooden frame" both resolve to "basic wood"). Returns
    (resolved_style, was_recognized) -- an unrecognized/empty request
    degrades to DEFAULT_FRAME_STYLE rather than raising, with
    was_recognized=False so the caller can say so plainly instead of
    silently substituting."""
    if not requested:
        return DEFAULT_FRAME_STYLE, False
    lowered = requested.strip().lower()
    if lowered in FRAME_STYLES:
        return lowered, True
    for style in FRAME_STYLES:
        if style in lowered or lowered in style:
            return style, True
    return DEFAULT_FRAME_STYLE, False


def _resolve_zone(destination_country: str) -> tuple[str, bool]:
    """(zone, destination_recognized). An unrecognized country name
    degrades to _DEFAULT_ZONE_FOR_UNKNOWN_DESTINATION -- the most
    expensive tier, deliberately, so an unrecognized destination never
    silently UNDER-quotes shipping -- with destination_recognized=False
    so the caller can flag the estimate as rougher than usual rather
    than presenting it with the same confidence as a matched country."""
    lowered = (destination_country or "").strip().lower()
    if lowered in _ZONE_BY_COUNTRY:
        return _ZONE_BY_COUNTRY[lowered], True
    return _DEFAULT_ZONE_FOR_UNKNOWN_DESTINATION, False


def compute_quote(
    width_cm: float,
    height_cm: float,
    medium: str,
    destination_country: str,
    frame_style: Optional[str] = None,
    glazing_override: Optional[bool] = None,
) -> dict:
    """
    The one function every other piece of System B (agent.py's tool
    wrapper, server.py's /quote route) ultimately calls for the actual
    numbers.

    Args:
        width_cm, height_cm: artwork dimensions in centimeters. Must
            both be positive -- see the "error" field below for what a
            non-positive or non-numeric value does instead of raising.
        medium: free text describing the artwork ("oil on canvas",
            "watercolor", "giclee print", ...) -- used only to guess
            whether glazing is conventionally needed (see
            _PAPER_BASED_MEDIUM_KEYWORDS above); never itself priced.
        destination_country: free text country name for the shipping
            estimate (e.g. "Lebanon", "France") -- matched
            case-insensitively against _ZONE_BY_COUNTRY.
        frame_style: one of FRAME_STYLES' keys, or a loose match (see
            _resolve_frame_style) -- defaults to DEFAULT_FRAME_STYLE if
            omitted or unrecognized.
        glazing_override: True/False to force glazing on or off,
            overriding the medium-based default -- None (the default)
            leaves the medium-based guess in place.

    Returns:
        {
          "dimensions_cm": {"width": float, "height": float},
          "medium": str,
          "frame": {"style": str, "style_recognized": bool, "cost_usd": float},
          "glazing": {"chosen": str, "needed": bool, "cost_usd": float},
          "shipping": {
              "destination_country": str, "zone": str,
              "destination_recognized": bool,
              "estimated_weight_kg": float, "cost_usd": float,
          },
          "subtotal_usd": float,
          "currency": "USD",
          "disclaimer": str,
          "generated_at": ISO-8601 UTC timestamp,
          "error": None,
        }

        On invalid dimensions (missing, non-numeric, or <= 0), returns
        the same shape with every cost field 0.0, "error" set to a
        plain-English reason, and every other field left at its
        best-effort value -- never raises. Callers (agent.py, and
        System A's own framing_tools.py on the other side of the
        network boundary) check "error" before trusting the numbers,
        the same "check a structured field, don't catch an exception"
        contract mcp_server/color_tools.py's own generate_palette
        already uses for its own "error" field.
    """
    try:
        width = float(width_cm)
        height = float(height_cm)
        if width <= 0 or height <= 0:
            raise ValueError("dimensions must be positive")
    except (TypeError, ValueError):
        return {
            "dimensions_cm": {"width": width_cm, "height": height_cm},
            "medium": medium,
            "frame": {"style": None, "style_recognized": False, "cost_usd": 0.0},
            "glazing": {"chosen": None, "needed": False, "cost_usd": 0.0},
            "shipping": {
                "destination_country": destination_country, "zone": None,
                "destination_recognized": False, "estimated_weight_kg": 0.0,
                "cost_usd": 0.0,
            },
            "subtotal_usd": 0.0,
            "currency": "USD",
            "disclaimer": (
                "Estimate only -- illustrative pricing for this coursework "
                "demo, not a real framing shop's rate card."
            ),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "error": (
                f"width_cm and height_cm must both be positive numbers "
                f"(got width_cm={width_cm!r}, height_cm={height_cm!r})."
            ),
        }

    perimeter_cm = 2 * (width + height)
    area_cm2 = width * height

    style, style_recognized = _resolve_frame_style(frame_style)
    frame_cost = round(perimeter_cm * FRAME_STYLES[style], 2)

    medium_lowered = (medium or "").strip().lower()
    default_needs_glazing = any(kw in medium_lowered for kw in _PAPER_BASED_MEDIUM_KEYWORDS)
    needs_glazing = default_needs_glazing if glazing_override is None else bool(glazing_override)
    glazing_choice = "uv-protective acrylic" if needs_glazing else "none"
    glazing_cost = round(area_cm2 * GLAZING_OPTIONS[glazing_choice], 2) if needs_glazing else 0.0

    zone, destination_recognized = _resolve_zone(destination_country)
    area_m2 = area_cm2 / 10_000
    density = _DENSITY_GLAZED_KG_PER_M2 if needs_glazing else _DENSITY_UNGLAZED_KG_PER_M2
    weight_kg = round(_BASE_PACKAGE_WEIGHT_KG + area_m2 * density, 2)
    shipping_cost = round(
        _ZONE_BASE_RATE_USD[zone] + weight_kg * _ZONE_PER_KG_RATE_USD[zone], 2
    )

    subtotal = round(frame_cost + glazing_cost + shipping_cost, 2)

    return {
        "dimensions_cm": {"width": width, "height": height},
        "medium": medium,
        "frame": {"style": style, "style_recognized": style_recognized, "cost_usd": frame_cost},
        "glazing": {"chosen": glazing_choice, "needed": needs_glazing, "cost_usd": glazing_cost},
        "shipping": {
            "destination_country": destination_country, "zone": zone,
            "destination_recognized": destination_recognized,
            "estimated_weight_kg": weight_kg, "cost_usd": shipping_cost,
        },
        "subtotal_usd": subtotal,
        "currency": "USD",
        "disclaimer": (
            "Estimate only -- illustrative pricing for this coursework demo, "
            "not a real framing shop's rate card."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "error": None,
    }


if __name__ == "__main__":
    import json

    demo = compute_quote(
        width_cm=40.6, height_cm=50.8, medium="oil on canvas",
        destination_country="France",
    )
    print(json.dumps(demo, indent=2))
