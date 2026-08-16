"""
Color-palette generation backing the new `generate_color_palette` MCP
tool (see server.py for the tool wrapper, agents/specialists.py's
color_palette_node for the specialist that calls it).

Deliberately NOT an LLM call anywhere in this module, same "structural
guardrail over prompt wording" reasoning invoice_tools.py's own module
docstring already gives for its own arithmetic: every hex/rgb value, every
scheme's hue math, and every "closest named color" lookup is plain,
reproducible Python (stdlib `colorsys` for the HSL conversions), never an
LLM's own guess at what "#3f7cac" or "a triadic scheme starting from
cerulean" resolves to. The two things that AREN'T pure math -- what a
color is actually called, and what feeling it might inspire -- are both
curated, honestly-a-heuristic lookup tables (named colors, hue->feeling
buckets, mood keyword->hue), the same documented tradeoff
specialists.py's own `_classify_tier` already makes for its
beginner/professional keyword cues: good enough to be useful, not a
substitute for real perceptual-color research or NLU.

Named colors come from two sources, merged, so this works either way the
person set the project up (matches the "either a library or a built-in
dictionary" framing this feature was asked for):
  1. `_FALLBACK_NAMED_COLORS` below -- always available, no dependency,
     leans toward names an artist would actually reach for (ochre,
     cerulean, viridian, sienna) that CSS's own named-color set doesn't
     have.
  2. `webcolors`'s CSS3 name set, if the package is installed -- adds
     ~150 more standard names. Entirely optional: this module works
     exactly the same, just with a smaller name vocabulary, if
     `webcolors` isn't installed. See `_named_color_table`'s own
     docstring for exactly how the two are merged.
"""

import base64
import math
import re
import sys
from typing import Optional


def _log(msg: str) -> None:
    print(f"[color_tools] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------
# Named colors
# ---------------------------------------------------------------------

# A deliberately painting-flavored set -- primaries/secondaries/tertiaries
# per the color wheel, common tints/shades, and pigment-style names an
# artist is more likely to type or want to see than a CSS keyword. Always
# available with zero dependencies; the single source of truth if
# `webcolors` isn't installed, and always merged in ahead of it either way
# (see `_named_color_table`) since these names are more relevant to this
# project's subject matter than CSS's own set.
_FALLBACK_NAMED_COLORS = {
    "black": "#000000", "white": "#ffffff", "gray": "#808080", "grey": "#808080",
    "silver": "#c0c0c0", "charcoal": "#36454f", "ivory": "#fffff0",
    "beige": "#f5f5dc", "cream": "#fffdd0", "taupe": "#483c32",
    "red": "#ff0000", "crimson": "#dc143c", "scarlet": "#ff2400", "maroon": "#800000",
    "ruby": "#e0115f", "vermilion": "#e34234", "brick red": "#b22222", "carmine": "#960018",
    "orange": "#ffa500", "coral": "#ff7f50", "peach": "#ffcba4", "amber": "#ffbf00",
    "rust": "#b7410e", "tangerine": "#f28500", "burnt orange": "#cc5500",
    "yellow": "#ffff00", "gold": "#ffd700", "mustard": "#ffdb58", "lemon": "#fff44f",
    "ochre": "#cc7722", "khaki": "#f0e68c", "saffron": "#f4c430",
    "green": "#008000", "emerald": "#50c878", "olive": "#808000", "mint": "#98ff98",
    "sage": "#9caf88", "forest green": "#228b22", "lime": "#32cd32", "viridian": "#40826d",
    "moss": "#8a9a5b", "jade": "#00a86b",
    "teal": "#008080", "turquoise": "#40e0d0", "cyan": "#00ffff", "aquamarine": "#7fffd4",
    "blue": "#0000ff", "navy": "#000080", "azure": "#007fff", "cobalt": "#0047ab",
    "cerulean": "#007ba7", "sky blue": "#87ceeb", "ultramarine": "#3f00ff",
    "sapphire": "#0f52ba", "steel blue": "#4682b4", "powder blue": "#b0e0e6",
    "purple": "#800080", "violet": "#8f00ff", "lavender": "#e6e6fa", "indigo": "#4b0082",
    "plum": "#8e4585", "orchid": "#da70d6", "lilac": "#c8a2c8", "amethyst": "#9966cc",
    "pink": "#ffc0cb", "magenta": "#ff00ff", "fuchsia": "#ff00ff", "rose": "#ff007f",
    "salmon": "#fa8072", "blush": "#de5d83",
    "brown": "#964b00", "sienna": "#a0522d", "umber": "#635147", "chestnut": "#954535",
    "chocolate": "#7b3f00", "tan": "#d2b48c", "sepia": "#704214", "terracotta": "#e2725b",
    "burgundy": "#800020", "wine": "#722f37",
}

_NAMED_COLOR_TABLE_CACHE: Optional[dict] = None


def _named_color_table() -> dict:
    """
    Lazily built, cached (module-level -- built once per process) merge
    of `_FALLBACK_NAMED_COLORS` with `webcolors`'s CSS3 name set, if that
    package is importable. `_FALLBACK_NAMED_COLORS` is inserted FIRST and
    never overwritten (see the `setdefault` below), so this project's own
    painting-relevant names always win over anything CSS3 might also
    define for the same word.

    Wrapped in a broad try/except, not just an ImportError guard: this
    is only ever a "nicer names" enhancement, never something the rest of
    this module can't function without, so ANY failure while importing
    or reading from webcolors (a version mismatch, a renamed API, etc.)
    degrades to the fallback dictionary alone rather than raising --
    same "an optional dependency's failure never breaks the required
    path" convention server.py's own _import_optional_tool_module
    already applies at the module-import level, applied here one level
    down, to a single optional feature within an always-available module.
    """
    global _NAMED_COLOR_TABLE_CACHE
    if _NAMED_COLOR_TABLE_CACHE is not None:
        return _NAMED_COLOR_TABLE_CACHE

    table = dict(_FALLBACK_NAMED_COLORS)
    try:
        import webcolors

        for name in webcolors.names(spec="css3"):
            table.setdefault(name, webcolors.name_to_hex(name, spec="css3"))
    except Exception as exc:  # noqa: BLE001 -- optional dependency, never fatal
        _log(
            f"webcolors unavailable or incompatible ({exc!r}) -- using the "
            f"built-in {len(table)}-name dictionary only"
        )

    _NAMED_COLOR_TABLE_CACHE = table
    return table


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _lookup_named_color(text: str) -> Optional[tuple[int, int, int]]:
    """
    Exact match first, then the longest named color that appears as a
    whole phrase inside `text` (so "sky blue" wins over a bare "blue"
    match when both are present) -- same "longest/most specific match
    wins" preference specialists.py's own product-search tier matching
    already uses. Returns None if no known name appears at all.
    """
    normalized = _normalize_text(text)
    table = _named_color_table()

    if normalized in table:
        return hex_to_rgb(table[normalized])

    candidates = [name for name in table if re.search(rf"\b{re.escape(name)}\b", normalized)]
    if not candidates:
        return None
    best = max(candidates, key=len)
    return hex_to_rgb(table[best])


def closest_color_name(r: int, g: int, b: int) -> str:
    """
    The named color (from `_named_color_table`) perceptually closest to
    (r, g, b), by "redmean" weighted distance -- a well-known, cheap
    approximation of perceptual color difference that weights each
    channel by how sensitive the human eye is to it in that part of the
    red/green range, rather than plain unweighted Euclidean distance
    (which tends to call slightly-different greens "closer" to each
    other than they actually look, relative to red/blue differences of
    the same numeric size).
    """
    table = _named_color_table()
    best_name, best_dist = None, float("inf")
    for name, hex_value in table.items():
        nr, ng, nb = hex_to_rgb(hex_value)
        rmean = (r + nr) / 2
        dr, dg, db = r - nr, g - ng, b - nb
        dist = (2 + rmean / 256) * dr * dr + 4 * dg * dg + (2 + (255 - rmean) / 256) * db * db
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name.title() if best_name else "Unnamed"


# ---------------------------------------------------------------------
# hex / rgb / hsl conversions
# ---------------------------------------------------------------------


def _clamp255(v: float) -> int:
    return max(0, min(255, round(v)))


def hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    h = hex_str.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return "#{:02x}{:02x}{:02x}".format(_clamp255(r), _clamp255(g), _clamp255(b))


def rgb_to_hsl(r: int, g: int, b: int) -> tuple[float, float, float]:
    """(hue in degrees 0-360, saturation 0-1, lightness 0-1) -- wraps
    colorsys.rgb_to_hls with the argument order fixed so callers never
    have to remember colorsys's own HLS (not HSL) ordering."""
    import colorsys

    h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    return h * 360, s, l


def hsl_to_rgb(h: float, s: float, l: float) -> tuple[int, int, int]:
    import colorsys

    r, g, b = colorsys.hls_to_rgb((h % 360) / 360, l, s)
    return _clamp255(r * 255), _clamp255(g * 255), _clamp255(b * 255)


# ---------------------------------------------------------------------
# Parsing a color given as text (hex / rgb triplet / name)
# ---------------------------------------------------------------------

_HEX_RE = re.compile(r"#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?\b")
_RGB_RE = re.compile(
    r"rgb\s*\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)"
    r"|\b(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\b"
)


def parse_color_text(text: str) -> Optional[tuple[int, int, int]]:
    """
    Best-effort extraction of an (r, g, b) triplet from free text -- a
    hex code (`#3f7cac` or `#fff`), an rgb triplet (`rgb(63, 124, 172)`
    or a bare `63, 124, 172`), or a known color name (exact or the
    longest name found as a substring, see `_lookup_named_color`).
    Checked in that order since a hex/rgb value is unambiguous where a
    name lookup could theoretically collide with common words. Returns
    None if nothing recognizable is found -- never raises, never guesses.
    """
    if not text or not text.strip():
        return None

    m = _HEX_RE.search(text)
    if m:
        return hex_to_rgb(m.group(0))

    m = _RGB_RE.search(text)
    if m:
        groups = [g for g in m.groups() if g is not None]
        r, g, b = (int(x) for x in groups)
        if all(0 <= v <= 255 for v in (r, g, b)):
            return (r, g, b)

    return _lookup_named_color(text)


# ---------------------------------------------------------------------
# Hue -> feeling associations (forward direction: color -> mood)
# ---------------------------------------------------------------------

# (hue upper bound in degrees, family label, one-line feeling
# description). Checked in order, first bound the hue falls under wins --
# 12 buckets of 30 degrees each around the wheel, roughly centered on
# each primary/secondary/tertiary hue. Honest limitation worth stating
# plainly, same spirit as _classify_tier's own docstring: color-emotion
# association is genuinely subjective and culturally variable -- this is
# one reasonable, commonly-cited reading per hue family, not a claim
# about how any individual person will feel looking at it.
_HUE_FEELINGS: list[tuple[float, str, str]] = [
    (15, "Red", "Bold and intense -- reads as passion, urgency, or raw energy."),
    (45, "Orange", "Warm and inviting -- reads as enthusiasm, playfulness, and comfort."),
    (70, "Yellow", "Bright and cheerful -- reads as optimism, energy, and attention."),
    (100, "Yellow-green", "Fresh and lively -- reads as renewal, growth, and youthful energy."),
    (160, "Green", "Natural and settled -- reads as calm, balance, and steady growth."),
    (195, "Teal/Cyan", "Clear and composed -- reads as clarity, sophistication, and quiet focus."),
    (250, "Blue", "Cool and steady -- reads as calm and trust, sometimes tipping into melancholy."),
    (275, "Indigo", "Deep and contemplative -- reads as depth, mystery, and quiet intensity."),
    (320, "Purple", "Rich and evocative -- reads as luxury, imagination, and mystery."),
    (345, "Magenta/Pink", "Soft and expressive -- reads as romance, tenderness, and playfulness."),
    (360.01, "Red", "Bold and intense -- reads as passion, urgency, or raw energy."),
]


def _hue_feeling(h: float) -> tuple[str, str]:
    for upper, family, feeling in _HUE_FEELINGS:
        if h < upper:
            return family, feeling
    return _HUE_FEELINGS[-1][1], _HUE_FEELINGS[-1][2]


def describe_feeling(h: float, s: float, l: float) -> tuple[str, str]:
    """
    (family label, feeling sentence) for an HSL color -- overridden with
    a neutral reading near the achromatic extremes (very light, very
    dark, or very low saturation) before falling through to the hue
    bucket, since "hue" barely means anything perceptually once
    saturation or lightness is at an extreme (a hue of 210 at 2%
    saturation looks gray, not blue).
    """
    if l < 0.08:
        return "Black", "Grounded and dramatic -- reads as power, elegance, or solemnity."
    if l > 0.94:
        return "White", "Open and clean -- reads as purity, simplicity, and calm."
    if s < 0.12:
        return "Gray", "Neutral and understated -- reads as balance, restraint, or quiet melancholy."
    return _hue_feeling(h)


# ---------------------------------------------------------------------
# Swatch rendering -- a tiny inline SVG square, base64-encoded as a
# data: URI
# ---------------------------------------------------------------------


def swatch_data_uri(hex_color: str, size: int = 40) -> str:
    """
    A small rounded-square SVG filled with `hex_color`, encoded as a
    `data:image/svg+xml;base64,...` URI so it can be dropped straight
    into markdown as `![alt](uri)` with no static file server involved --
    the same self-contained approach mcp_server/image_tools.py's own
    retrieve_images_with_data/format_markdown_image_embedded already
    uses for corpus images (see that module's docstring). The frontend's
    MarkdownText.tsx already allowlists `data:image/*;base64` src values
    specifically to unblock that image path, so this needed no frontend
    change to render.
    """
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}">'
        f'<rect width="{size}" height="{size}" rx="6" fill="{hex_color}" '
        f'stroke="#00000033" stroke-width="1"/></svg>'
    )
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def describe_color(r: int, g: int, b: int) -> dict:
    """
    One color's full descriptor -- hex, rgb, closest name, hue family,
    feeling sentence, and a ready-to-embed swatch image. Every color this
    module ever returns (the base color and every scheme member) is
    shaped through this one function, so a caller never has to build this
    dict by hand at more than one call site.
    """
    hex_value = rgb_to_hex(r, g, b)
    h, s, l = rgb_to_hsl(r, g, b)
    family, feeling = describe_feeling(h, s, l)
    return {
        "hex": hex_value,
        "rgb": {"r": r, "g": g, "b": b},
        "name": closest_color_name(r, g, b),
        "family": family,
        "feeling": feeling,
        "swatch": swatch_data_uri(hex_value),
    }


# ---------------------------------------------------------------------
# Color schemes -- built in HSL space around a base (h, s, l)
# ---------------------------------------------------------------------


def scheme_monochromatic(h: float, s: float, l: float) -> list[tuple[float, float, float]]:
    """
    One hue, five lightness steps -- the base color's OWN lightness is
    always one of the five (the fixed stop nearest to it is replaced with
    the exact base value, rather than the base being approximated by
    whichever fixed stop happens to be close), so a monochromatic palette
    always includes the color the person actually asked for, not just
    colors near it. Saturation is pulled down a little at the two
    darkest/lightest stops -- a fully saturated hue at 15% or 85%
    lightness tends to look like a garish neon rather than a usable tint
    or shade.
    """
    stops = [0.85, 0.65, 0.50, 0.35, 0.15]
    closest_idx = min(range(len(stops)), key=lambda i: abs(stops[i] - l))
    stops[closest_idx] = l

    out = []
    for lv in stops:
        sv = s if 0.12 <= lv <= 0.88 else max(0.2, s * 0.6)
        out.append((h, sv, lv))
    return out


def scheme_analogous(h: float, s: float, l: float) -> list[tuple[float, float, float]]:
    return [((h - 30) % 360, s, l), (h, s, l), ((h + 30) % 360, s, l)]


def scheme_complementary(h: float, s: float, l: float) -> list[tuple[float, float, float]]:
    return [(h, s, l), ((h + 180) % 360, s, l)]


def scheme_triadic(h: float, s: float, l: float) -> list[tuple[float, float, float]]:
    return [(h, s, l), ((h + 120) % 360, s, l), ((h + 240) % 360, s, l)]


_SCHEME_BUILDERS = {
    "monochromatic": scheme_monochromatic,
    "analogous": scheme_analogous,
    "complementary": scheme_complementary,
    "triadic": scheme_triadic,
}

_SCHEME_ALIASES = {
    "mono": "monochromatic", "monochrome": "monochromatic",
    "analog": "analogous",
    "complement": "complementary", "complements": "complementary", "opposite": "complementary",
    "triad": "triadic", "triadal": "triadic",
}


def normalize_scheme_name(scheme: Optional[str]) -> Optional[str]:
    """A recognized scheme name, or None -- an unrecognized value (typo,
    or a scheme this module doesn't implement) is treated exactly like
    "no scheme specified" by the caller (generate_palette), i.e. ALL
    schemes come back, rather than silently returning nothing for a
    request that clearly wanted something."""
    if not scheme:
        return None
    key = scheme.strip().lower()
    key = _SCHEME_ALIASES.get(key, key)
    return key if key in _SCHEME_BUILDERS else None


# ---------------------------------------------------------------------
# Mood -> color (reverse direction)
# ---------------------------------------------------------------------

# mood/atmosphere keyword -> (hue degrees, saturation 0-1, lightness 0-1),
# a reasonable, moderately vivid starting point for that mood -- not
# claimed to be the ONLY color that could express it. Neighboring moods
# deliberately share neighboring hues (the reds, then oranges, then
# yellows, working around the wheel) so a request naming several related
# moods averages to a sensible color rather than a random one -- see
# `color_from_mood`'s own docstring for how multiple matches combine.
_MOOD_KEYWORDS: dict[str, tuple[float, float, float]] = {
    # reds
    "passion": (355, 0.75, 0.45), "passionate": (355, 0.75, 0.45),
    "love": (345, 0.70, 0.55), "romance": (345, 0.55, 0.65), "romantic": (345, 0.55, 0.65),
    "anger": (0, 0.80, 0.42), "angry": (0, 0.80, 0.42), "danger": (5, 0.80, 0.40),
    "bold": (5, 0.75, 0.45), "powerful": (5, 0.60, 0.30), "intense": (355, 0.75, 0.40),
    "energetic": (10, 0.80, 0.50), "energy": (10, 0.80, 0.50), "vibrant": (15, 0.80, 0.55),
    # oranges
    "warm": (30, 0.70, 0.55), "warmth": (30, 0.70, 0.55), "cozy": (28, 0.55, 0.55),
    "playful": (28, 0.75, 0.60), "friendly": (30, 0.60, 0.60), "enthusiasm": (25, 0.75, 0.55),
    "autumn": (25, 0.60, 0.45), "harvest": (30, 0.55, 0.45),
    # yellows
    "happy": (50, 0.85, 0.60), "happiness": (50, 0.85, 0.60), "cheerful": (50, 0.85, 0.62),
    "joyful": (48, 0.85, 0.60), "joy": (48, 0.85, 0.60), "optimistic": (50, 0.80, 0.60),
    "sunny": (48, 0.90, 0.60), "bright": (50, 0.85, 0.60),
    # yellow-greens
    "fresh": (85, 0.55, 0.50), "spring": (95, 0.50, 0.55), "youthful": (90, 0.55, 0.55),
    "renewal": (100, 0.50, 0.45),
    # greens
    "nature": (130, 0.45, 0.40), "natural": (130, 0.40, 0.40), "earthy": (100, 0.35, 0.35),
    "calm": (150, 0.35, 0.50), "peaceful": (150, 0.30, 0.55), "serene": (170, 0.30, 0.60),
    "growth": (140, 0.50, 0.42), "envy": (135, 0.55, 0.35), "jealous": (135, 0.55, 0.35),
    "balanced": (145, 0.30, 0.45),
    # teal/cyan
    "clarity": (185, 0.50, 0.50), "sophisticated": (190, 0.35, 0.35), "refreshing": (185, 0.50, 0.55),
    "tropical": (175, 0.65, 0.50), "crisp": (190, 0.45, 0.55),
    # blues
    "sad": (215, 0.35, 0.40), "sadness": (215, 0.35, 0.40), "melancholy": (220, 0.30, 0.35),
    "cool": (210, 0.45, 0.55), "trust": (210, 0.50, 0.45), "serenity": (205, 0.35, 0.60),
    "cold": (210, 0.30, 0.50), "icy": (200, 0.30, 0.65), "lonely": (220, 0.25, 0.35),
    "quiet": (210, 0.25, 0.50), "reflective": (215, 0.35, 0.40), "introspective": (225, 0.30, 0.35),
    # indigo/violet
    "mysterious": (265, 0.45, 0.30), "mystery": (265, 0.45, 0.30), "dreamy": (255, 0.40, 0.55),
    "spiritual": (260, 0.35, 0.40), "night": (250, 0.40, 0.20), "deep": (255, 0.40, 0.30),
    # purples
    "luxurious": (285, 0.45, 0.35), "luxury": (285, 0.45, 0.35), "royal": (270, 0.55, 0.35),
    "elegant": (280, 0.30, 0.30), "creative": (280, 0.50, 0.50), "magical": (275, 0.50, 0.50),
    "regal": (270, 0.50, 0.30),
    # pinks
    "gentle": (330, 0.40, 0.70), "soft": (330, 0.35, 0.75), "tender": (340, 0.45, 0.70),
    "sweet": (335, 0.55, 0.70), "innocent": (330, 0.30, 0.80), "delicate": (335, 0.30, 0.78),
    # neutrals / other atmospheres
    "minimal": (0, 0.0, 0.85), "clean": (0, 0.0, 0.90), "pure": (0, 0.0, 0.92),
    "somber": (230, 0.15, 0.25), "gloomy": (225, 0.20, 0.30), "dramatic": (0, 0.05, 0.15),
    "vintage": (35, 0.30, 0.50), "nostalgic": (35, 0.30, 0.55), "rustic": (30, 0.35, 0.40),
    "moody": (240, 0.25, 0.25),
}


def color_from_mood(mood_text: str) -> Optional[dict]:
    """
    Best-effort, deterministic keyword match: which `_MOOD_KEYWORDS`
    entries appear (as a substring, case-insensitive) in `mood_text`,
    combined into one representative color. Returns None if literally
    nothing matched, so the caller reports that plainly instead of
    guessing a color from zero signal -- same "never fabricate" pattern
    invoice_tools.py/image_tools.py both already apply to their own
    empty-result cases.

    Combining multiple matches: hue is a CIRCULAR mean (vector average
    of each matched hue's position on the wheel, via atan2), since a
    plain arithmetic mean of e.g. 350 degrees and 10 degrees would wrongly
    average to 180 (green) instead of 0 (red) -- both are "reddish," a
    few degrees apart, on opposite sides of the wrap-around. Saturation
    and lightness are plain arithmetic means; neither wraps.

    Returns {"rgb": (r,g,b), "matched_keywords": [kw, ...]} on success.
    """
    normalized = _normalize_text(mood_text)
    matched = [(kw, hsl) for kw, hsl in _MOOD_KEYWORDS.items() if kw in normalized]
    if not matched:
        return None

    sin_sum = sum(math.sin(math.radians(h)) for _, (h, _, _) in matched)
    cos_sum = sum(math.cos(math.radians(h)) for _, (h, _, _) in matched)
    mean_hue = math.degrees(math.atan2(sin_sum, cos_sum)) % 360
    mean_s = sum(s for _, (_, s, _) in matched) / len(matched)
    mean_l = sum(l for _, (_, _, l) in matched) / len(matched)

    r, g, b = hsl_to_rgb(mean_hue, mean_s, mean_l)
    return {"rgb": (r, g, b), "matched_keywords": [kw for kw, _ in matched]}


# ---------------------------------------------------------------------
# Top-level orchestrator -- the function behind the generate_color_palette
# MCP tool
# ---------------------------------------------------------------------


def _error_result(message: str) -> dict:
    return {
        "input_type": None,
        "resolved_from_mood": None,
        "base_color": None,
        "schemes": {},
        "error": message,
    }


def generate_palette(
    color: Optional[str] = None,
    mood: Optional[str] = None,
    scheme: Optional[str] = None,
) -> dict:
    """
    Build a color palette from either an explicit `color` (hex/rgb/name
    text) or a `mood`/feeling description, and one or all four schemes
    (monochromatic, analogous, complementary, triadic).

    If both `color` and `mood` are given, `color` wins -- an explicit
    color is unambiguous where a mood match is always a best-effort
    guess, the same priority agents/specialists.py's color_palette_node
    already gives when deciding which one to send in the first place.
    Neither resolving to a usable color returns an `error` string rather
    than guessing or raising.

    `scheme`, if given, is normalized via `normalize_scheme_name`; an
    unrecognized value is treated the same as omitted (ALL FOUR schemes
    come back) rather than silently returning nothing.

    Returns:
      {
        "input_type": "color" | "mood" | None,
        "resolved_from_mood": [matched keyword, ...] | None,
        "base_color": {hex, rgb, name, family, feeling, swatch} | None,
        "schemes": {scheme_name: [ {hex, rgb, name, family, feeling,
                                     swatch}, ... ], ...},
        "error": str | None,
      }
    Never raises.
    """
    resolved_from_mood = None

    if color:
        rgb = parse_color_text(color)
        input_type = "color"
        if rgb is None:
            return _error_result(
                f"I couldn't recognize {color!r} as a color -- try a hex code "
                "like #3f7cac, an rgb triplet like 63, 124, 172, or a common "
                "color name like \"cerulean\" or \"forest green\"."
            )
    elif mood:
        match = color_from_mood(mood)
        input_type = "mood"
        if match is None:
            return _error_result(
                f"I couldn't connect {mood!r} to a color -- try naming a "
                "clearer mood or feeling (e.g. \"calm and peaceful\", \"bold "
                "and dramatic\", \"warm and cozy\")."
            )
        rgb = match["rgb"]
        resolved_from_mood = match["matched_keywords"]
    else:
        return _error_result(
            "I need either a color (a hex code, rgb triplet, or color name) "
            "or a mood/feeling to build a palette from."
        )

    r, g, b = rgb
    base = describe_color(r, g, b)
    h, s, l = rgb_to_hsl(r, g, b)

    resolved_scheme = normalize_scheme_name(scheme)
    scheme_names = [resolved_scheme] if resolved_scheme else list(_SCHEME_BUILDERS.keys())

    schemes = {}
    for name in scheme_names:
        hsl_members = _SCHEME_BUILDERS[name](h, s, l)
        schemes[name] = [describe_color(*hsl_to_rgb(*member)) for member in hsl_members]

    return {
        "input_type": input_type,
        "resolved_from_mood": resolved_from_mood,
        "base_color": base,
        "schemes": schemes,
        "error": None,
    }


if __name__ == "__main__":
    demo_color = generate_palette(color="cerulean", scheme="triadic")
    print(f"cerulean, triadic only -> base={demo_color['base_color']['hex']} "
          f"({demo_color['base_color']['name']})")
    for item in demo_color["schemes"]["triadic"]:
        print(f"  {item['hex']}  {item['name']:<20}  {item['feeling']}")

    print()
    demo_mood = generate_palette(mood="calm and peaceful, a little mysterious")
    print(f"mood='calm and peaceful, a little mysterious' -> matched "
          f"{demo_mood['resolved_from_mood']}, base={demo_mood['base_color']['hex']} "
          f"({demo_mood['base_color']['name']})")
    for scheme_name, items in demo_mood["schemes"].items():
        print(f"  {scheme_name}: {[it['hex'] for it in items]}")

    print()
    demo_bad = generate_palette(color="not a real color at all")
    print(f"unrecognized color -> error={demo_bad['error']!r}")
