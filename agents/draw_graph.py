"""
agents/draw_graph.py

Renders the REAL compiled LangGraph pipeline (agents/graph.py's own
build_graph()) to a diagram -- for the final report, the README, or
just eyeballing the wiring after a change to graph.py's edges.

Deliberately draws the ACTUAL CompiledStateGraph object build_graph()
returns, never a hand-drawn copy of its topology kept in a second file
-- a hand-drawn version would silently drift out of sync with the real
edges the first time graph.py changes and this file doesn't.

Needs the same live stack every other real entry point in this project
needs, because build_graph() calls build_specialists() internally,
which stands up the real MCP client + tool bindings and the real
per-specialist LLMs (see graph.py's own build_graph() docstring):
    - the MCP server reachable (stdio subprocess, or MCP_TRANSPORT=http
      against a running mcp-server container)
    - Ollama reachable, since build_specialists() builds every
      specialist's LLM up front, not lazily on first use
    - a corpus_meta document list, since corpus_meta_node's system
      prompt is rendered once, at build time, from a live document list

Run exactly the same way every other module-level script in this
project runs, from the project root (one level above agents/):
    py -3.12 -m agents.draw_graph
    py -3.12 -m agents.draw_graph --format png-local
    py -3.12 -m agents.draw_graph --format mermaid
    py -3.12 -m agents.draw_graph --format ascii

Output formats -- "png" is the default, specifically because a text/
ASCII diagram isn't what "draw the graph" means for a report or a
README; an actual image is:

- "png" (DEFAULT) -- an actual rendered image, via LangGraph's own
  draw_mermaid_png(draw_method=MermaidDrawMethod.API). No extra
  packages beyond what this project already installs (langchain-core is
  already a dependency of everything in agents/) -- but it round-trips
  the mermaid source through the public mermaid.ink API over HTTPS, so
  it needs outbound internet access from wherever this runs. Falls back
  to "mermaid" (below) on ANY failure -- offline machine, mermaid.ink
  being down, a corporate firewall -- rather than crashing the whole
  script over a diagram, and the fallback message tells you to try
  "png-local" next.
- "png-local" -- also an actual rendered image, same
  draw_mermaid_png() call, but with
  draw_method=MermaidDrawMethod.PYPPETEER: renders the diagram locally
  in a headless Chromium browser instead of calling mermaid.ink at all.
  Needs `pip install pyppeteer` (not in this project's requirements.txt
  by default, since it's a fairly heavy optional extra -- pyppeteer
  downloads its own ~150MB Chromium build the first time it runs, which
  itself needs internet access ONCE; every render after that is fully
  offline). Use this if "png" fails because mermaid.ink specifically is
  unreachable but you do have general internet access (or already have
  Chromium cached from a previous run).
- "mermaid" -- writes agents/graph.mmd, plain Mermaid text. No network
  call at all, works fully offline, no extra packages. Paste its
  contents into https://mermaid.live to view/export, or render it
  locally with the mermaid-cli npm package:
  `mmdc -i agents/graph.mmd -o agents/graph.png`. Both PNG formats
  above fall back to this if they fail, so this path has to work
  standalone regardless of network/package state.
- "ascii" -- text-art straight to the terminal (and saved to
  agents/graph.txt), via LangGraph's draw_ascii(). Needs the `grandalf`
  package (`pip install grandalf --break-system-packages`), not
  installed by default in this project's requirements -- degrades to
  "mermaid" if it's missing.

A THIRD real-image option exists in LangGraph itself
(Graph.draw_png(), backed by pygraphviz/Graphviz) but is deliberately
NOT wired up here -- it needs the Graphviz SYSTEM package installed
(not just a pip package; on Windows that's a separate installer, not
`pip install`), which is a much heavier ask than the two PNG options
above for the same end result. Worth knowing it exists if neither "png"
nor "png-local" works for you, but not implemented in this file.

IMPORTANT STRUCTURAL NOTE (a real bug this file's own structure avoids,
not a hypothetical): draw_mermaid_png(draw_method=MermaidDrawMethod.
PYPPETEER) calls asyncio.run() INTERNALLY to drive its own headless-
browser rendering (see langchain_core's own graph_mermaid.py source).
Calling it from CODE THAT IS ITSELF ALREADY RUNNING INSIDE an
asyncio.run() block raises "RuntimeError: asyncio.run() cannot be
called from a running event loop" -- a confirmed, commonly-hit failure
(see langchain-ai/langchain issue #26958) that trips people up
specifically when they try to render the PNG from inside the same
async function that built the graph. This file avoids it structurally:
build_graph() runs inside its OWN, short-lived asyncio.run() call
(_build_graph_sync() below), which returns control to fully
SYNCHRONOUS code before any drawing method -- mermaid API, pyppeteer,
or ascii -- is ever called. If you're adapting this script and moving
the drawing call back inside an async function, you will hit that
exact RuntimeError; keep the build and the draw on opposite sides of
one asyncio.run() boundary, not both inside it.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from agents.graph import build_graph

OUT_DIR = Path(__file__).resolve().parent


async def _build():
    print(
        "[draw_graph] building the real graph (build_specialists() -- needs "
        "Ollama and the MCP server reachable)...",
        file=sys.stderr,
    )
    graph = await build_graph()
    names = getattr(graph, "known_specialist_names", ())
    print(
        f"[draw_graph] graph built -- {len(names)} specialists: {', '.join(names)}",
        file=sys.stderr,
    )
    return graph


def _build_graph_sync():
    """
    The ONLY asyncio.run() call in this whole script -- runs to
    completion and returns before main() calls any drawing method. See
    this module's own top docstring ("IMPORTANT STRUCTURAL NOTE") for
    exactly why that ordering matters: draw_mermaid_png()'s own
    pyppeteer path calls asyncio.run() internally, which raises if
    there's already one running around it.
    """
    return asyncio.run(_build())


def _write_mermaid(graph) -> Path:
    mermaid_src = graph.get_graph().draw_mermaid()
    out_path = OUT_DIR / "graph.mmd"
    out_path.write_text(mermaid_src, encoding="utf-8")
    return out_path


def _write_png_api(graph) -> Path:
    from langchain_core.runnables.graph import MermaidDrawMethod

    out_path = OUT_DIR / "graph.png"
    try:
        png_bytes = graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
    except Exception as e:  # noqa: BLE001 -- any rendering/network failure degrades to mermaid text, never crashes the whole script
        print(
            f"[draw_graph] draw_mermaid_png(API) failed ({type(e).__name__}: {e}) -- "
            "this needs outbound internet access to mermaid.ink specifically. If you "
            "have general internet access, try `--format png-local` instead (renders "
            "locally via a headless browser, no mermaid.ink involved). Falling back to "
            "the offline .mmd text format for now.",
            file=sys.stderr,
        )
        return _write_mermaid(graph)
    out_path.write_bytes(png_bytes)
    return out_path


def _write_png_local(graph) -> Path:
    from langchain_core.runnables.graph import MermaidDrawMethod

    out_path = OUT_DIR / "graph.png"
    try:
        png_bytes = graph.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.PYPPETEER, output_file_path=str(out_path)
        )
    except ImportError:
        print(
            "[draw_graph] draw_mermaid_png(PYPPETEER) needs the `pyppeteer` package "
            "(pip install pyppeteer --break-system-packages) -- falling back to the "
            "offline .mmd text format instead.",
            file=sys.stderr,
        )
        return _write_mermaid(graph)
    except Exception as e:  # noqa: BLE001 -- e.g. first-run Chromium download failed for lack of network -- degrade, don't crash
        print(
            f"[draw_graph] draw_mermaid_png(PYPPETEER) failed ({type(e).__name__}: {e}) -- "
            "pyppeteer downloads its own ~150MB Chromium the first time it runs, which "
            "itself needs internet access once. Falling back to the offline .mmd text "
            "format instead.",
            file=sys.stderr,
        )
        return _write_mermaid(graph)
    # draw_mermaid_png already wrote out_path when output_file_path is
    # passed, but it also returns the same bytes -- writing again here
    # would be redundant, so this branch just confirms the path exists.
    if not out_path.exists():
        out_path.write_bytes(png_bytes)
    return out_path


def _write_ascii(graph) -> Path:
    try:
        ascii_art = graph.get_graph().draw_ascii()
    except ImportError:
        print(
            "[draw_graph] draw_ascii() needs the `grandalf` package "
            "(pip install grandalf --break-system-packages) -- falling back to "
            "the offline .mmd text format instead.",
            file=sys.stderr,
        )
        return _write_mermaid(graph)
    print(ascii_art)
    out_path = OUT_DIR / "graph.txt"
    out_path.write_text(ascii_art, encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw the real compiled InMind LangGraph pipeline."
    )
    parser.add_argument(
        "--format",
        choices=["png", "png-local", "mermaid", "ascii"],
        default="png",
        help="Output format (default: png -- an actual image, via mermaid.ink).",
    )
    args = parser.parse_args()

    # Build first, entirely inside its own asyncio.run() call -- see
    # this module's own top docstring for why the draw step below must
    # run AFTER that call has already returned, in plain synchronous
    # code, not nested inside it.
    graph = _build_graph_sync()

    if args.format == "png":
        out_path = _write_png_api(graph)
    elif args.format == "png-local":
        out_path = _write_png_local(graph)
    elif args.format == "mermaid":
        out_path = _write_mermaid(graph)
    else:
        out_path = _write_ascii(graph)

    print(f"[draw_graph] wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
