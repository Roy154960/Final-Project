"""
run_all_evaluations.py -- one entry point for every evaluation in this
project EXCEPT local_rag's own evaluation suite.

Scope, on purpose:
    INCLUDED  agents/eval_routing.py    (routing accuracy + confusion matrix,
                                          94 questions incl. the hard/multilingual
                                          ones added alongside this script)
              agents/eval_language.py   (routing-across-languages + end-to-end
                                          language fidelity)
              agents/eval_phase5.py     (10 designed queries through the real,
                                          compiled graph -- routing + answer
                                          material for a human correctness read)
    EXCLUDED  local_rag/evaluation/     (ragas_eval.py, build_eval_set.py,
                                          metrics.py) -- this is its own
                                          complete, separately-run evaluation
                                          suite for the local RAG pipeline
                                          specifically (RAGAS faithfulness/
                                          relevance/precision/recall), with its
                                          own report format and its own way of
                                          being invoked (`python -m
                                          evaluation.ragas_eval` from inside
                                          local_rag/). It is not touched,
                                          wrapped, or duplicated here.

This script does NOT reimplement any evaluation logic. It only orchestrates:
runs each of the three eval_*.py modules above exactly the way their own
docstrings already say to run them (as a subprocess, `python -m agents.X`,
from the project root -- see e.g. eval_phase5.py's own module docstring for
why: those scripts do `from agents import ...`, which only resolves with the
project root on sys.path), then copies the result files each one ALREADY
writes (into agents/, next to itself, unchanged) into one combined,
timestamped folder at the project root -- because right now they're three
separate write locations and nothing puts them side by side for a single
"here's the state of the pipeline" read.

Prerequisites (same as running each eval_*.py directly -- see each one's own
module docstring for the full detail):
    - eval_routing.py:            a running Ollama only.
    - eval_language.py (part 1):  a running Ollama only.
    - eval_language.py (--full):  Ollama + the MCP server + an ingested corpus.
    - eval_phase5.py:             Ollama + the MCP server + an ingested corpus.

Run, from the project root:
    python run_all_evaluations.py            # everything, needs the full live
                                              # stack (Ollama + MCP server +
                                              # an ingested corpus)
    python run_all_evaluations.py --fast     # routing-only: eval_routing.py
                                              # and eval_language.py's Part 1
                                              # only -- skips eval_phase5.py
                                              # and eval_language.py's Part 2
                                              # entirely, since only those two
                                              # need the corpus/MCP server.
                                              # Ollama itself is still required
                                              # either way (the supervisor's
                                              # own routing calls need it).

A run that partially fails still writes what it has: if eval_phase5.py's
Ollama call errors out, eval_routing.py's and eval_language.py's results
(already written, from before the failure) are still collected. Check
SUMMARY.md's per-evaluation status column, and run_log.txt for the full
captured stdout/stderr of whichever one failed, before assuming a clean run.

Writes:
    evaluation_results/run_<YYYYMMDD_HHMMSS>/
        eval_routing_results.md / eval_routing_results.json
        eval_language_results.md / eval_language_results.json
        eval_results.md / eval_results.json         (eval_phase5.py's own
                                                       output filenames --
                                                       unchanged, so a diff
                                                       against a file you
                                                       already have from a
                                                       manual run still works)
        routing_confusion_matrix.png                (copied from
                                                       screenshots/, if
                                                       eval_routing.py wrote
                                                       one this run)
        SUMMARY.md                                   (pass/fail per
                                                       evaluation, this run)
        run_log.txt                                  (full stdout/stderr of
                                                       every subprocess)

Nothing here modifies agents/eval_routing.py, agents/eval_language.py,
agents/eval_phase5.py, or any local_rag/ file. The three eval_*.py modules
still also write their own copies of their results next to themselves in
agents/, exactly as before -- this script only adds a second, combined copy
on top of that, it doesn't replace or relocate the originals.
"""

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AGENTS_DIR = ROOT / "agents"
SCREENSHOTS_DIR = ROOT / "screenshots"

EVALUATIONS = [
    {
        "name": "eval_routing",
        "module": "agents.eval_routing",
        "needs_full_stack": False,
        "fast_args": [],
        "full_args": [],
        "outputs": [
            AGENTS_DIR / "eval_routing_results.md",
            AGENTS_DIR / "eval_routing_results.json",
        ],
        "extra_outputs": [
            SCREENSHOTS_DIR / "routing_confusion_matrix.png",
        ],
    },
    {
        "name": "eval_language",
        "module": "agents.eval_language",
        "needs_full_stack": False,  # Part 1 (routing-across-languages) is fast;
                                     # --full also runs Part 2 (needs the corpus).
        "fast_args": [],
        "full_args": ["--full"],
        "outputs": [
            AGENTS_DIR / "eval_language_results.md",
            AGENTS_DIR / "eval_language_results.json",
        ],
        "extra_outputs": [],
    },
    {
        "name": "eval_phase5",
        "module": "agents.eval_phase5",
        "needs_full_stack": True,  # always needs Ollama + MCP server + corpus
        "fast_args": None,         # None => skipped entirely in --fast mode
        "full_args": [],
        "outputs": [
            AGENTS_DIR / "eval_results.md",
            AGENTS_DIR / "eval_results.json",
        ],
        "extra_outputs": [],
    },
]


def _run_one(ev: dict, fast: bool, out_dir: Path, log_lines: list) -> str:
    """
    Runs one evaluation module as a subprocess, copies whatever output
    files it wrote into out_dir (even on a non-zero exit -- a script that
    crashed partway through may still have written a partial, still-
    useful report), and returns a short status string for SUMMARY.md.
    """
    if fast and ev["fast_args"] is None:
        msg = f"[skip] {ev['name']}: needs the full live stack (Ollama + MCP server + corpus); skipped in --fast mode."
        print(msg)
        log_lines.append(msg)
        return "SKIPPED (--fast, needs full stack)"

    args = ev["fast_args"] if fast else ev["full_args"]
    cmd = [sys.executable, "-m", ev["module"], *args]
    print(f"[run] {' '.join(cmd)}")
    log_lines.append(f"\n=== {ev['name']} ===\ncmd: {' '.join(cmd)}")

    # Record each expected output's mtime BEFORE running -- a prior manual
    # run (or a prior call of this script) may have already left a file at
    # this exact path. If this subprocess fails before ever reaching its
    # own write step, that old file is still sitting there afterwards and
    # would otherwise get silently copied into this run's folder looking
    # exactly like a fresh result.
    all_outputs = ev["outputs"] + ev.get("extra_outputs", [])
    before_mtimes = {p: (p.stat().st_mtime if p.exists() else None) for p in all_outputs}

    try:
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired as exc:
        log_lines.append(f"TIMED OUT after {exc.timeout}s\n--- stdout so far ---\n{exc.stdout}\n--- stderr so far ---\n{exc.stderr}")
        status = "TIMED OUT"
        result = None
    else:
        log_lines.append(
            f"returncode: {result.returncode}\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
        status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"

    # Copy whatever output files exist, regardless of exit code -- a
    # crash after the routing loop but before the very last write can
    # still leave a previous run's file sitting there stale, or (for
    # eval_phase5.py/eval_language.py, which write their .md/.json only
    # once at the very end) leave nothing new at all. Either way, copy
    # what's actually on disk right now and say so, rather than silently
    # skipping.
    any_stale = False
    for src in all_outputs:
        if not src.exists():
            log_lines.append(f"[warn] expected output not found (evaluation may have crashed before writing it): {src}")
            continue
        is_stale = before_mtimes[src] is not None and src.stat().st_mtime == before_mtimes[src]
        dest_name = f"STALE_{src.name}" if is_stale else src.name
        dest = out_dir / dest_name
        shutil.copy2(src, dest)
        if is_stale:
            any_stale = True
            log_lines.append(
                f"[warn] {src} was NOT modified by this run (same mtime as before) -- this evaluation "
                f"likely crashed before writing it. Copied anyway, prefixed STALE_, as {dest} -- treat "
                f"it as a leftover from a previous manual run, not this run's result."
            )
        else:
            log_lines.append(f"copied {src} -> {dest}")

    if any_stale and status == "OK":
        status = "OK (but some outputs were stale -- see run_log.txt)"

    return status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run every non-local_rag evaluation in this project and save the "
                     "combined results together under evaluation_results/."
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Routing-only: eval_routing.py + eval_language.py's Part 1. Skips "
             "eval_phase5.py and eval_language.py's Part 2 entirely, since only "
             "those two need the MCP server and an ingested corpus. Ollama is "
             "still required either way.",
    )
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "evaluation_results" / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run_all_evaluations] writing combined results to {out_dir}")
    print("[run_all_evaluations] excludes local_rag/evaluation/ -- that suite is "
          "run and reported separately (see this script's own module docstring).\n")

    log_lines: list = []
    summary_rows: list = []

    for ev in EVALUATIONS:
        status = _run_one(ev, args.fast, out_dir, log_lines)
        summary_rows.append((ev["name"], status))

    (out_dir / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")

    summary_lines = [
        f"# Combined evaluation run -- {run_id}",
        "",
        "Excludes local_rag's own evaluation suite (local_rag/evaluation/), which "
        "has its own separate report and is run on its own -- see this script's "
        "module docstring for why.",
        "",
        "| Evaluation | Status |",
        "|---|---|",
    ]
    for name, status in summary_rows:
        summary_lines.append(f"| {name} | {status} |")
    summary_lines += [
        "",
        f"Mode: {'--fast (routing-only)' if args.fast else 'full (needs the live stack)'}",
        "",
        "Full stdout/stderr of every subprocess run this time: `run_log.txt`, same folder.",
    ]
    (out_dir / "SUMMARY.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"\n[run_all_evaluations] done. Combined results in: {out_dir}")
    for name, status in summary_rows:
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
