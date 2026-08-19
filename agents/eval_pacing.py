"""
Shared call-pacing helper for the eval_*.py scripts (eval_routing.py,
eval_phase5.py, eval_language.py) ONLY -- not imported anywhere else in
agents/, and deliberately not touching agents/llm_provider.py's or
local_rag/groq_client.py's own retry logic.

WHY THIS LIVES HERE, NOT IN THE SHARED GROQ CLIENT:

groq_client.py already retries once on a 429, sleeping for whatever
Groq's own `retry-after` header says (capped at a fixed ceiling -- see
that module's own module docstring), then falls back to local Ollama
for that turn if the retry ALSO 429s. That's the right tradeoff for
interactive chat: a real conversation turn shouldn't feel stuck for a
long stretch just because Groq blipped once.

An eval run is a completely different traffic shape. eval_routing.py
fires 94+ questions back-to-back; eval_language.py's fidelity part runs
several multi-turn conversations the same way; eval_phase5.py runs the
full compiled graph (contextualize + supervisor + specialist, several
LLM calls each) per query, also back-to-back. None of that has the
natural spacing a human typing/reading gives ordinary chat traffic --
it's exactly the shape that burns through Groq's requests-per-minute
window fast enough to make the existing single retry insufficient, not
because that retry logic is wrong, but because sustained throughput
over a long run is a different problem than one transient blip.

Lengthening the shared retry to fix THAT would make every ordinary chat
turn feel sluggish on every Groq blip, for a problem ordinary chat
traffic doesn't actually have. So the fix is pacing, and it lives only
in the three scripts that actually produce this traffic shape.
"""

from __future__ import annotations

import asyncio
import os
import sys

from local_rag.usage_tracker import get_usage_snapshot

# Plain, fixed pacing -- not adaptive to remaining quota (the snapshot
# printed below is for visibility, not for deciding how long to sleep;
# see this module's own top docstring for why simple, predictable
# pacing was chosen over something smarter here).
#
# Overridable per run without editing code:
#   EVAL_GROQ_PACE_SECONDS=5 python run_all_evaluations.py   # your Groq
#       account's free-tier window is stricter than 2s comfortably covers
#   EVAL_GROQ_PACE_SECONDS=0 python run_all_evaluations.py   # disable
#       entirely, e.g. testing purely against local Ollama on purpose
DEFAULT_PACE_SECONDS = 5.0


async def pace(label: str = "") -> None:
    """
    Call once per eval iteration, AFTER that iteration's LLM call(s)
    complete and its result is already recorded -- sleeping BEFORE
    would just delay a call that was going to happen anyway, not space
    it out from the next one.

    Sleeps DEFAULT_PACE_SECONDS (or $EVAL_GROQ_PACE_SECONDS if set), and
    prints the current remaining Groq quota alongside the pause, so a
    long eval run's own stderr shows whether pacing is actually needed
    or just cautious. Reads local_rag/usage_tracker.py's own
    get_usage_snapshot() -- the SAME persisted, real rate-limit-header
    state groq_client.py already writes on every Groq response
    (record_groq_rate_limit_headers) and agents/api.py's GET /v1/usage
    already reads for the chat UI's usage badge. Not a new tracking
    mechanism, just reading the one that already exists.
    """
    seconds = float(os.environ.get("EVAL_GROQ_PACE_SECONDS", DEFAULT_PACE_SECONDS))
    if seconds <= 0:
        return

    snapshot = get_usage_snapshot().get("models", {})
    quota_bits = [
        f"{model}: tpm_remaining={info.get('tpm_remaining', '?')} "
        f"rpd_remaining={info.get('rpd_remaining', '?')}"
        for model, info in snapshot.items()
    ]
    quota_line = " | ".join(quota_bits) if quota_bits else "no Groq calls recorded yet"
    tag = f" ({label})" if label else ""
    print(f"[eval_pacing] pausing {seconds}s{tag} -- {quota_line}", file=sys.stderr)

    await asyncio.sleep(seconds)
