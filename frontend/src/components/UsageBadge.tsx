import { useEffect, useState } from "react";
import { fetchUsage } from "../api";
import type { GroqModelUsage } from "../api";

// How often to re-poll GET /v1/usage -- a chat turn every few seconds
// at most, so anything faster than this just re-fetches the same
// numbers; anything much slower would make the badge visibly stale
// right after sending a message. Not tied to message-send events
// directly (no shared state plumbing needed) -- a plain interval keeps
// this component fully self-contained.
const POLL_INTERVAL_MS = 15_000;

// Label + which remaining/limit pair to read, per model -- kept short
// and deliberately not the raw model id, so the badge stays readable at
// header width. Mirrors the three Groq models agents/llm_provider.py
// and local_rag/config.py actually use (GROQ_LARGE_MODEL,
// GROQ_SMALL_MODEL, GROQ_VISION_MODEL) -- if those ever change, update
// the ids here to match.
const TRACKED_MODELS: { id: string; label: string }[] = [
  { id: "llama-3.3-70b-versatile", label: "large" },
  { id: "llama-3.1-8b-instant", label: "small" },
  { id: "qwen/qwen3.6-27b", label: "vision" },
];

function requestsRemainingFraction(usage: GroqModelUsage | undefined): number | null {
  if (!usage || usage.rpd_remaining === undefined || usage.rpd_limit === undefined) return null;
  const remaining = Number(usage.rpd_remaining);
  const limit = Number(usage.rpd_limit);
  if (!limit || Number.isNaN(remaining) || Number.isNaN(limit)) return null;
  return remaining / limit;
}

/**
 * Small header badge showing how much of Groq's free-tier daily request
 * budget is left for each model this session actually uses -- backs
 * agents/api.py's GET /v1/usage (see that endpoint's own docstring for
 * why this is the ONLY usage number surfaced to the UI; cost and
 * per-node traces stay dev-only, on disk, never over HTTP).
 *
 * A model with no data yet (never called this run) renders as "—", not
 * an error -- GET /v1/usage returns an empty entry for it, which is the
 * normal, expected state right after the server starts.
 */
export function UsageBadge() {
  const [modelsUsage, setModelsUsage] = useState<Record<string, GroqModelUsage>>({});
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;

    const poll = () => {
      fetchUsage()
        .then((body) => {
          if (!cancelled) {
            setModelsUsage(body.models ?? {});
            setFailed(false);
          }
        })
        .catch(() => {
          // A single failed poll (server briefly restarting, etc.)
          // shouldn't flip the badge to an error state -- only stop
          // rendering numbers once it's clear the endpoint genuinely
          // isn't there (see the `failed` check below, only used to
          // hide the badge entirely, never to show a scary message).
          if (!cancelled) setFailed(true);
        });
    };

    poll();
    const interval = setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  if (failed && Object.keys(modelsUsage).length === 0) {
    // Older server build without GET /v1/usage, or it's briefly
    // unreachable -- fail quiet rather than showing a broken-looking
    // badge in the header of an otherwise-working chat.
    return null;
  }

  return (
    <div className="aui-usage-badge" title="Groq free-tier requests remaining today, per model">
      <span className="aui-usage-label">groq</span>
      {TRACKED_MODELS.map(({ id, label }) => {
        const usage = modelsUsage[id];
        const fraction = requestsRemainingFraction(usage);
        const low = fraction !== null && fraction < 0.1;
        const fallenBack = usage?.backend_status === "fallback_to_local";
        return (
          <span
            key={id}
            className={`aui-usage-pill${low ? " aui-usage-pill--low" : ""}${
              fallenBack ? " aui-usage-pill--fallback" : ""
            }`}
          >
            {label}: {usage ? `${usage.rpd_remaining}/${usage.rpd_limit}` : "—"}
            {fallenBack ? " (local)" : ""}
          </span>
        );
      })}
    </div>
  );
}
