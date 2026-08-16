import { useCallback, useEffect, useRef, useState } from "react";
import { branchThread, deleteThread, fetchChats } from "../api";
import type { ChatSummary } from "../api";

interface ChatHistoryProps {
  activeThreadId: string | null;
  onSelect: (threadId: string) => void;
  /** Called after the sidebar has already deleted a thread server-side,
   * so the parent (App.tsx) can reset the main panel if that was the
   * thread currently open there. A no-op from the parent's own
   * perspective when the deleted thread wasn't the active one. */
  onDeleted: (threadId: string) => void;
  /** Bumped by the parent after a send completes / a thread resets, so a
   * new or newly-active conversation shows up (or moves to the top of)
   * the list without the person needing to manually refresh. */
  refreshKey: number;
}

function formatUpdatedAt(iso: string | null): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  const now = new Date();
  const sameDay = d.toDateString() === now.toDateString();
  return sameDay
    ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : d.toLocaleDateString([], { month: "short", day: "numeric" });
}

/**
 * Browsing past conversations -- lists every thread GET /chats returns
 * (most recently active first) and lets the person click back into one.
 * Loading/switching a thread is delegated entirely to the parent via
 * `onSelect` (App.tsx wires this to useBackendChat's own `selectThread`,
 * see runtime.ts) -- this component only fetches and renders the list
 * itself, same "one job per component" split Message.tsx/Composer.tsx
 * already follow.
 *
 * Each row also has its own "options" (⋮) menu with two actions:
 *   - Branch: POST /chat/{thread_id}/branch (api.ts's branchThread) --
 *     copies that conversation's history onto a new, independent
 *     thread_id, then switches to it via the same `onSelect` a normal
 *     row click uses. The original conversation is untouched either way.
 *   - Delete: DELETE /chat/{thread_id} (api.ts's deleteThread) -- removed
 *     from this list immediately on success; if the deleted thread was
 *     the one open in the main panel, `onDeleted` tells the parent to
 *     reset it (see App.tsx's handleThreadDeleted / runtime.ts's
 *     forgetThreadIfActive).
 */
export function ChatHistory({ activeThreadId, onSelect, onDeleted, refreshKey }: ChatHistoryProps) {
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  // thread_id of the row whose "⋮" menu is currently open -- only ever
  // one at a time, so a single id (rather than a Set) is enough.
  const [openMenuId, setOpenMenuId] = useState<string | null>(null);
  // thread_id currently mid-branch/mid-delete, so that row's own buttons
  // can disable themselves rather than letting a second click fire a
  // second request against the same thread while the first is in flight.
  const [busyId, setBusyId] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    fetchChats()
      .then((res) => setChats(res.chats))
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    load();
  }, [load, refreshKey]);

  // Dismiss the open per-chat menu on any click outside it -- a plain
  // dropdown, not a modal, so it should close the way a native context
  // menu would rather than needing an explicit "close" click.
  useEffect(() => {
    if (!openMenuId) return;
    function handleOutsideClick(event: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setOpenMenuId(null);
      }
    }
    document.addEventListener("mousedown", handleOutsideClick);
    return () => document.removeEventListener("mousedown", handleOutsideClick);
  }, [openMenuId]);

  const handleBranch = useCallback(
    async (threadId: string) => {
      setOpenMenuId(null);
      setBusyId(threadId);
      setActionError(null);
      try {
        const res = await branchThread(threadId);
        // Refresh the list so the new branch shows up (it's a brand-new
        // thread the sidebar hasn't seen yet), then switch to it -- same
        // callback a normal row click uses, so forced-tool state, the
        // main panel's message list, etc. all reset exactly the same way.
        load();
        onSelect(res.thread_id);
      } catch (err) {
        setActionError(err instanceof Error ? err.message : String(err));
      } finally {
        setBusyId(null);
      }
    },
    [load, onSelect],
  );

  const handleDelete = useCallback(
    async (threadId: string, title: string) => {
      setOpenMenuId(null);
      const label = title || "this conversation";
      if (!window.confirm(`Delete "${label}"? This can't be undone.`)) return;
      setBusyId(threadId);
      setActionError(null);
      try {
        await deleteThread(threadId);
        setChats((prev) => prev.filter((c) => c.thread_id !== threadId));
        onDeleted(threadId);
      } catch (err) {
        setActionError(err instanceof Error ? err.message : String(err));
      } finally {
        setBusyId(null);
      }
    },
    [onDeleted],
  );

  return (
    <aside className="aui-sidebar">
      <div className="aui-sidebar-header">
        <span>Chats</span>
        <button
          type="button"
          className="aui-sidebar-refresh"
          onClick={load}
          disabled={loading}
          title="Refresh"
        >
          ⟳
        </button>
      </div>

      {(error || actionError) && (
        <div className="aui-sidebar-error">{error ?? actionError}</div>
      )}

      {!error && chats.length === 0 && !loading && (
        <div className="aui-sidebar-empty">No past conversations yet.</div>
      )}

      <ul className="aui-sidebar-list">
        {chats.map((c) => {
          const isBusy = busyId === c.thread_id;
          return (
            <li key={c.thread_id} className="aui-sidebar-row">
              <button
                type="button"
                className={
                  "aui-sidebar-item" + (c.thread_id === activeThreadId ? " aui-sidebar-item-active" : "")
                }
                onClick={() => onSelect(c.thread_id)}
                title={c.title}
                disabled={isBusy}
              >
                <div className="aui-sidebar-item-title">{c.title || "(empty)"}</div>
                <div className="aui-sidebar-item-meta">
                  <span>{formatUpdatedAt(c.updated_at)}</span>
                  <span>{c.message_count} msgs</span>
                </div>
              </button>

              <div className="aui-sidebar-item-options">
                <button
                  type="button"
                  className="aui-sidebar-item-menu-btn"
                  onClick={() => setOpenMenuId((cur) => (cur === c.thread_id ? null : c.thread_id))}
                  disabled={isBusy}
                  title="Chat options"
                  aria-label="Chat options"
                  aria-haspopup="menu"
                  aria-expanded={openMenuId === c.thread_id}
                >
                  {isBusy ? "…" : "⋮"}
                </button>

                {openMenuId === c.thread_id && (
                  <div className="aui-sidebar-item-menu" role="menu" ref={menuRef}>
                    <button type="button" role="menuitem" onClick={() => handleBranch(c.thread_id)}>
                      Branch conversation
                    </button>
                    <button
                      type="button"
                      role="menuitem"
                      className="aui-sidebar-item-menu-danger"
                      onClick={() => handleDelete(c.thread_id, c.title)}
                    >
                      Delete
                    </button>
                  </div>
                )}
              </div>
            </li>
          );
        })}
      </ul>
    </aside>
  );
}
