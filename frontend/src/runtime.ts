import { useCallback, useEffect, useRef, useState } from "react";
import type { AppendMessage, TextMessagePart, ThreadMessageLike } from "@assistant-ui/react";
import { useExternalStoreRuntime } from "@assistant-ui/react";
import { editMessage, fetchHistory, postChat, retryMessage } from "./api";
import { createPersonalRagAttachmentAdapter } from "./attachments";
import type { ChatMessage } from "./types";

const THREAD_ID_STORAGE_KEY = "magrag_thread_id";

// Mirrors agents/api.py's own _MAX_MESSAGE_CHARS (12000) -- kept here as
// a plain constant rather than fetched from the server, since it needs
// to be checked synchronously as the person types/sends, before any
// network round trip. The server enforces the real limit regardless of
// what this constant says (see api.py's own ChatRequest field
// validators) -- this is purely an early, friendlier rejection so a
// person pasting a huge block of text gets an immediate, specific
// message instead of waiting on a request that was always going to be
// rejected.
const MAX_MESSAGE_CHARS = 12000;

// Matches the exact marker format createPersonalRagAttachmentAdapter's
// own send() (attachments.ts) builds for a completed upload -- e.g.
// `<attachment name=IMG_1234.png status="ingested into this
// conversation's personal knowledge base" chunks=1>`. assistant-ui
// merges that marker's own "text" content part into the SAME
// AppendMessage.content array as whatever the person actually typed, so
// a plain `.find(part => part.type === "text")` (the old
// extractUserText below) can't tell the two apart by `type` alone --
// only by recognizing this specific, program-generated shape.
const ATTACHMENT_MARKER_RE = /^<attachment name=.*? status="[^"]*"(?:\s+chunks=\d+)?>$/;

function isAttachmentMarkerText(text: string): boolean {
  return ATTACHMENT_MARKER_RE.test(text.trim());
}

// What the person actually TYPED -- every "text" content part that
// ISN'T one of attachments.ts's own synthetic markers, joined back
// together. Distinct from extractAttachmentMarkers below so the two can
// be recombined deliberately (see onNew) instead of one silently
// clobbering the other depending on assistant-ui's own internal
// content-part ordering, which the previous single-`.find()`
// implementation was exposed to.
function extractUserText(message: AppendMessage): string {
  if (message.role !== "user") return "";
  return message.content
    .filter((part): part is TextMessagePart => part.type === "text" && !isAttachmentMarkerText(part.text))
    .map((part) => part.text)
    .join("\n")
    .trim();
}

// The attachment marker text part(s) alone, in the order assistant-ui
// put them in `message.content` -- empty when nothing was attached to
// this send. Used by onNew to (a) detect a stand-alone attachment send
// (attached with nothing typed) so it can skip sending a nonsensical
// fake "question" to the graph, and (b) still forward the marker(s)
// alongside a REAL typed question, so the model can see what was
// attached -- same behavior attachments.ts's own docstring already
// promises, just no longer at risk of the marker and the real text
// silently overwriting each other.
function extractAttachmentMarkers(message: AppendMessage): string[] {
  if (message.role !== "user") return [];
  return message.content
    .filter((part): part is TextMessagePart => part.type === "text" && isAttachmentMarkerText(part.text))
    .map((part) => part.text);
}

function toThreadMessageLike(message: ChatMessage): ThreadMessageLike {
  return {
    id: message.id,
    role: message.role,
    content: message.content,
    metadata: message.name ? { custom: { name: message.name } } : undefined,
  };
}

// A brand-new, never-yet-used thread id, generated CLIENT-SIDE rather
// than waiting for the server to hand one back after the first /chat
// call. Same shape (a uuid4 string) agents/api.py's own `thread_id =
// req.thread_id or str(uuid4())` would have generated server-side --
// ChatRequest.thread_id's own validator (_THREAD_ID_RE) already accepts
// a caller-supplied id, so sending one from the very first request is
// not a new server-side allowance, just this file choosing to use it.
//
// Why this matters now and didn't before: attaching a file (see
// attachments.ts's createPersonalRagAttachmentAdapter) uploads straight
// into POST /chat/{thread_id}/upload, which needs a real thread_id to
// tag the ingested chunks with -- and a person should be able to attach
// a file to a brand-new conversation BEFORE typing and sending their
// first message, not only after. Pre-generating one on mount (and again
// every time the thread is reset, see clearLocalThread below) means a
// valid thread_id always exists for an attachment to upload against,
// even though the checkpointer itself has no messages for it yet until
// the first /chat call actually lands -- exactly the same "unknown
// thread_id is not an error, it just has no state yet" behavior GET
// /chat/{thread_id}/history already documents.
function generateThreadId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  // Extremely old-browser fallback -- not cryptographically strong, but
  // this only ever needs to be a plausible-looking, collision-unlikely
  // id for a local single-user dev tool, never a security boundary.
  return `${Date.now().toString(16)}-${Math.random().toString(16).slice(2)}`;
}

/**
 * Bridges assistant-ui's ExternalStoreRuntime to agents/api.py.
 *
 * Why ExternalStoreRuntime rather than LocalRuntime: the FastAPI backend
 * (agents/api.py), not the browser, owns conversation history -- it's
 * persisted server-side via LangGraph's checkpointer, keyed by thread_id
 * (see agents/api.py's own module docstring for why that matters: the
 * `invoice` specialist reads back prior `product_search` turns from that
 * same persisted history). ExternalStoreRuntime is built for exactly this
 * shape: "you own the message array, we render whatever you give us" --
 * versus LocalRuntime, which expects to own message state itself and
 * would need a full ThreadHistoryAdapter (branching repository format)
 * bolted on to reach parity with what the backend already does simply.
 *
 * `messages` is plain React state seeded from GET /chat/{thread_id}/history
 * on first load of a stored thread_id, then appended to locally after each
 * POST /chat response -- so a reload re-fetches from the server (source of
 * truth), but a normal send/receive cycle doesn't round-trip an extra GET.
 * Every message in `messages` carries the REAL, checkpointer-persisted id
 * the backend assigned it (ChatResponseBody.human_message_id /
 * TurnMessage.id / HistoryMessageBody.id -- see agents/api.py's own
 * comments on why those were added), not a client-invented one -- that's
 * what lets onEdit/onReload below target a message immediately after it's
 * sent, with no history round trip needed in between.
 *
 * `forcedTool` (nullable specialist name from the tool-selector dropdown,
 * see App.tsx) is read fresh on every send rather than captured once --
 * it's forwarded as ChatRequestBody.tool on the NEXT message only; it does
 * not retroactively affect messages already in `messages`, and switching
 * it back to null (the default "Auto" option) goes straight back to normal
 * supervisor-routed behavior on the following send, exactly mirroring how
 * agents/api.py's own `_new_turn_state` resets `forced_route` every turn
 * server-side rather than letting it persist.
 */
export function useBackendChat(forcedTool: string | null) {
  const [threadId, setThreadId] = useState<string | null>(() => {
    if (typeof window === "undefined") return null;
    const stored = localStorage.getItem(THREAD_ID_STORAGE_KEY);
    if (stored) return stored;
    const fresh = generateThreadId();
    localStorage.setItem(THREAD_ID_STORAGE_KEY, fresh);
    return fresh;
  });
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Incremented once per completed send and once per reset -- exposed so
  // a sidebar listing past conversations (components/ChatHistory.tsx)
  // knows to re-fetch GET /chats after something that could change that
  // list (a new thread created, an existing one getting a new turn, a
  // thread being deleted) without polling or re-deriving "did something
  // change" from `messages`/`threadId` itself.
  const [historyVersion, setHistoryVersion] = useState(0);
  // Guards against re-fetching history for a thread_id we already loaded
  // (or already have in sync locally right after a send) -- without this,
  // the effect below would refetch on every render that happens to see the
  // same threadId, not just on first mount / thread switch. Also used to
  // mark a freshly-generated (never-sent-to) thread_id as "loaded" so the
  // effect below doesn't bother GETting history for it that's guaranteed
  // to come back empty.
  const loadedThreadRef = useRef<string | null>(
    threadId /* generateThreadId() above always assigns one synchronously in a browser */
      ? threadId
      : null,
  );

  // `getThreadId` (not a captured `threadId` value) because
  // createPersonalRagAttachmentAdapter is built once and its `send`
  // callback needs to read whichever thread_id is CURRENT at upload
  // time, not whichever one was current when the adapter object itself
  // was constructed.
  const threadIdRef = useRef(threadId);
  threadIdRef.current = threadId;
  const [attachmentAdapter] = useState(() =>
    createPersonalRagAttachmentAdapter(() => threadIdRef.current ?? generateThreadId()),
  );

  useEffect(() => {
    if (!threadId || loadedThreadRef.current === threadId) return;
    loadedThreadRef.current = threadId;
    fetchHistory(threadId)
      .then((history) => {
        setMessages(
          history.messages.map((m) => ({
            id: m.id,
            role: m.role === "human" ? "user" : "assistant",
            content: m.content,
            name: m.name ?? undefined,
          })),
        );
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, [threadId]);

  // Pulls this thread's own persisted history straight from the server
  // and replaces `messages` with it wholesale -- the authoritative
  // source of truth, same one the mount-time effect above uses for a
  // freshly-opened thread. Used by onNew below whenever an attachment
  // was part of the send: uploading an image writes its own
  // Human/AI message pair straight into the checkpointer via
  // agents/api.py's _post_captioned_images_to_chat, OUTSIDE of anything
  // the normal "append the new user text + the new assistant reply"
  // local update further down could ever know about on its own (that
  // update only ever knows about the ONE turn it just ran). Without
  // this, that pair sits correctly in the database but stays invisible
  // in the UI until something else happens to trigger a fetch (a page
  // reload, switching threads and back) -- exactly the "doesn't show up
  // until I refresh" symptom this fixes.
  const syncMessagesFromServer = useCallback(async (id: string) => {
    const history = await fetchHistory(id);
    setMessages(
      history.messages.map((m) => ({
        id: m.id,
        role: m.role === "human" ? "user" : "assistant",
        content: m.content,
        name: m.name ?? undefined,
      })),
    );
    loadedThreadRef.current = id;
    setHistoryVersion((v) => v + 1);
  }, []);

  const onNew = useCallback(
    async (message: AppendMessage) => {
      const typedText = extractUserText(message).trim();
      const attachmentMarkers = extractAttachmentMarkers(message);
      const hasAttachments = attachmentMarkers.length > 0;

      // A STAND-ALONE attachment send -- an image (or PDF) attached with
      // nothing actually typed alongside it (the whole point of this
      // check: "send the model a stand-alone picture and it would
      // caption it, visible in the chat"). By the time onNew fires here,
      // createPersonalRagAttachmentAdapter's own send() (attachments.ts)
      // has ALREADY run POST /chat/{thread_id}/upload, which -- for an
      // image -- has ALREADY captioned it and posted both the caption
      // and the image itself straight into this thread's own persisted
      // history (see agents/api.py's _post_captioned_images_to_chat).
      // There is no real question here to route through the graph at
      // all. Sending the synthetic `<attachment ...>` marker to POST
      // /chat as if it WERE one would make the supervisor try to
      // "answer" a machine-readable status string, producing an
      // unrelated, confusing reply stacked right after the real caption
      // -- so this skips that round trip entirely and instead just syncs
      // the caption/image message(s) the upload already wrote
      // server-side into this thread's local view.
      if (hasAttachments && !typedText) {
        setError(null);
        if (threadId) {
          setIsRunning(true);
          try {
            await syncMessagesFromServer(threadId);
          } catch (err) {
            setError(err instanceof Error ? err.message : String(err));
          } finally {
            setIsRunning(false);
          }
        }
        return;
      }

      const text = typedText;

      // Reject unacceptable input client-side, immediately, with a
      // message the person can act on -- rather than either silently
      // dropping it (the old `if (!text) return`, which gave no
      // feedback at all for e.g. a whitespace-only send) or sending it
      // anyway and waiting on a round trip that agents/api.py's own
      // field validators (ChatRequest.message) were always going to
      // reject. The server remains the real, authoritative check either
      // way -- this is purely a faster, friendlier first line of
      // defense, not a replacement for it.
      if (!text) {
        setError("Message can't be empty or contain only whitespace.");
        return;
      }
      if (text.length > MAX_MESSAGE_CHARS) {
        setError(
          `Message is too long (${text.length.toLocaleString()} characters). ` +
            `Please shorten it to ${MAX_MESSAGE_CHARS.toLocaleString()} characters or fewer.`,
        );
        return;
      }

      setError(null);
      // Optimistic, locally-invented id -- upgraded to the real,
      // checkpointer-persisted one (ChatResponseBody.human_message_id)
      // the moment the response comes back, below. Never left as-is: a
      // retry/edit sent before that upgrade landed would have nothing
      // real to target server-side.
      const localId = `pending-${Date.now()}`;
      // Shown locally as just what the person typed -- never the raw
      // `<attachment ...>` marker, which is an internal detail for the
      // MODEL to see, not a bubble a person should have to read.
      const userMsg: ChatMessage = { id: localId, role: "user", content: text };
      setMessages((prev) => [...prev, userMsg]);
      setIsRunning(true);

      // A question sent ALONGSIDE an attachment still carries the
      // marker(s) too, appended after what was actually typed -- same
      // "the model can see what was attached" behavior attachments.ts's
      // own docstring already documents (e.g. so a follow-up like "what
      // is this?" has something to resolve against), just built from
      // the explicitly-separated parts above instead of whichever one
      // assistant-ui happened to order first in message.content.
      const messageForServer = [text, ...attachmentMarkers].join("\n");

      try {
        const res = await postChat({
          message: messageForServer,
          thread_id: threadId ?? undefined,
          tool: forcedTool ?? undefined,
        });
        if (threadId !== res.thread_id) {
          setThreadId(res.thread_id);
          localStorage.setItem(THREAD_ID_STORAGE_KEY, res.thread_id);
          // The thread we just created/continued is already in sync with
          // what `messages` holds locally (this turn included) -- mark it
          // loaded so the effect above doesn't immediately re-fetch and
          // briefly flash an empty/stale list.
          loadedThreadRef.current = res.thread_id;
        }
        // The upload that ran before this turn already wrote its own
        // caption/image confirmation pair straight into the checkpointer
        // (see agents/api.py's _post_captioned_images_to_chat) when an
        // attachment was involved -- but resyncing the WHOLE thread here
        // to pick that pair back up (the old behavior) is exactly what
        // made a genuine question sent ALONGSIDE an attachment look
        // duplicated: the confirmation pair, then this turn's own
        // question, then this turn's own answer, reading like two
        // separate exchanges instead of one (a confirmed live report:
        // attaching an image and typing "explain this" in the same send
        // showed the image once with an auto-caption, then, separately
        // below it, the real question and its real answer). The
        // confirmation pair is still safely persisted server-side either
        // way -- it's what makes the image searchable, and what a fresh
        // GET /chat/{thread_id}/history reload will still show -- this
        // just stops THIS live send from re-displaying it a second time
        // on top of the turn the person can already see they just sent
        // (which already carries its own attachment preview via
        // assistant-ui's own AttachmentPrimitive rendering), whether or
        // not an attachment was involved.
        //
        // A server-side attempt at also retracting that pair from
        // PERSISTED state (so a reload wouldn't show it either) was
        // tried and reverted: it depended on the FOLLOW-UP turn routing
        // correctly to a specialist that actually references the same
        // upload, and when routing misfired (a confirmed live report --
        // a small local model sending an obviously image-related
        // question to `retrieval_qa` instead of `personal_docs`), the
        // retraction removed the one thing that reliably showed the
        // image, leaving nothing useful at all. See agents/supervisor.py's
        // own deterministic routing check for the fix that actually
        // targets THAT failure instead.
        const lastTurnMessage = res.turn_messages[res.turn_messages.length - 1];
        const assistantMsg: ChatMessage = {
          id: lastTurnMessage?.id ?? `local-${Date.now()}-a`,
          role: "assistant",
          content: res.answer,
          name: res.answered_by ?? undefined,
        };
        setMessages((prev) => {
          const upgraded = prev.map((m) =>
            m.id === localId ? { ...m, id: res.human_message_id } : m,
          );
          return [...upgraded, assistantMsg];
        });
        setHistoryVersion((v) => v + 1);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
        if (hasAttachments && threadId) {
          // Even though THIS /chat call failed, the upload that ran
          // before it already succeeded and wrote its own caption/image
          // message(s) server-side -- resync so those don't stay
          // invisible just because the follow-up question's answer
          // didn't come back. Best-effort: if this second fetch also
          // fails, fall through to the same inline error bubble the
          // no-attachment path already shows below.
          try {
            await syncMessagesFromServer(threadId);
          } catch {
            // ignore -- the setError above already surfaced the
            // original failure; nothing further to do here.
          }
        }
        setMessages((prev) => [
          ...prev,
          {
            id: `local-${Date.now()}-err`,
            role: "assistant",
            content:
              "Couldn't reach the agent API. Check that `python -m agents.api` is running and reachable.",
            name: "error",
          },
        ]);
      } finally {
        setIsRunning(false);
      }
    },
    [threadId, forcedTool, syncMessagesFromServer],
  );

  // "Retry"/"regenerate" -- resend a prompt that's already in this
  // thread and get a NEW answer back IN PLACE OF the old one (never
  // both stacked in the transcript). `parentId` is assistant-ui's own
  // id for "the message right before the one being regenerated" -- for
  // a normal chat (every assistant message's parent is the human
  // message it answered) this is exactly the HumanMessage id
  // agents/api.py's POST /chat/{thread_id}/retry wants as `message_id`.
  // A null parentId (no message before the one being reloaded) isn't a
  // shape this app's own threads ever produce -- every thread starts
  // with a HumanMessage -- so it's treated as an error rather than
  // silently guessing which turn to retry.
  const onReload = useCallback(
    async (parentId: string | null) => {
      if (!threadId) {
        setError("No active conversation to retry.");
        return;
      }
      if (!parentId) {
        setError("Can't retry this message.");
        return;
      }
      setError(null);
      setIsRunning(true);
      try {
        const res = await retryMessage(threadId, parentId, forcedTool ?? undefined);
        const lastTurnMessage = res.turn_messages[res.turn_messages.length - 1];
        const assistantMsg: ChatMessage = {
          id: lastTurnMessage?.id ?? `local-${Date.now()}-a`,
          role: "assistant",
          content: res.answer,
          name: res.answered_by ?? undefined,
        };
        setMessages((prev) => {
          const idx = prev.findIndex((m) => m.id === res.human_message_id);
          const prefix = idx >= 0 ? prev.slice(0, idx + 1) : prev;
          return [...prefix, assistantMsg];
        });
        setHistoryVersion((v) => v + 1);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setIsRunning(false);
      }
    },
    [threadId, forcedTool],
  );

  // "Edit" -- change an earlier prompt's text; the runtime core calls
  // this with `message.sourceId` set to the ORIGINAL message's id (see
  // AppendMessage's own "The ID of the message that was edited" field)
  // and the edited text as this message's own content. Branches a
  // brand-new thread_id server-side (POST /chat/{thread_id}/edit --
  // see that endpoint's own docstring for why "edit" means "branch,"
  // same as this project's existing sidebar Branch action) and switches
  // the UI over to it, the same way onNew switches over to a
  // server-assigned thread_id on a brand-new conversation's first send.
  const onEdit = useCallback(
    async (message: AppendMessage) => {
      const rawText = extractUserText(message);
      const text = rawText.trim();

      if (!text) {
        setError("Message can't be empty or contain only whitespace.");
        return;
      }
      if (text.length > MAX_MESSAGE_CHARS) {
        setError(
          `Message is too long (${text.length.toLocaleString()} characters). ` +
            `Please shorten it to ${MAX_MESSAGE_CHARS.toLocaleString()} characters or fewer.`,
        );
        return;
      }
      if (!threadId || !message.sourceId) {
        setError("Can't edit this message right now.");
        return;
      }

      setError(null);
      setIsRunning(true);
      try {
        const res = await editMessage(threadId, message.sourceId, text, forcedTool ?? undefined);
        const editedSourceId = message.sourceId;
        const lastTurnMessage = res.chat.turn_messages[res.chat.turn_messages.length - 1];
        const editedUserMsg: ChatMessage = {
          id: res.chat.human_message_id,
          role: "user",
          content: text,
        };
        const assistantMsg: ChatMessage = {
          id: lastTurnMessage?.id ?? `local-${Date.now()}-a`,
          role: "assistant",
          content: res.chat.answer,
          name: res.chat.answered_by ?? undefined,
        };
        setMessages((prev) => {
          const idx = prev.findIndex((m) => m.id === editedSourceId);
          const prefix = idx >= 0 ? prev.slice(0, idx) : prev;
          return [...prefix, editedUserMsg, assistantMsg];
        });
        setThreadId(res.thread_id);
        localStorage.setItem(THREAD_ID_STORAGE_KEY, res.thread_id);
        loadedThreadRef.current = res.thread_id;
        setHistoryVersion((v) => v + 1);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setIsRunning(false);
      }
    },
    [threadId, forcedTool],
  );

  // Local-only teardown of "the conversation currently open in the main
  // panel" -- shared by resetThread ("New conversation") and
  // forgetThreadIfActive (called after the sidebar deletes a thread
  // server-side). Never talks to the server itself; the two callers
  // below decide separately whether a server-side call is warranted.
  // Immediately generates and stores a FRESH thread_id (rather than
  // leaving it null until the next message is sent) -- see
  // generateThreadId's own comment for why an attachment needs a real
  // thread_id to upload against even before the first message goes out.
  const clearLocalThread = useCallback(() => {
    const fresh = generateThreadId();
    loadedThreadRef.current = fresh;
    setThreadId(fresh);
    setMessages([]);
    setError(null);
    localStorage.setItem(THREAD_ID_STORAGE_KEY, fresh);
  }, []);

  // "New conversation" -- starts a fresh thread WITHOUT deleting the one
  // just left. Deliberately does NOT call deleteThread: the old thread
  // must stay reachable from the sidebar (components/ChatHistory.tsx)
  // after clicking this, exactly like every other chat UI's "new chat"
  // button.
  const resetThread = useCallback(() => {
    clearLocalThread();
    setHistoryVersion((v) => v + 1);
  }, [clearLocalThread]);

  // Called by the sidebar (components/ChatHistory.tsx) AFTER it has
  // already deleted a thread server-side via DELETE /chat/{thread_id} --
  // this only handles the case where the thread just deleted was the one
  // open in the main panel, so the person isn't left staring at now-
  // orphaned messages for a conversation that no longer exists anywhere.
  // Deleting a thread that ISN'T the active one needs no action here at
  // all (the main panel is showing something else already); the sidebar
  // still calls this unconditionally, and the `!==` check below is what
  // makes that a no-op in that case rather than needing two call sites.
  const forgetThreadIfActive = useCallback(
    (deletedThreadId: string) => {
      if (deletedThreadId === threadId) {
        clearLocalThread();
      }
      setHistoryVersion((v) => v + 1);
    },
    [threadId, clearLocalThread],
  );

  // Switches to an EXISTING thread_id -- e.g. one the person picked from
  // the chat-history sidebar (see components/ChatHistory.tsx) rather than
  // one this hook just created via onNew. Never deletes anything --
  // switching away from a thread must not delete it, and (as of the
  // resetThread change above) neither does starting a new one; deletion
  // only ever happens via the sidebar's explicit per-chat "Delete"
  // option, which calls the api.ts deleteThread() function directly.
  // Clearing `loadedThreadRef` before setting `threadId` is what makes the
  // existing history-fetch effect above treat this as a fresh thread to
  // load rather than a no-op (it only refetches when
  // `loadedThreadRef.current !== threadId`), same guard `onNew` already
  // relies on after starting a brand-new thread.
  const selectThread = useCallback((nextThreadId: string) => {
    if (nextThreadId === threadId) return;
    loadedThreadRef.current = null;
    setMessages([]);
    setError(null);
    setThreadId(nextThreadId);
    localStorage.setItem(THREAD_ID_STORAGE_KEY, nextThreadId);
  }, [threadId]);

  const runtime = useExternalStoreRuntime<ChatMessage>({
    messages,
    isRunning,
    onNew,
    onEdit,
    onReload,
    convertMessage: toThreadMessageLike,
    adapters: { attachments: attachmentAdapter },
  });

  return {
    runtime,
    threadId,
    isRunning,
    error,
    resetThread,
    selectThread,
    forgetThreadIfActive,
    historyVersion,
  };
}
