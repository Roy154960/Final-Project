import type { AttachmentAdapter, PendingAttachment, CompleteAttachment } from "@assistant-ui/react";
import { uploadDocument } from "./api";

// Mirrors local_rag/personal_rag.py's SUPPORTED_UPLOAD_EXTS -- images,
// PDFs, and plain text files. Kept here as a plain constant rather than
// fetched from the server for the same reason runtime.ts's own
// MAX_MESSAGE_CHARS is: the composer's file picker needs this
// synchronously, before any network round trip. The server enforces the
// real check regardless (see personal_rag.ingest_upload's own
// ValueError) -- this is purely a friendlier first filter on which
// files even show up as choosable.
const ACCEPT = "application/pdf,text/plain,image/png,image/jpeg,image/webp,image/bmp";

/**
 * Attaching a file in the composer uploads it straight into THIS
 * conversation's own personal RAG (agents/api.py's POST
 * /chat/{thread_id}/upload -> local_rag/personal_rag.py's "temp" Chroma
 * collection, filtered by thread_id) -- not into the message itself. The
 * file's raw bytes/text never enter the prompt; only a short marker line
 * is added to the message so the person (and the model) can see what was
 * attached, while the actual content becomes searchable via the
 * personal_docs specialist the next time the supervisor routes to it.
 *
 * `getThreadId` is a function, not a captured value, because the
 * attachment can be added and sent before a thread_id would otherwise
 * exist (runtime.ts pre-generates one on mount specifically so uploads
 * never have to wait for the first /chat round trip -- see that file's
 * own comment on THREAD_ID_STORAGE_KEY).
 */
export function createPersonalRagAttachmentAdapter(getThreadId: () => string): AttachmentAdapter {
  return {
    accept: ACCEPT,

    async add(state: { file: File }): Promise<PendingAttachment> {
      return {
        id: `${Date.now()}-${state.file.name}`,
        type: state.file.type.startsWith("image/") ? "image" : "document",
        name: state.file.name,
        contentType: state.file.type,
        file: state.file,
        status: { type: "requires-action", reason: "composer-send" },
      };
    },

    async send(attachment: PendingAttachment): Promise<CompleteAttachment> {
      const threadId = getThreadId();
      const stats = await uploadDocument(threadId, attachment.file);
      const summary =
        stats.n_chunks > 0
          ? `<attachment name=${stats.filename} status="ingested into this conversation's personal knowledge base" chunks=${stats.n_chunks}>`
          : `<attachment name=${stats.filename} status="uploaded, but nothing extractable was found in it">`;
      return {
        ...attachment,
        status: { type: "complete" },
        content: [{ type: "text", text: summary }],
      };
    },

    async remove() {
      // No server-side undo for a single file -- the chunks already live
      // in the shared "temp" collection under this thread_id and get
      // cleaned up as a whole when the thread itself is deleted (see
      // agents/api.py's DELETE /chat/{thread_id} ->
      // personal_rag.delete_thread_data). Removing the attachment here
      // only ever un-does it from the COMPOSER (before it's been sent);
      // there is nothing to remove server-side yet at that point.
    },
  };
}
