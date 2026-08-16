import { AttachmentPrimitive, ComposerPrimitive } from "@assistant-ui/react";

// Mirrors agents/api.py's ChatRequest.message max_length (12000) and
// runtime.ts's own MAX_MESSAGE_CHARS -- a native textarea maxLength stops
// a person from typing/pasting past the limit at all, on top of (not
// instead of) the friendlier, specific rejection message runtime.ts's
// onNew shows if this is ever bypassed (e.g. programmatic paste events
// some browsers don't clamp the same way).
const MAX_MESSAGE_CHARS = 12000;

// One pending/attached file, shown as a small pill above the input while
// it uploads (attachments.ts's createPersonalRagAttachmentAdapter runs
// its `send()` -- the actual POST /chat/{thread_id}/upload -- the moment
// a file is picked, not when the whole message is sent, so this can show
// up mid-upload) and after (once it's become a normal attachment on the
// sent message). Bound to the current attachment's own context by
// ComposerPrimitive.Attachments below -- see that primitive's own
// `components.Attachment` slot.
function ComposerAttachment() {
  return (
    <AttachmentPrimitive.Root className="aui-attachment-pill">
      <AttachmentPrimitive.Name />
      <AttachmentPrimitive.Remove className="aui-attachment-remove">×</AttachmentPrimitive.Remove>
    </AttachmentPrimitive.Root>
  );
}

export function Composer() {
  return (
    <ComposerPrimitive.Root className="aui-composer">
      <ComposerPrimitive.Attachments components={{ Attachment: ComposerAttachment }} />
      <div className="aui-composer-row">
        <ComposerPrimitive.AddAttachment className="aui-composer-attach" title="Attach an image, PDF, or text file">
          +
        </ComposerPrimitive.AddAttachment>
        <ComposerPrimitive.Input
          className="aui-composer-input"
          placeholder="Ask about the corpus, or attach an image/PDF/text file to ask about it directly..."
          rows={1}
          maxLength={MAX_MESSAGE_CHARS}
        />
        <ComposerPrimitive.Send className="aui-composer-send">Send</ComposerPrimitive.Send>
      </div>
    </ComposerPrimitive.Root>
  );
}
