import { ActionBarPrimitive, ComposerPrimitive, MessagePrimitive } from "@assistant-ui/react";
import { useAuiState } from "@assistant-ui/react";
import { MarkdownText } from "./MarkdownText";

/**
 * The specialist (or "input_guard") that produced an assistant message,
 * read off the `metadata.custom.name` field the runtime hook
 * (src/runtime.ts's toThreadMessageLike) attaches per message. Only
 * meaningful inside a component rendered by ThreadPrimitive.Messages,
 * where useAuiState's `s.message` resolves to the message currently being
 * rendered (each message primitive nests its own scope).
 */
function useAnsweredByName(): string | undefined {
  return useAuiState((s) => {
    const custom = s.message.metadata?.custom;
    const name = custom && typeof custom === "object" ? (custom as Record<string, unknown>).name : undefined;
    return typeof name === "string" ? name : undefined;
  });
}

export function UserMessage() {
  return (
    <MessagePrimitive.Root className="aui-message aui-message-user">
      {/* ComposerPrimitive.If (not MessagePrimitive.If -- this version of
          assistant-ui exposes "editing" as the message's own EditComposer
          state, not a MessagePrimitive.If filter) switches between the
          normal read-only bubble and an inline edit composer. Rendering
          <ComposerPrimitive.Root> HERE, inside this message's own scope,
          binds it to THIS message's edit composer (see
          useActionBarEdit/ActionBarPrimitive.Edit below, which starts
          that same scoped edit session) -- sending it calls
          runtime.ts's onEdit with `sourceId` set to this message's own
          id, never the thread-level composer's onNew. */}
      <ComposerPrimitive.If editing={false}>
        <div className="aui-bubble aui-bubble-user">
          <MessagePrimitive.Content />
        </div>
        <ActionBarPrimitive.Root className="aui-action-bar" hideWhenRunning autohide="not-last">
          <ActionBarPrimitive.Edit className="aui-action-bar-button">Edit</ActionBarPrimitive.Edit>
        </ActionBarPrimitive.Root>
      </ComposerPrimitive.If>
      <ComposerPrimitive.If editing>
        <ComposerPrimitive.Root className="aui-composer aui-edit-composer">
          <ComposerPrimitive.Input className="aui-composer-input" rows={1} autoFocus />
          <div className="aui-composer-row">
            <ComposerPrimitive.Cancel className="aui-action-bar-button">Cancel</ComposerPrimitive.Cancel>
            <ComposerPrimitive.Send className="aui-composer-send">Save &amp; branch</ComposerPrimitive.Send>
          </div>
        </ComposerPrimitive.Root>
      </ComposerPrimitive.If>
    </MessagePrimitive.Root>
  );
}

export function AssistantMessage() {
  const name = useAnsweredByName();
  const isRefused = name === "input_guard";
  const isError = name === "error";

  return (
    <MessagePrimitive.Root className="aui-message aui-message-assistant">
      {name && !isError && <div className="aui-tag">{name}</div>}
      <div
        className={
          "aui-bubble aui-bubble-assistant" +
          (isRefused ? " aui-bubble-refused" : "") +
          (isError ? " aui-bubble-error" : "")
        }
      >
        <MessagePrimitive.Content components={{ Text: MarkdownText }} />
      </div>
      {!isError && (
        <ActionBarPrimitive.Root className="aui-action-bar" hideWhenRunning autohide="not-last">
          <ActionBarPrimitive.Reload className="aui-action-bar-button">Retry</ActionBarPrimitive.Reload>
        </ActionBarPrimitive.Root>
      )}
    </MessagePrimitive.Root>
  );
}
