import { ThreadPrimitive } from "@assistant-ui/react";
import { AssistantMessage, UserMessage } from "./Message";
import { Composer } from "./Composer";
import { ToolSelector } from "./ToolSelector";

interface ThreadProps {
  forcedTool: string | null;
  onForcedToolChange: (tool: string | null) => void;
}

export function Thread({ forcedTool, onForcedToolChange }: ThreadProps) {
  return (
    <ThreadPrimitive.Root className="aui-thread">
      <ThreadPrimitive.Viewport className="aui-viewport">
        <ThreadPrimitive.Empty>
          <div className="aui-empty">
            <svg
              className="aui-empty-icon"
              width="30"
              height="30"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.4"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <path d="M12 19c-3 0-5-1.5-5-4.5 0-2 1.3-3.2 3-3.2 1.2 0 2 .7 2 1.8 0 .9-.6 1.4-1.3 1.4-.5 0-.9-.3-.9-.8" />
              <path d="M12 19c3.5 0 8-2.7 8-8.5C20 5.9 16.4 3 12 3S4 5.9 4 8.8" />
              <circle cx="8.3" cy="9.2" r="0.9" fill="currentColor" stroke="none" />
              <circle cx="12" cy="7" r="0.9" fill="currentColor" stroke="none" />
              <circle cx="15.7" cy="9.2" r="0.9" fill="currentColor" stroke="none" />
            </svg>
            <div>Ask about the corpus to get started.</div>
          </div>
        </ThreadPrimitive.Empty>
        <ThreadPrimitive.Messages
          components={{
            UserMessage,
            AssistantMessage,
          }}
        />
        <ThreadPrimitive.ScrollToBottom className="aui-scroll-to-bottom">
          ↓
        </ThreadPrimitive.ScrollToBottom>
      </ThreadPrimitive.Viewport>
      <div className="aui-composer-bar">
        <ToolSelector value={forcedTool} onChange={onForcedToolChange} />
        <Composer />
      </div>
    </ThreadPrimitive.Root>
  );
}
