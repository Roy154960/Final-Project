import { useCallback, useState } from "react";
import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { useBackendChat } from "./runtime";
import { Thread } from "./components/Thread";
import { ChatHistory } from "./components/ChatHistory";
import { UsageBadge } from "./components/UsageBadge";

export default function App() {
  const [forcedTool, setForcedTool] = useState<string | null>(null);
  const { runtime, threadId, error, resetThread, selectThread, forgetThreadIfActive, historyVersion } =
    useBackendChat(forcedTool);

  // Switching threads (from the sidebar, or after a branch is created)
  // drops any forced-tool selection left over from the PREVIOUS thread --
  // a tool forced for one conversation shouldn't silently keep overriding
  // the supervisor after jumping into an unrelated one.
  const handleSelectThread = useCallback(
    (nextThreadId: string) => {
      setForcedTool(null);
      selectThread(nextThreadId);
    },
    [selectThread],
  );

  // Deleting the currently-open thread from the sidebar leaves the main
  // panel showing messages for a conversation that no longer exists --
  // drop the forced-tool selection the same way switching threads does,
  // and let useBackendChat's own forgetThreadIfActive decide whether a
  // local reset is actually needed (it's a no-op if the deleted thread
  // wasn't the active one).
  const handleThreadDeleted = useCallback(
    (deletedThreadId: string) => {
      setForcedTool(null);
      forgetThreadIfActive(deletedThreadId);
    },
    [forgetThreadIfActive],
  );

  const handleReset = useCallback(() => {
    setForcedTool(null);
    resetThread();
  }, [resetThread]);

  return (
    <div className="aui-app-shell">
      <ChatHistory
        activeThreadId={threadId}
        onSelect={handleSelectThread}
        onDeleted={handleThreadDeleted}
        refreshKey={historyVersion}
      />
      <div className="aui-app">
        <header className="aui-header">
          <div>
            <h1>Multi-Agent RAG Chat</h1>
            <p className="aui-header-tagline">Painting treatises, technique &amp; art-supply lookup</p>
            <div className="aui-thread-id">thread: {threadId ? threadId.slice(0, 8) : "(new)"}</div>
          </div>
          <UsageBadge />
          <button type="button" onClick={handleReset}>
            New conversation
          </button>
        </header>

        {error && <div className="aui-error-banner">{error}</div>}

        <AssistantRuntimeProvider runtime={runtime}>
          <Thread forcedTool={forcedTool} onForcedToolChange={setForcedTool} />
        </AssistantRuntimeProvider>
      </div>
    </div>
  );
}
