export interface ChatRequestBody {
  message: string;
  thread_id?: string;
  tool?: string;
}

export interface TurnMessage {
  id: string;
  name: string | null;
  content: string;
}

export interface ChatResponseBody {
  thread_id: string;
  answer: string;
  answered_by: string | null;
  blocked: boolean;
  iteration_count: number;
  human_message_id: string;
  turn_messages: TurnMessage[];
}

export interface HistoryMessageBody {
  id: string;
  role: "human" | "ai";
  name: string | null;
  content: string;
}

export interface HistoryResponseBody {
  thread_id: string;
  messages: HistoryMessageBody[];
}

export interface ChatSummary {
  thread_id: string;
  title: string;
  updated_at: string | null;
  message_count: number;
}

export interface ChatListResponseBody {
  chats: ChatSummary[];
}

export interface BranchResponseBody {
  thread_id: string;
  branched_from: string;
  message_count: number;
}

export interface EditResponseBody {
  thread_id: string;
  branched_from: string;
  edited_message_id: string;
  chat: ChatResponseBody;
}

export interface UploadResponseBody {
  thread_id: string;
  filename: string;
  n_chunks: number;
  modality: "pdf" | "image";
}

export interface ToolInfo {
  name: string;
  description: string;
}

export interface ToolListResponseBody {
  tools: ToolInfo[];
}

export interface GroqModelUsage {
  rpd_limit?: string;
  rpd_remaining?: string;
  rpd_reset?: string;
  tpm_limit?: string;
  tpm_remaining?: string;
  tpm_reset?: string;
  backend_status?: "ok" | "fallback_to_local";
  updated_at?: number;
  last_error?: string;
  last_error_at?: number;
}

export interface UsageResponseBody {
  models: Record<string, GroqModelUsage>;
  free_tier: boolean;
  docs: string;
}

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8001";

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export async function postChat(body: ChatRequestBody): Promise<ChatResponseBody> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return handle<ChatResponseBody>(res);
}

/**
 * Regenerate an answer -- "resend this prompt and get a new answer back
 * instead of the old one." `messageId` omitted retries the most recent
 * turn; pass it (a TurnMessage.id / HistoryMessageBody.id /
 * ChatResponseBody.human_message_id) to retry an earlier one instead.
 * The thread_id never changes -- the old answer is replaced in place,
 * server-side (see agents/api.py's POST /chat/{thread_id}/retry).
 */
export async function retryMessage(
  threadId: string,
  messageId?: string,
  tool?: string,
): Promise<ChatResponseBody> {
  const res = await fetch(`${API_BASE}/chat/${threadId}/retry`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message_id: messageId ?? null, tool: tool ?? null }),
  });
  return handle<ChatResponseBody>(res);
}

/**
 * Edit a past prompt -- branches a brand-new thread_id from everything
 * BEFORE the edited message, plus the edited text, and runs one turn on
 * it immediately (see agents/api.py's POST /chat/{thread_id}/edit). The
 * original thread is left completely untouched -- switch the UI over to
 * `EditResponseBody.thread_id` to show the branch.
 */
export async function editMessage(
  threadId: string,
  messageId: string,
  content: string,
  tool?: string,
): Promise<EditResponseBody> {
  const res = await fetch(`${API_BASE}/chat/${threadId}/edit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message_id: messageId, content, tool: tool ?? null }),
  });
  return handle<EditResponseBody>(res);
}

/**
 * Attach an image, PDF, or plain text file to a conversation's own personal RAG (see
 * agents/api.py's POST /chat/{thread_id}/upload and
 * local_rag/personal_rag.py's "temp" collection). `threadId` does not
 * need to have any chat history yet -- see that endpoint's own docstring.
 */
export async function uploadDocument(threadId: string, file: File): Promise<UploadResponseBody> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_BASE}/chat/${threadId}/upload`, {
    method: "POST",
    body: formData,
  });
  return handle<UploadResponseBody>(res);
}

export async function fetchHistory(threadId: string): Promise<HistoryResponseBody> {
  const res = await fetch(`${API_BASE}/chat/${threadId}/history`);
  return handle<HistoryResponseBody>(res);
}

export async function deleteThread(
  threadId: string,
): Promise<{ thread_id: string; deleted: boolean }> {
  const res = await fetch(`${API_BASE}/chat/${threadId}`, { method: "DELETE" });
  return handle(res);
}

/** Browse past conversations -- backs the chat-history sidebar. */
export async function fetchChats(limit = 30): Promise<ChatListResponseBody> {
  const res = await fetch(`${API_BASE}/chats?limit=${limit}`);
  return handle<ChatListResponseBody>(res);
}

/**
 * Copy a thread's history, as of right now, onto a brand-new
 * independent thread_id -- backs the sidebar's per-chat "Branch" option
 * (see components/ChatHistory.tsx). The two threads share no further
 * state after this call: continuing either one never touches the other.
 */
export async function branchThread(threadId: string): Promise<BranchResponseBody> {
  const res = await fetch(`${API_BASE}/chat/${threadId}/branch`, { method: "POST" });
  return handle<BranchResponseBody>(res);
}

/** Valid values for ChatRequestBody.tool -- backs the tool-selector dropdown. */
export async function fetchTools(): Promise<ToolListResponseBody> {
  const res = await fetch(`${API_BASE}/tools`);
  return handle<ToolListResponseBody>(res);
}

/**
 * Groq free-tier rate-limit usage -- backs the small usage badge at the
 * top of the chat window (see components/Thread.tsx / App.tsx). Polled
 * on an interval rather than pushed, since this is a low-stakes,
 * eventually-consistent display (see agents/api.py's GET /v1/usage for
 * where the numbers actually come from -- straight off Groq's own
 * response headers, not a separate counter this frontend keeps itself).
 */
export async function fetchUsage(): Promise<UsageResponseBody> {
  const res = await fetch(`${API_BASE}/v1/usage`);
  return handle<UsageResponseBody>(res);
}
