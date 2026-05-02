"use client";

import {
  ChangeEvent,
  FormEvent,
  KeyboardEvent,
  useEffect,
  useRef,
  useState,
  useTransition,
} from "react";
import {
  askQuestion,
  ChatResponse,
  DocumentSummary,
  fetchDocuments,
  fetchStatus,
  rebuildIndex,
  SystemStatus,
  uploadDocument,
} from "../lib/api";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  response?: ChatResponse;
};

const STARTERS = [
  {
    icon: (
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
      >
        <path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    ),
    color: "#e8f0fe",
    iconColor: "#4285f4",
    text: "Summarize all uploaded documents",
  },
  {
    icon: (
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
      >
        <circle cx="11" cy="11" r="8" />
        <path d="m21 21-4.35-4.35" />
      </svg>
    ),
    color: "#e6f4ea",
    iconColor: "#1e8e3e",
    text: "What are the key topics across the files?",
  },
  {
    icon: (
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
      >
        <path d="M9.663 17h4.673M12 3v1m6.364 1.636-.707.707M21 12h-1M4 12H3m3.343-5.657-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
      </svg>
    ),
    color: "#fef9e7",
    iconColor: "#f9ab00",
    text: "Explain the main ideas in simple terms",
  },
];

function formatTime(value: string): string {
  const d = new Date(value);
  const now = new Date();
  const diffMs = now.getTime() - d.getTime();
  const diffHrs = diffMs / 3600000;
  if (diffHrs < 24) {
    return new Intl.DateTimeFormat("en", {
      hour: "numeric",
      minute: "2-digit",
    }).format(d);
  }
  return new Intl.DateTimeFormat("en", {
    month: "short",
    day: "numeric",
  }).format(d);
}

function formatStatus(status: DocumentSummary["status"]): string {
  const map: Record<string, string> = {
    ready: "Ready",
    processing: "Indexing",
    failed: "Failed",
    uploaded: "Queued",
  };
  return map[status] ?? status;
}

function formatSourceSummary(response: ChatResponse): string | null {
  const docs = new Set(response.citations.map((c) => c.source));
  const p = response.citations.length;
  if (p === 0) return null;
  return `${p} source${p !== 1 ? "s" : ""} · ${docs.size} file${docs.size !== 1 ? "s" : ""}`;
}

// ─── Icons ─────────────────────────────────────────────────
const SendIcon = () => (
  <svg viewBox="0 0 24 24" fill="currentColor">
    <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
  </svg>
);

const UploadIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12" />
  </svg>
);

const PdfIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
    <path d="M7 21H17a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
    <path d="M13 3v5a1 1 0 001 1h5" />
    <path d="M10.5 14.5v-2h1a1 1 0 010 2h-1z" />
    <path d="M9 12v5M13.5 12h-1v5h1a1.5 1.5 0 000-3h-1M16 12v5M16 14.5h1.5" />
  </svg>
);

const LibraryIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
    <path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01" />
  </svg>
);

const SourcesIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M10 13a5 5 0 007.54.54l3-3a5 5 0 00-7.07-7.07l-1.72 1.71" />
    <path d="M14 11a5 5 0 00-7.54-.54l-3 3a5 5 0 007.07 7.07l1.71-1.71" />
  </svg>
);

const ErrorIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="10" />
    <path d="M12 8v4M12 16h.01" />
  </svg>
);

const SynkoraLogo = () => (
  <svg viewBox="0 0 18 18" fill="white" xmlns="http://www.w3.org/2000/svg">
    <path
      d="M9 1.5C5.5 1.5 2.5 4.5 2.5 8c0 2 .9 3.8 2.3 5l-.8 2.5 2.5-.8C7.2 15.1 8.1 15.5 9 15.5c3.5 0 6.5-3 6.5-6.5S12.5 1.5 9 1.5z"
      opacity=".2"
    />
    <path d="M9 2.5C6 2.5 3.5 5 3.5 8S6 13.5 9 13.5c.8 0 1.5-.2 2.2-.5l2.3.8-.8-2.3c.5-.8.8-1.7.8-2.5C13.5 5 11 2.5 9 2.5zm0 1.5c2.2 0 4 1.8 4 4S11.2 12 9 12c-.7 0-1.4-.2-2-.6l-.2-.1-1.4.5.5-1.4-.1-.2C5.2 9.6 5 8.8 5 8.1c0-2.3 1.8-4.1 4-4.1z" />
  </svg>
);

// ─── Main Component ─────────────────────────────────────────
export function ChatShell() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();
  const [isUploading, setIsUploading] = useState(false);
  const [reindexingId, setReindexingId] = useState<string | null>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    let active = true;
    const refresh = async () => {
      try {
        const [sys, docs] = await Promise.all([
          fetchStatus(),
          fetchDocuments(),
        ]);
        if (!active) return;
        setStatus(sys);
        setDocuments(docs);
      } catch (err) {
        if (active)
          setError(err instanceof Error ? err.message : "Failed to load.");
      }
    };
    void refresh();
    const interval = setInterval(() => void refresh(), 5000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isPending]);

  const readyCount = documents.filter((d) => d.status === "ready").length;

  const submitQuestion = (nextQuery: string) => {
    const cleaned = nextQuery.trim();
    if (!cleaned) return;
    if (readyCount === 0) {
      setError("Upload and index a PDF before asking questions.");
      return;
    }
    setError(null);
    setQuery("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";

    const userMsg: Message = {
      id: `u-${Date.now()}`,
      role: "user",
      content: cleaned,
    };
    setMessages((cur) => [...cur, userMsg]);

    startTransition(async () => {
      try {
        const response = await askQuestion(cleaned, conversationId);
        setConversationId(response.conversation_id);
        setMessages((cur) => [
          ...cur,
          {
            id: `a-${Date.now()}`,
            role: "assistant",
            content: response.answer,
            response,
          },
        ]);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Something went wrong.");
      }
    });
  };

  const handleSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    submitQuestion(query);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submitQuestion(query);
    }
  };

  const handleTextareaChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setQuery(e.target.value);
    const ta = e.target;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 200)}px`;
  };

  const handleRefresh = async (documentId: string) => {
    setReindexingId(documentId);
    setError(null);
    try {
      await rebuildIndex(documentId);
      const [sys, docs] = await Promise.all([fetchStatus(), fetchDocuments()]);
      setStatus(sys);
      setDocuments(docs);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to reindex.");
    } finally {
      setReindexingId(null);
    }
  };

  const handleUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setIsUploading(true);
    setError(null);
    try {
      await uploadDocument(file);
      const [sys, docs] = await Promise.all([fetchStatus(), fetchDocuments()]);
      setStatus(sys);
      setDocuments(docs);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      e.target.value = "";
      setIsUploading(false);
    }
  };

  return (
    <div className="app-root">
      {/* ── Top Nav ── */}
      <nav className="topnav">
        <a className="topnav-logo" href="#">
          <div className="logo-mark">
            <SynkoraLogo />
          </div>
          <span className="logo-wordmark">
            Synkora <span>AI</span>
          </span>
        </a>

        <div className="topnav-end">
          <div className="model-badge">
            <div className="model-dot" />
            {status?.chat_model ?? "Loading..."}
          </div>
          <div className="avatar">S</div>
        </div>
      </nav>

      {/* ── App Body ── */}
      <div className="app-body">
        {/* ── Sidebar ── */}
        <aside className="sidebar">
          {/* Upload section */}
          <div className="sidebar-section">
            <label className="upload-btn">
              <UploadIcon />
              {isUploading ? "Uploading..." : "Upload PDF"}
              <input
                type="file"
                accept="application/pdf"
                className="hidden-file-input"
                onChange={handleUpload}
                disabled={isUploading}
              />
            </label>

            <div className="stats-row">
              <div className="stat-card">
                <div className="stat-value">
                  {status?.document_count ?? documents.length}
                </div>
                <div className="stat-label">Total files</div>
              </div>
              <div className="stat-card">
                <div className="stat-value" style={{ color: "var(--green)" }}>
                  {readyCount}
                </div>
                <div className="stat-label">Ready</div>
              </div>
            </div>
          </div>

          {/* Document list */}
          <div className="sidebar-section">
            <div className="sidebar-label">
              <LibraryIcon />
              Library
            </div>

            {documents.length === 0 ? (
              <div className="doc-empty">
                <PdfIcon />
                No documents yet.
                <br />
                Upload a PDF to begin.
              </div>
            ) : (
              <div className="doc-list">
                {documents.map((doc) => (
                  <div className="doc-item" key={doc.id}>
                    <div className="doc-item-head">
                      <div className="doc-icon">
                        <PdfIcon />
                      </div>
                      <div className="doc-info">
                        <div className="doc-name" title={doc.file_name}>
                          {doc.file_name}
                        </div>
                        <div className="doc-meta">
                          {doc.status === "ready"
                            ? `${doc.page_count ?? 0} pages · ${doc.chunk_count ?? 0} chunks`
                            : (doc.error_message ?? "Preparing...")}
                        </div>
                      </div>
                    </div>
                    <div className="doc-footer">
                      <span className={`status-chip status-${doc.status}`}>
                        {formatStatus(doc.status)}
                      </span>
                      <div
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: "6px",
                        }}
                      >
                        <span
                          style={{
                            fontSize: "11px",
                            color: "var(--text-tertiary)",
                          }}
                        >
                          {formatTime(doc.updated_at)}
                        </span>
                        <button
                          className="reindex-btn"
                          disabled={reindexingId === doc.id}
                          onClick={() => handleRefresh(doc.id)}
                          type="button"
                        >
                          {reindexingId === doc.id ? "..." : "Reindex"}
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </aside>

        {/* ── Main Column ── */}
        <main className="main-col">
          {/* Context bar */}
          <div className="context-bar">
            <div className={`context-chip ${readyCount > 0 ? "active" : ""}`}>
              <div className="context-chip-dot" />
              {readyCount > 0
                ? `Searching ${readyCount} ready file${readyCount !== 1 ? "s" : ""}`
                : "No files ready"}
            </div>
            {status && (
              <div className="context-chip">
                <svg
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <circle cx="12" cy="12" r="3" />
                  <path d="M19.07 4.93a10 10 0 010 14.14M4.93 4.93a10 10 0 000 14.14" />
                </svg>
                Hybrid retrieval
              </div>
            )}
          </div>

          {/* Chat area */}
          <div className="chat-area">
            {messages.length === 0 ? (
              <div className="welcome-state">
                <h1 className="welcome-title">
                  Hello, I&apos;m <span className="blue">Synkora</span>{" "}
                  <span className="purple">AI</span>
                </h1>
                <p className="welcome-sub">
                  Ask anything about your uploaded PDFs. I search across all
                  ready documents and cite exactly where each answer comes from.
                </p>

                <div className="starter-grid">
                  {STARTERS.map((s, i) => (
                    <button
                      key={i}
                      className="starter-card"
                      disabled={readyCount === 0}
                      onClick={() => submitQuestion(s.text)}
                    >
                      <div
                        className="starter-card-icon"
                        style={{ background: s.color, color: s.iconColor }}
                      >
                        {s.icon}
                      </div>
                      <span className="starter-card-text">{s.text}</span>
                    </button>
                  ))}
                </div>

                {readyCount === 0 && (
                  <div
                    style={{
                      padding: "16px 20px",
                      background: "var(--surface)",
                      border: "1px solid var(--border)",
                      borderRadius: "var(--radius-xl)",
                      fontSize: "14px",
                      color: "var(--text-secondary)",
                      lineHeight: 1.6,
                    }}
                  >
                    <strong style={{ color: "var(--text-primary)" }}>
                      Get started
                    </strong>
                    <br />
                    Upload a PDF using the sidebar. Once indexing completes, the
                    starter prompts will unlock and you can begin chatting.
                  </div>
                )}
              </div>
            ) : (
              <div className="messages">
                {messages.map((msg) => {
                  if (msg.role === "user") {
                    return (
                      <div className="msg-user" key={msg.id}>
                        <div className="msg-user-bubble">{msg.content}</div>
                      </div>
                    );
                  }

                  const sourceSummary = msg.response
                    ? formatSourceSummary(msg.response)
                    : null;
                  return (
                    <div className="msg-assistant" key={msg.id}>
                      <div className="msg-assistant-avatar">
                        <SynkoraLogo />
                      </div>
                      <div className="msg-assistant-body">
                        {msg.response && (
                          <div className="msg-chips">
                            <span className="msg-chip chip-confidence">
                              {Math.round(msg.response.confidence * 100)}%
                              confidence
                            </span>
                            <span
                              className={`msg-chip ${msg.response.grounded ? "chip-grounded" : "chip-ungrounded"}`}
                            >
                              {msg.response.grounded
                                ? "Grounded"
                                : "Insufficient context"}
                            </span>
                            {sourceSummary && (
                              <span className="msg-chip chip-sources">
                                {sourceSummary}
                              </span>
                            )}
                          </div>
                        )}

                        <p className="msg-answer">{msg.content}</p>

                        {msg.response?.trace_id || msg.response?.exports?.length || msg.response?.agent_trace?.length ? (
                          <details>
                            <summary className="citations-toggle">
                              Debug & exports
                            </summary>
                            <div className="citations-grid">
                              {msg.response?.trace_id ? <div className="citation-card"><strong>Trace ID:</strong> {msg.response.trace_id}</div> : null}
                              {msg.response?.agent_trace?.length ? <div className="citation-card"><strong>Agent steps:</strong> {msg.response.agent_trace.map((step) => `${step.agent} (${step.status})`).join(', ')}</div> : null}
                              {msg.response?.exports?.length ? <div className="citation-card"><strong>Exports:</strong> {msg.response.exports.map((artifact) => `${artifact.format.toUpperCase()} @ ${artifact.path}`).join(' · ')}</div> : null}
                            </div>
                          </details>
                        ) : null}

                        {msg.response?.citations.length ? (
                          <details>
                            <summary className="citations-toggle">
                              <SourcesIcon />
                              View {msg.response.citations.length} supporting
                              source
                              {msg.response.citations.length !== 1 ? "s" : ""}
                            </summary>
                            <div className="citations-grid">
                              {msg.response.citations.map((c) => (
                                <div className="citation-card" key={c.chunk_id}>
                                  <div className="citation-head">
                                    <span
                                      className="citation-source"
                                      title={c.source}
                                    >
                                      {c.source}
                                    </span>
                                    {c.page && (
                                      <span className="citation-page">
                                        p.{c.page}
                                      </span>
                                    )}
                                  </div>
                                  <div className="citation-score">
                                    Score {Math.round(c.score * 100)}%
                                  </div>
                                  <p className="citation-excerpt">
                                    {c.excerpt}
                                  </p>
                                </div>
                              ))}
                            </div>
                          </details>
                        ) : null}
                      </div>
                    </div>
                  );
                })}

                {isPending && (
                  <div className="msg-pending msg-assistant">
                    <div className="msg-assistant-avatar">
                      <SynkoraLogo />
                    </div>
                    <div className="typing-indicator">
                      <div className="typing-dot" />
                      <div className="typing-dot" />
                      <div className="typing-dot" />
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>
            )}
          </div>

          {/* Composer */}
          <div className="composer-area">
            <div className="composer-wrap">
              {error && (
                <div className="error-banner">
                  <ErrorIcon />
                  {error}
                </div>
              )}
              <form onSubmit={handleSubmit}>
                <div className="composer-box">
                  <textarea
                    ref={textareaRef}
                    className="composer-textarea"
                    value={query}
                    onChange={handleTextareaChange}
                    onKeyDown={handleKeyDown}
                    placeholder="Ask a question about your documents..."
                    rows={1}
                    aria-label="Ask a question"
                  />
                  <div className="composer-footer">
                    <div className="composer-tools">
                      <button
                        className="composer-tool-btn"
                        type="button"
                        title="Attach file"
                      >
                        <svg
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                        >
                          <path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48" />
                        </svg>
                      </button>
                    </div>
                    <button
                      className="send-btn"
                      type="submit"
                      disabled={isPending || !query.trim() || readyCount === 0}
                      title="Send"
                    >
                      <SendIcon />
                    </button>
                  </div>
                </div>
                <p className="composer-hint">
                  Press Enter to send · Shift+Enter for new line · Answers are
                  grounded in your PDFs
                </p>
              </form>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
