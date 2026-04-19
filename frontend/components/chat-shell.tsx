"use client";

import { ChangeEvent, FormEvent, useEffect, useState, useTransition } from "react";
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

const STARTER_PROMPTS = [
  "Summarize the uploaded documents.",
  "What are the key topics across the files?",
  "Explain the main ideas simply.",
];

function formatSourceSummary(response: ChatResponse): string | null {
  const documents = new Set(response.citations.map((citation) => citation.source));
  const pages = response.citations.length;

  if (pages === 0) {
    return null;
  }

  const documentLabel = documents.size === 1 ? "document" : "documents";
  const sourceLabel = pages === 1 ? "source" : "sources";
  return `${pages} ${sourceLabel} from ${documents.size} ${documentLabel}`;
}

function formatStatus(status: DocumentSummary["status"]): string {
  switch (status) {
    case "ready":
      return "Ready";
    case "processing":
      return "Indexing";
    case "failed":
      return "Failed";
    default:
      return "Uploaded";
  }
}

function formatTime(value: string): string {
  return new Intl.DateTimeFormat("en", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(value));
}

export function ChatShell() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [query, setQuery] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();
  const [isUploading, setIsUploading] = useState(false);
  const [reindexingId, setReindexingId] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    const refreshWorkspace = async () => {
      try {
        const [systemStatus, uploadedDocuments] = await Promise.all([
          fetchStatus(),
          fetchDocuments(),
        ]);

        if (!active) {
          return;
        }

        setStatus(systemStatus);
        setDocuments(uploadedDocuments);
      } catch (err) {
        if (active) {
          setError(err instanceof Error ? err.message : "Failed to load documents.");
        }
      }
    };

    void refreshWorkspace();

    const interval = window.setInterval(() => {
      void refreshWorkspace();
    }, 5000);

    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, []);

  const readyCount = documents.filter((document) => document.status === "ready").length;

  const submitQuestion = (nextQuery: string) => {
    const cleaned = nextQuery.trim();
    if (!cleaned) {
      return;
    }

    if (readyCount === 0) {
      setError("Upload a PDF and wait for indexing before asking questions.");
      return;
    }

    setError(null);
    setQuery("");

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: "user",
      content: cleaned,
    };

    setMessages((current) => [...current, userMessage]);

    startTransition(async () => {
      try {
        const response = await askQuestion(cleaned);
        setMessages((current) => [
          ...current,
          {
            id: `assistant-${Date.now()}`,
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

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    submitQuestion(query);
  };

  const handleRefresh = async (documentId: string) => {
    setReindexingId(documentId);
    setError(null);

    try {
      await rebuildIndex(documentId);
      const [nextStatus, nextDocuments] = await Promise.all([fetchStatus(), fetchDocuments()]);
      setStatus(nextStatus);
      setDocuments(nextDocuments);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to reindex the document.");
    } finally {
      setReindexingId(null);
    }
  };

  const handleUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    setIsUploading(true);
    setError(null);

    try {
      await uploadDocument(file);
      const [nextStatus, nextDocuments] = await Promise.all([fetchStatus(), fetchDocuments()]);
      setStatus(nextStatus);
      setDocuments(nextDocuments);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to upload the document.");
    } finally {
      event.target.value = "";
      setIsUploading(false);
    }
  };

  return (
    <main className="app-shell">
      <section className="app-frame">
        <header className="topbar">
          <div>
            <span className="eyebrow">Document Studio</span>
            <h1>Ask across your uploaded PDFs</h1>
            <p>The assistant searches every ready document automatically, so there is no manual selection step.</p>
          </div>

          <div className="topbar-stats">
            <div className="stat-pill">
              <span>Total</span>
              <strong>{status?.document_count ?? documents.length}</strong>
            </div>
            <div className="stat-pill">
              <span>Ready</span>
              <strong>{readyCount}</strong>
            </div>
            <label className="upload-button">
              <input
                accept="application/pdf"
                className="hidden-input"
                onChange={handleUpload}
                type="file"
              />
              {isUploading ? "Uploading..." : "Upload PDF"}
            </label>
          </div>
        </header>

        <div className="workspace-grid">
          <aside className="sidebar">
            <div className="panel panel-tight">
              <div className="panel-head">
                <div>
                  <span className="eyebrow">Library</span>
                  <h2>Documents</h2>
                </div>
              </div>

              <div className="document-list">
                {documents.length === 0 ? (
                  <div className="empty-card">
                    <p>No documents yet. Upload a PDF to begin.</p>
                  </div>
                ) : (
                  documents.map((document) => (
                    <div className="document-item static" key={document.id}>
                      <div className="document-item-head">
                        <strong>{document.file_name}</strong>
                        <span className={`status-badge status-${document.status}`}>
                          {formatStatus(document.status)}
                        </span>
                      </div>
                      <p>
                        {document.status === "ready"
                          ? `${document.chunk_count ?? 0} chunks - ${document.page_count ?? 0} pages`
                          : document.error_message ?? "Preparing document index."}
                      </p>
                      <div className="document-item-footer">
                        <span className="item-meta">{formatTime(document.updated_at)}</span>
                        <button
                          className="mini-button"
                          disabled={reindexingId === document.id}
                          onClick={() => handleRefresh(document.id)}
                          type="button"
                        >
                          {reindexingId === document.id ? "Reindexing..." : "Reindex"}
                        </button>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </aside>

          <section className="main-column">
            <div className="panel selected-panel">
              <div className="selected-summary">
                <div>
                  <span className="eyebrow">Search Mode</span>
                  <h2>All ready documents</h2>
                  <p>
                    Every question searches the full ready-document library and returns citations showing which file supported the answer.
                  </p>
                </div>

                <div className="selected-actions">
                  <div className="summary-chip">
                    <span>Scope</span>
                    <strong>Library-wide</strong>
                  </div>
                  <div className="summary-chip">
                    <span>Model</span>
                    <strong>{status?.chat_model ?? "Loading..."}</strong>
                  </div>
                </div>
              </div>

              <div className="prompt-row">
                {STARTER_PROMPTS.map((prompt) => (
                  <button
                    key={prompt}
                    className="prompt-chip"
                    disabled={readyCount === 0}
                    onClick={() => submitQuestion(prompt)}
                    type="button"
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>

            <div className="panel chat-panel">
              <div className="messages">
                {messages.length === 0 ? (
                  <div className="empty-state">
                    <h3>{readyCount > 0 ? "Ask your first question" : "Waiting for a ready document"}</h3>
                    <p>
                      {readyCount > 0
                        ? "Questions now run across all ready PDFs automatically."
                        : "The chat composer unlocks as soon as at least one uploaded PDF finishes indexing."}
                    </p>
                  </div>
                ) : (
                  messages.map((message) => {
                    const sourceSummary = message.response ? formatSourceSummary(message.response) : null;

                    return (
                      <article
                        className={`message ${message.role === "assistant" ? "assistant" : "user"}`}
                        key={message.id}
                      >
                        <div className="message-head">
                          <span className="message-role">
                            {message.role === "assistant" ? "Assistant" : "You"}
                          </span>
                          {message.response ? (
                            <div className="message-meta">
                              <span className="meta-chip">
                                {Math.round(message.response.confidence * 100)}% confidence
                              </span>
                              <span className="meta-chip">
                                {message.response.grounded ? "Grounded" : "Insufficient context"}
                              </span>
                            </div>
                          ) : null}
                        </div>

                        <p>{message.content}</p>

                        {sourceSummary ? <span className="page-tag">{sourceSummary}</span> : null}

                        {message.response?.citations.length ? (
                          <details className="sources-panel">
                            <summary>
                              View supporting sources
                            </summary>
                            <div className="sources">
                              {message.response.citations.map((citation) => (
                                <div className="source-card" key={citation.chunk_id}>
                                  <div className="source-head">
                                    <strong>{citation.source}</strong>
                                    <span>Page {citation.page ?? "?"}</span>
                                  </div>
                                  <p>{citation.excerpt}</p>
                                </div>
                              ))}
                            </div>
                          </details>
                        ) : null}
                      </article>
                    );
                  })
                )}

                {isPending ? (
                  <article className="message assistant pending">
                    <div className="message-head">
                      <span className="message-role">Assistant</span>
                    </div>
                    <p>Searching your ready document library...</p>
                  </article>
                ) : null}
              </div>

              <form className="composer" onSubmit={handleSubmit}>
                <textarea
                  aria-label="Ask a question"
                  className="composer-input"
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Ask a question about any uploaded PDF..."
                  rows={4}
                  value={query}
                />
                <button
                  className="primary-button"
                  disabled={isPending || !query.trim() || readyCount === 0}
                  type="submit"
                >
                  {isPending ? "Thinking..." : "Send"}
                </button>
              </form>
            </div>

            {error ? <p className="error-text">{error}</p> : null}
          </section>
        </div>
      </section>
    </main>
  );
}
