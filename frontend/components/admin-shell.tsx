"use client";

import { ChangeEvent, FormEvent, useCallback, useEffect, useState } from "react";
import Image from "next/image";
import Link from "next/link";
import logo from "../images/logo.png";
import {
  deleteDocument,
  DocumentSummary,
  fetchDocuments,
  fetchStatus,
  loginAdmin,
  rebuildIndex,
  SystemStatus,
  uploadDocument,
  verifyAdminSession,
} from "../lib/api";

const SESSION_KEY = "synkora_admin_session";

type AdminSessionState = {
  token: string;
  expiresAt: number;
};

const UploadIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12" />
  </svg>
);

const PdfIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
    <path d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
    <path d="M13 3v5a1 1 0 001 1h5" />
    <path d="M8 13h8M8 17h5" />
  </svg>
);

function formatTime(value: string): string {
  return new Intl.DateTimeFormat("en", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(value));
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

function readStoredSession(): AdminSessionState | null {
  if (typeof window === "undefined") return null;

  try {
    const raw = window.sessionStorage.getItem(SESSION_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as AdminSessionState;
    if (!parsed.token || parsed.expiresAt <= Date.now()) {
      window.sessionStorage.removeItem(SESSION_KEY);
      return null;
    }
    return parsed;
  } catch {
    window.sessionStorage.removeItem(SESSION_KEY);
    return null;
  }
}

function storeSession(session: AdminSessionState | null): void {
  if (typeof window === "undefined") return;

  if (!session) {
    window.sessionStorage.removeItem(SESSION_KEY);
    return;
  }

  window.sessionStorage.setItem(SESSION_KEY, JSON.stringify(session));
}

export function AdminShell() {
  const [session, setSession] = useState<AdminSessionState | null>(null);
  const [checkingSession, setCheckingSession] = useState(true);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isLoggingIn, setIsLoggingIn] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [reindexingId, setReindexingId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [documentToDelete, setDocumentToDelete] = useState<DocumentSummary | null>(null);

  const refresh = useCallback(async () => {
    const [sys, docs] = await Promise.all([fetchStatus(), fetchDocuments()]);
    setStatus(sys);
    setDocuments(docs);
  }, []);

  useEffect(() => {
    let active = true;

    const boot = async () => {
      const stored = readStoredSession();
      if (!stored) {
        if (active) setCheckingSession(false);
        return;
      }

      try {
        await verifyAdminSession(stored.token);
        if (!active) return;
        setSession(stored);
        await refresh();
      } catch {
        storeSession(null);
      } finally {
        if (active) setCheckingSession(false);
      }
    };

    void boot();
    return () => {
      active = false;
    };
  }, [refresh]);

  const handleLogin = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setIsLoggingIn(true);
    setError(null);

    try {
      const result = await loginAdmin(username.trim(), password);
      const nextSession = {
        token: result.access_token,
        expiresAt: Date.now() + result.expires_in * 1000,
      };
      storeSession(nextSession);
      setSession(nextSession);
      setPassword("");
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to sign in.");
    } finally {
      setIsLoggingIn(false);
    }
  };

  const handleLogout = () => {
    storeSession(null);
    setSession(null);
    setDocuments([]);
    setStatus(null);
    setUsername("");
    setPassword("");
    setError(null);
  };

  const handleUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file || !session) return;

    setIsUploading(true);
    setError(null);
    try {
      await uploadDocument(file, session.token);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      event.target.value = "";
      setIsUploading(false);
    }
  };

  const handleReindex = async (documentId: string) => {
    if (!session) return;

    setReindexingId(documentId);
    setError(null);
    try {
      await rebuildIndex(documentId, session.token);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to reindex.");
    } finally {
      setReindexingId(null);
    }
  };

  const handleDelete = async () => {
    if (!session || !documentToDelete) return;

    setDeletingId(documentToDelete.id);
    setError(null);
    try {
      await deleteDocument(documentToDelete.id, session.token);
      setDocumentToDelete(null);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete document.");
    } finally {
      setDeletingId(null);
    }
  };

  const readyCount = documents.filter((document) => document.status === "ready").length;

  if (checkingSession) {
    return (
      <main className="admin-auth-page">
        <div className="admin-auth-panel">Checking admin session...</div>
      </main>
    );
  }

  if (!session) {
    return (
      <main className="admin-auth-page">
        <form className="admin-auth-panel" onSubmit={handleLogin}>
          <Image alt="Synkora AI" className="admin-brand-logo" priority src={logo} />
          <div>
            <h1>Admin login</h1>
            <p>Sign in to upload and reindex the PDF knowledge base.</p>
          </div>

          {error && <div className="error-banner">{error}</div>}

          <label className="admin-field">
            <span>Username</span>
            <input
              autoComplete="username"
              onChange={(event) => setUsername(event.target.value)}
              value={username}
            />
          </label>
          <label className="admin-field">
            <span>Password</span>
            <input
              autoComplete="current-password"
              onChange={(event) => setPassword(event.target.value)}
              type="password"
              value={password}
            />
          </label>

          <button className="admin-primary-btn" disabled={isLoggingIn} type="submit">
            {isLoggingIn ? "Signing in..." : "Sign in"}
          </button>

          <Link className="admin-secondary-link" href="/">
            Back to chat
          </Link>
        </form>
      </main>
    );
  }

  return (
    <div className="app-root admin-root">
      <nav className="topnav">
        <Link className="topnav-logo" href="/" aria-label="Synkora AI chat">
          <div className="logo-mark logo-mark--brand">
            <Image alt="Synkora AI" className="brand-logo-image" priority src={logo} />
          </div>
        </Link>

        <div className="topnav-end">
          <Link className="admin-nav-link" href="/">
            Chat
          </Link>
          <button className="admin-nav-link" onClick={handleLogout} type="button">
            Sign out
          </button>
        </div>
      </nav>

      <main className="admin-workspace">
        <section className="admin-header">
          <div>
            <p className="admin-eyebrow">Document administration</p>
            <h1>Manage the PDF library</h1>
            <p>
              Upload source files, monitor indexing, and reindex documents when the knowledge base changes.
            </p>
          </div>
          <label className="upload-btn admin-upload-btn">
            <UploadIcon />
            {isUploading ? "Uploading..." : "Upload PDF"}
            <input
              accept="application/pdf"
              className="hidden-file-input"
              disabled={isUploading}
              onChange={handleUpload}
              type="file"
            />
          </label>
        </section>

        {error && <div className="error-banner admin-error">{error}</div>}

        <section className="admin-stats">
          <div className="admin-stat">
            <span>Total files</span>
            <strong>{status?.document_count ?? documents.length}</strong>
          </div>
          <div className="admin-stat">
            <span>Ready</span>
            <strong>{readyCount}</strong>
          </div>
          <div className="admin-stat">
            <span>Model</span>
            <strong>{status?.chat_model ?? "Loading..."}</strong>
          </div>
        </section>

        <section className="admin-library">
          <div className="admin-section-head">
            <h2>Library</h2>
            <button className="admin-secondary-btn" onClick={() => void refresh()} type="button">
              Refresh
            </button>
          </div>

          {documents.length === 0 ? (
            <div className="admin-empty">
              <PdfIcon />
              No PDFs have been uploaded yet.
            </div>
          ) : (
            <div className="admin-table" role="table" aria-label="Uploaded documents">
              {documents.map((document) => (
                <div className="admin-table-row" key={document.id} role="row">
                  <div className="admin-doc-title">
                    <PdfIcon />
                    <div>
                      <strong title={document.file_name}>{document.file_name}</strong>
                      <span>
                        {document.page_count ?? 0} pages - {document.chunk_count ?? 0} chunks
                      </span>
                    </div>
                  </div>
                  <span className={`status-chip status-${document.status}`}>
                    {formatStatus(document.status)}
                  </span>
                  <span className="admin-updated">{formatTime(document.updated_at)}</span>
                  <div className="admin-row-actions">
                    <button
                      className="reindex-btn admin-reindex-btn"
                      disabled={reindexingId === document.id || deletingId === document.id}
                      onClick={() => void handleReindex(document.id)}
                      type="button"
                    >
                      {reindexingId === document.id ? "Reindexing..." : "Reindex"}
                    </button>
                    <button
                      className="admin-danger-btn"
                      disabled={deletingId === document.id || reindexingId === document.id}
                      onClick={() => setDocumentToDelete(document)}
                      type="button"
                    >
                      {deletingId === document.id ? "Deleting..." : "Delete"}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>
      </main>

      {documentToDelete && (
        <div className="admin-modal-backdrop" role="presentation">
          <div
            aria-labelledby="delete-document-title"
            aria-modal="true"
            className="admin-confirm-modal"
            role="dialog"
          >
            <h2 id="delete-document-title">Delete document?</h2>
            <p>
              This will remove the uploaded PDF and its indexed chunks from the consultation knowledge base.
            </p>
            <div className="admin-delete-target">
              <PdfIcon />
              <span>{documentToDelete.file_name}</span>
            </div>
            <div className="admin-modal-actions">
              <button
                className="admin-secondary-btn"
                disabled={deletingId === documentToDelete.id}
                onClick={() => setDocumentToDelete(null)}
                type="button"
              >
                Cancel
              </button>
              <button
                className="admin-danger-btn admin-danger-btn--solid"
                disabled={deletingId === documentToDelete.id}
                onClick={() => void handleDelete()}
                type="button"
              >
                {deletingId === documentToDelete.id ? "Deleting..." : "Delete document"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
