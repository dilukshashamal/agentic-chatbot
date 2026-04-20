export type Citation = {
  chunk_id: string;
  source: string;
  page: number | null;
  score: number;
  excerpt: string;
};

export type DocumentSummary = {
  id: string;
  file_name: string;
  status: "uploaded" | "processing" | "ready" | "failed";
  page_count: number | null;
  chunk_count: number | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
};

export type ChatResponse = {
  conversation_id: string;
  answer: string;
  grounded: boolean;
  confidence: number;
  answer_mode: "grounded" | "insufficient_context";
  citations: Citation[];
  retrieved_chunks: number;
  system_notes: string[];
  route: string;
  memory_summary?: string | null;
  agent_trace?: Array<{
    agent: string;
    status: "completed" | "skipped" | "fallback" | "failed";
    summary: string;
    retries: number;
  }>;
  exports?: Array<{
    format: "json" | "pdf" | "docx";
    path: string;
    created_at: string;
  }>;
};

export type SystemStatus = {
  status: "ok";
  api_name: string;
  document_count: number;
  ready_document_count: number;
  chat_model: string;
  embedding_model: string;
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const maybeJson = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(maybeJson?.detail ?? "The server returned an unexpected error.");
  }

  return (await response.json()) as T;
}

export async function fetchStatus(): Promise<SystemStatus> {
  const response = await fetch(`${API_BASE_URL}/health`, {
    cache: "no-store",
  });

  return handleResponse<SystemStatus>(response);
}

export async function fetchDocuments(): Promise<DocumentSummary[]> {
  const response = await fetch(`${API_BASE_URL}/api/v1/documents`, {
    cache: "no-store",
  });

  return handleResponse<DocumentSummary[]>(response);
}

export async function uploadDocument(file: File): Promise<{ message: string; document: DocumentSummary }> {
  const body = new FormData();
  body.append("file", file);

  const response = await fetch(`${API_BASE_URL}/api/v1/documents/upload`, {
    method: "POST",
    body,
  });

  return handleResponse<{ message: string; document: DocumentSummary }>(response);
}

export async function askQuestion(
  query: string,
  conversationId?: string | null,
): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/chat/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      conversation_id: conversationId ?? null,
      query,
      top_k: 4,
      include_sources: true,
    }),
  });

  return handleResponse<ChatResponse>(response);
}

export async function rebuildIndex(documentId: string): Promise<{ message: string; chunk_count: number }> {
  const response = await fetch(`${API_BASE_URL}/api/v1/documents/${documentId}/reindex`, {
    method: "POST",
  });

  return handleResponse<{ message: string; chunk_count: number }>(response);
}
