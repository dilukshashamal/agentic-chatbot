# Synkora AI RAG Chatbot

Synkora AI is a multi-document RAG platform for uploading PDFs, indexing them into `pgvector`, and chatting with grounded answers through a polished Next.js dashboard. The current stack now includes multi-agent orchestration, multi-level memory, model management, and built-in observability with `MLflow`, `Prometheus`, and `Grafana`.

![Dashboard overview](images/dashboard_img1.png)

## Highlights

- `FastAPI` backend with document upload, indexing, and chat APIs
- `Next.js` frontend for document management and chat
- `PostgreSQL + pgvector` for persistent vector search
- `MLflow + Prometheus + Grafana` for experiment tracking and monitoring
- background PDF indexing after upload
- multi-agent orchestration layer for routing, memory, grounding, analysis, and tool use
- conversation persistence with checkpoints for multi-turn workflows
- multi-level memory with short-term, long-term, and knowledge-graph layers
- model registry and experiment tracking for chat, embedding, retrieval, and prompt versions
- source citations attached to grounded answers

## Product Screens

### Main Dashboard

![Main dashboard view 1](images/dashboard_img1.png)

![Main dashboard](images/dashboard_img2.png)

### Admin Dashboard

![Admin dashboard](images/admin_img.png)

### MLflow Tracking

![MLflow tracking](images/mlflow.png)

### Prometheus Monitoring

![Prometheus monitoring](images/prometheus_dash.png)

### Grafana Observability

![Grafana observability](images/grafana_dash.png)

## Multi-Agent Flow

The backend now supports a supervisor-style orchestration flow built around specialized agents:

- `Router Agent`: chooses the best path for the user request
- `Document Understanding Agent`: summarization, entities, and key insights
- `Analytical Agent`: cross-document comparisons and trend-style reasoning
- `Citation Agent`: grounding validation and source attribution
- `Memory Agent`: conversation context and user preferences
- `Tool Use Agent`: search, calculation, transformation, charting, and export support

When optional libraries are unavailable, the system degrades gracefully to simpler retrieval instead of crashing.

## Memory Architecture

The backend now includes a multi-level memory system:

### Short-Term Memory

- last 10 conversation turns
- active document focus tracking
- query refinement history for follow-up handling

### Long-Term Memory

- user preferences and interaction patterns
- custom instructions and tone preferences
- document access frequency and recency
- FAQ tracking based on repeated user questions
- embedded conversation summaries for semantic recall

### Knowledge Graph Memory

- topic and entity nodes extracted from conversations
- relationship edges built from co-occurring entities/topics
- top topic tracking per conversation
- forgetting support for GDPR-style removal

### Memory Commands

Supported command-style prompts include:

- `remember this: ...`
- `remember that: ...`
- `forget this: ...`
- `forget that`
- `forget everything`

These commands are handled directly by the memory layer without requiring document retrieval.

### Memory Providers

The current production-ready behavior uses the local Postgres-backed memory store.

Config supports:

- `local`
- `mem0`
- `zep`

If `mem0` or `zep` is selected without real provider configuration, the system safely falls back to the local memory implementation.

## Architecture

```text
rag_chatbot/
|-- backend/
|   |-- alembic/
|   |-- app/
|   |   |-- api/routes/
|   |   |-- core/
|   |   |-- db/
|   |   |-- models/
|   |   `-- services/
|   |-- data/
|   |-- tests/
|   |-- Dockerfile
|   `-- requirements.txt
|-- frontend/
|   |-- app/
|   |-- components/
|   |-- lib/
|   |-- Dockerfile
|   `-- package.json
|-- images/
|-- monitoring/
|   |-- grafana/
|   |-- mlflow/
|   `-- prometheus/
|-- vectorstores/
|-- docker-compose.yml
|-- .env.example
|-- rag_notebook.ipynb
|-- legacy_streamlit_app.py
`-- README.md
```

## Dashboard

The current dashboard supports:

- PDF upload from the sidebar
- document readiness and indexing status
- grounded chat responses with confidence and citation badges
- multi-file retrieval across ready documents
- conversation continuity through backend conversation IDs

## Infrastructure

The local Docker stack now includes application services and an observability layer:

- `frontend`: Next.js dashboard on port `3000`
- `backend`: FastAPI RAG API on port `8000`
- `db`: PostgreSQL with `pgvector` on port `5432`
- `mlflow`: experiment tracking and registry metadata on port `5000`
- `prometheus`: metrics scraping and target inspection on port `9090`
- `grafana`: dashboarding and observability views on port `3001`

Monitoring and model-management coverage now includes:

- experiment logging for retrieval and generation runs
- model registry metadata for chat and embedding versions
- pipeline, prompt, and retrieval configuration version tracking
- Prometheus metrics for HTTP traffic, query latency, cost, indexing, shadow evaluations, and provider publishing
- a provisioned Grafana datasource and starter observability dashboard

## Architecture Overview

```mermaid
graph TD
    User([User]) -->|HTTP| Frontend[Next.js Dashboard Port 3000]
    Frontend -->|REST API| Backend[FastAPI Backend Port 8000]

    subgraph "Multi-Agent Orchestration"
        Backend --> Router[Router Agent]
        Router --> DocAgent[Doc Understanding Agent]
        Router --> AnalyticalAgent[Analytical Agent]
        Router --> CitationAgent[Citation Agent]
        Router --> MemoryAgent[Memory Agent]
        Router --> ToolAgent[Tool Use Agent]
    end

    subgraph "Observability Layer"
        Prometheus[Prometheus Port 9090] -->|Scrapes Metrics| Backend
        Grafana[Grafana Port 3001] -->|Visualizes| Prometheus
        Backend -->|Tracks Experiments| MLflow[MLflow Port 5000]
    end

    subgraph "Data Storage & Retrieval"
        Backend <-->|Semantic Search| PgVector[(PostgreSQL + pgvector)]
        Backend <-->|Hybrid Search| AzureSearch[(Azure AI Search)]
        Backend <-->|Short/Long/Graph Memory| PostgresMemory[(PostgreSQL Memory Store)]
    end

    subgraph "Azure Ecosystem"
        Backend <-->|Completions & Embeddings| AzureOpenAI{Azure OpenAI}
    end
```

## Azure Integration Deep Dive

The architecture is tightly integrated with Azure's enterprise-grade AI ecosystem, allowing for advanced retrieval, model security, and high performance.

### 1. Azure OpenAI (LLMs & Embeddings)

The core generation and embedding operations in the orchestration layer rely heavily on **Azure OpenAI**.
- **Chat Completions**: Used by all system agents (Router, Document Understanding, Analytical, Citation, etc.) to securely generate insights. The system routes requests via the `AZURE_OPENAI_CHAT_DEPLOYMENT` model deployment.
- **Embeddings**: Document chunks and user queries are embedded using the `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`. This guarantees that private company data never leaves the Azure tenant.
- **Implementation**: The backend configures `AzureChatOpenAI` and `AzureOpenAIEmbeddings` using standard enterprise authentication parameters: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION` (default `2024-12-01-preview`), and `AZURE_OPENAI_API_KEY`.

### 2. Azure AI Search (Vector Store & Hybrid Retrieval)

When `RETRIEVAL_BACKEND` is set to `azure_ai_search` or `hybrid`, the system delegates document indexing and semantic search to **Azure AI Search**.
- **Index Management**: The system automatically bootstraps the required search index (defined by `AZURE_AI_SEARCH_INDEX_NAME`) using a highly tuned schema that includes exact fields for `document_id`, `chunk_id`, raw `content`, and a `content_vector`.
- **HNSW Vector Profiling**: Under the hood, the backend configures `HnswAlgorithmConfiguration` (`hnsw-default`) ensuring low-latency, approximate nearest neighbor (ANN) retrieval across massive document datasets.
- **Chunk Lifecycle**: During document re-indexing, the system uses compound IDs (`document_id:chunk_index`) to cleanly purge old chunks before upserting the newly embedded payloads.
- **Score Normalization & Blending**: In the recommended `hybrid` mode, raw similarity scores returned by Azure AI Search are mathematically normalized (using `max(0.0, min(1.0, raw_score / (raw_score + 1.0)))`) so they can be smoothly blended with BM25 keyword overlaps and text quality scores. The results are merged with `pgvector` fallbacks to guarantee robust retrieval even if an Azure service goes offline.

## Docker Setup

### 1. Create the root `.env`

Copy `.env.example` to `.env` and set your provider-specific values.

For Azure OpenAI (recommended for your current setup):

```env
LLM_PROVIDER=azure_openai
RETRIEVAL_BACKEND=hybrid
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=https://law-consult-openai-prod.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-5.4
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large
CHAT_MODEL=gpt-5.4
EMBEDDING_MODEL=text-embedding-3-large

# Azure AI Search (required when RETRIEVAL_BACKEND=azure_ai_search or hybrid)
AZURE_AI_SEARCH_ENDPOINT=https://law-search-index.search.windows.net
AZURE_AI_SEARCH_API_KEY=your_azure_ai_search_api_key
AZURE_AI_SEARCH_INDEX_NAME=law-search-index

POSTGRES_DB=rag_chatbot
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_PORT=5432
```

For Gemini, keep `LLM_PROVIDER=google` and set `GOOGLE_API_KEY`.

Note: `AZURE_OPENAI_ENDPOINT` can be either the Azure OpenAI endpoint (`...openai.azure.com`) or the Cognitive Services endpoint (`...cognitiveservices.azure.com`) from Deployments + Endpoint.

### 2. Start the stack

From the project root:

```bash
docker compose up --build
```

Services:

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`
- Postgres: `localhost:5432`
- MLflow: `http://localhost:5000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3001`

The compose stack now brings up the app and observability layer together:

- `mlflow` for experiment tracking and registry metadata
- `prometheus` for scraping backend metrics
- `grafana` with a provisioned Prometheus datasource and starter dashboard

The backend also runs additive compatibility migrations at startup for evolving local schemas, so older Postgres volumes can usually be patched forward without being deleted.

If you recently pulled changes that added new tables or dependencies, rebuild the backend image so startup creates the new schema:

```bash
docker compose up --build backend
```

### 2.5 Run backend migrations (Alembic)

Before starting the backend for the first time (or after pulling schema changes), run:

```bash
cd backend
alembic upgrade head
```

The backend startup also attempts to apply pending migrations automatically, but running this command explicitly is recommended for production rollouts.

### 3. Monitoring Defaults

Optional additions to the root `.env`:

```env
MODEL_REGISTRY_PROVIDER=mlflow
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_REGISTRY_URI=http://mlflow:5000
MLFLOW_EXPERIMENT_NAME=rag-chatbot
METRICS_ENABLED=true
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=admin
```

Monitoring endpoints after startup:

- Grafana: `http://localhost:3001`
- Prometheus: `http://localhost:9090`
- Prometheus targets: `http://localhost:9090/targets`
- MLflow UI: `http://localhost:5000`
- Backend metrics: `http://localhost:8000/metrics`

Default Grafana login:

- username: `admin`
- password: `admin`

Grafana includes a starter `RAG Observability` dashboard for:

- HTTP request rate and p95 latency
- average query latency by route
- estimated query cost by model
- experiment event volume
- provider publish success and failure

### 4. Live updates while developing

- frontend changes update through the mounted `./frontend` volume
- backend changes reload through `uvicorn --reload`
- if you change dependencies or either Dockerfile, rebuild with:

```bash
docker compose up --build
```

## Retrieval Backends (Best Practices)

The RAG system supports three retrieval modes controlled by `RETRIEVAL_BACKEND`:

### 1. PgVector (Default)

- **Mode**: `RETRIEVAL_BACKEND=pgvector`
- **Behavior**: Pure PostgreSQL vector search using `pgvector` extension.
- **When to use**: Development, testing, or when Azure AI Search is not available.
- **Pros**: No external dependencies, lowest latency, fully local.
- **Cons**: No advanced ranking; similarity scoring only.

### 2. Azure AI Search (Strict)

- **Mode**: `RETRIEVAL_BACKEND=azure_ai_search`
- **Behavior**: All retrieval goes through Azure AI Search; no pgvector fallback.
- **When to use**: Production with strict Azure-only requirements.
- **Pros**: Advanced ranking, BM25 + vector fusion, enterprise scale.
- **Cons**: Requires valid Azure credentials at startup; fails hard if Azure is unavailable.
- **Configuration**:
  - `AZURE_AI_SEARCH_ENDPOINT` (required, e.g., `https://your-service.search.windows.net`)
  - `AZURE_AI_SEARCH_API_KEY` (required)
  - `AZURE_AI_SEARCH_INDEX_NAME` (required, e.g., `law-search-index`)

### 3. Hybrid (Recommended for Production)

- **Mode**: `RETRIEVAL_BACKEND=hybrid`
- **Behavior**: Merges results from both pgvector and Azure AI Search; falls back gracefully if Azure fails.
- **When to use**: Production deployments where resilience and quality matter most.
- **Pros**:
  - Best relevance through ranking fusion
  - Automatic fallback to pgvector if Azure Search is unreachable
  - App continues serving (with degraded quality) if Azure AI Search is down
  - Smooth migration path (pgvector always works, Azure added incrementally)
- **Cons**: Slightly higher indexing latency (writes to both backends); requires Azure credentials but doesn't crash if they're wrong.
- **Best practices**:
  - Always populate `AZURE_AI_SEARCH_ENDPOINT`, `AZURE_AI_SEARCH_API_KEY`, and `AZURE_AI_SEARCH_INDEX_NAME` in `.env`.
  - If Azure credentials are missing or invalid, the app boots with a **warning** and uses pgvector only.
  - Startup logs will indicate if Azure Search validation succeeded or fell back to pgvector.
  - Monitor backend logs for `"Azure AI Search validation failed"` messages to detect credential issues.
  - Once Azure Search is healthy, restart the backend to re-validate and activate Azure in hybrid queries.

### Configuration Example (Hybrid with Fallback)

```env
LLM_PROVIDER=azure_openai
RETRIEVAL_BACKEND=hybrid

AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-instance.openai.azure.com/
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-5.4
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large

# Azure AI Search (optional in hybrid; falls back to pgvector if missing or unreachable)
AZURE_AI_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_AI_SEARCH_API_KEY=your_search_key
AZURE_AI_SEARCH_INDEX_NAME=law-search-index
```

### Switching Modes

To switch retrieval backends without code changes:

1. Update `.env` with `RETRIEVAL_BACKEND=<mode>`.
2. Restart the backend: `docker compose restart backend` or `docker compose up --build backend`.
3. Check startup logs for validation messages.

**Migration Path**:

- Start with `pgvector` to validate core functionality.
- Move to `hybrid` once Azure Search is configured and tested.
- Switch to `azure_ai_search` only if you need strict Azure-only enforcement for compliance.

## Local Non-Docker Setup

### Backend

Create `backend/.env`:

```env
GOOGLE_API_KEY=your_google_api_key
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag_chatbot
```

Then run:

```bash
pip install -r backend/requirements.txt
cd backend
uvicorn app.main:app --reload
```

### Frontend

Set:

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

Then run:

```bash
cd frontend
npm install
npm run dev
```

## API Overview

Important endpoints:

- `GET /health`
- `GET /api/v1/documents`
- `POST /api/v1/documents/upload`
- `POST /api/v1/documents/{document_id}/reindex`
- `POST /api/v1/chat/query`
- `GET /api/v1/chat/conversations/{conversation_id}`
- `GET /api/v1/chat/conversations/{conversation_id}/export/{format}`

Chat requests can be scoped to a specific document with `document_id`, or run across all ready documents when no document is provided.

The chat response now also includes memory-related fields such as:

- short-term memory snippets
- long-term semantic memory hits
- memory actions from `remember` / `forget` commands
- knowledge-graph topic summaries

## Notes

- uploaded PDFs are stored in `backend/data/uploads/`
- exports are written to `backend/data/exports/`
- the backend creates tables and the `vector` extension on startup
- additive compatibility migrations run on startup for recently introduced columns on existing tables
- `pgvector` is the active vector store for the current app
- conversation memory, semantic summaries, graph nodes, and document access history are persisted in Postgres
- the legacy notebook and `legacy_streamlit_app.py` remain in the repo as prototype history

## Optional Dependencies

Some orchestration features depend on optional libraries:

- `langgraph` for graph-based orchestration and checkpoint execution
- `beautifulsoup4` for lightweight web search parsing
- `matplotlib` for chart generation
- `python-docx` for DOCX export
- `reportlab` for PDF export

Optional memory-provider environment variables:

- `MEM0_API_KEY`
- `MEM0_BASE_URL`
- `ZEP_API_KEY`
- `ZEP_BASE_URL`

If one of these is missing, the backend falls back to a simpler path where possible.

## Legacy Prototype

The original Streamlit prototype is intentionally retained as `legacy_streamlit_app.py` for reference/testing of the early single-PDF flow. It is not part of the production FastAPI + Next.js deployment path.
