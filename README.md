# Synkora AI RAG Chatbot

Synkora AI is a multi-document RAG application for uploading PDFs, indexing them into `pgvector`, and chatting with grounded answers through a polished Next.js dashboard.

![Synkora dashboard](images/synkora_dash.png)

## Highlights

- `FastAPI` backend with document upload, indexing, and chat APIs
- `Next.js` frontend for document management and chat
- `PostgreSQL + pgvector` for persistent vector search
- background PDF indexing after upload
- multi-agent orchestration layer for routing, memory, grounding, analysis, and tool use
- conversation persistence with checkpoints for multi-turn workflows
- source citations attached to grounded answers

## Multi-Agent Flow

The backend now supports a supervisor-style orchestration flow built around specialized agents:

- `Router Agent`: chooses the best path for the user request
- `Document Understanding Agent`: summarization, entities, and key insights
- `Analytical Agent`: cross-document comparisons and trend-style reasoning
- `Citation Agent`: grounding validation and source attribution
- `Memory Agent`: conversation context and user preferences
- `Tool Use Agent`: search, calculation, transformation, charting, and export support

When optional libraries are unavailable, the system degrades gracefully to simpler retrieval instead of crashing.

## Architecture

```text
rag_chatbot/
|-- backend/
|   |-- app/
|   |   |-- api/routes/
|   |   |-- core/
|   |   |-- db/
|   |   |-- models/
|   |   `-- services/
|   |-- data/
|   |-- Dockerfile
|   `-- requirements.txt
|-- frontend/
|   |-- app/
|   |-- components/
|   |-- lib/
|   |-- Dockerfile
|   `-- package.json
|-- images/
|-- docker-compose.yml
|-- .env.example
|-- rag_notebook.ipynb
|-- app.py
`-- README.md
```

## Dashboard

The current dashboard supports:

- PDF upload from the sidebar
- document readiness and indexing status
- grounded chat responses with confidence and citation badges
- multi-file retrieval across ready documents
- conversation continuity through backend conversation IDs

## Docker Setup

### 1. Create the root `.env`

Copy `.env.example` to `.env` and set your real Google API key:

```env
GOOGLE_API_KEY=your_google_api_key
POSTGRES_DB=rag_chatbot
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_PORT=5432
```

### 2. Start the stack

From the project root:

```bash
docker compose up --build
```

Services:

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`
- Postgres: `localhost:5432`

### 3. Live updates while developing

- frontend changes update through the mounted `./frontend` volume
- backend changes reload through `uvicorn --reload`
- if you change dependencies or either Dockerfile, rebuild with:

```bash
docker compose up --build
```

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

## Notes

- uploaded PDFs are stored in `backend/data/uploads/`
- exports are written to `backend/data/exports/`
- the backend creates tables and the `vector` extension on startup
- `pgvector` is the active vector store for the current app
- the legacy notebook and `app.py` remain in the repo as prototype history

## Optional Dependencies

Some orchestration features depend on optional libraries:

- `langgraph` for graph-based orchestration and checkpoint execution
- `beautifulsoup4` for lightweight web search parsing
- `matplotlib` for chart generation
- `python-docx` for DOCX export
- `reportlab` for PDF export

If one of these is missing, the backend falls back to a simpler path where possible.

## Suggested Next Improvements

- add Alembic migrations instead of relying on startup table creation
- add automated backend tests for conversation persistence and agent fallbacks
- expose agent traces and export actions more directly in the frontend
- retire the legacy prototype files once the new stack fully replaces them
