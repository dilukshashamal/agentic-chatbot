# RAG System

This project is now structured as a multi-document RAG app with:

- `FastAPI` backend
- `Next.js` frontend
- `PostgreSQL + pgvector` for vector search
- background PDF indexing after upload

The old notebook and Streamlit files are still present as prototype history, but the active app lives in `backend/` and `frontend/`.

## Architecture

```text
rag_chatbot/
├─ backend/
│  ├─ app/
│  │  ├─ api/routes/
│  │  ├─ core/
│  │  ├─ db/
│  │  ├─ models/
│  │  └─ services/
│  ├─ data/
│  ├─ .env.example
│  ├─ Dockerfile
│  └─ requirements.txt
├─ frontend/
│  ├─ app/
│  ├─ components/
│  ├─ lib/
│  ├─ .env.example
│  ├─ Dockerfile
│  └─ package.json
├─ docker-compose.yml
├─ .env.example
├─ rag_notebook.ipynb
├─ app.py
└─ README.md
```

## Docker Setup

The repository now includes a local Docker stack with:

- `db`: PostgreSQL with the `pgvector` extension
- `backend`: FastAPI app
- `frontend`: Next.js app

### 1. Create root `.env`

Create a root `.env` from `.env.example`:

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

Apps will be available at:

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`
- Postgres: `localhost:5432`

The backend connects to Postgres through the Compose service name `db`.

## Local Non-Docker Setup

If you want to run the backend directly on your machine instead of inside Docker, create `backend/.env` from `backend/.env.example`:

```env
GOOGLE_API_KEY=your_google_api_key
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag_chatbot
```

Then:

```bash
pip install -r backend/requirements.txt
cd backend
uvicorn app.main:app --reload
```

For the frontend:

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

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

Chat requests are now document-scoped, so each question must include a `document_id`.

## Notes

- Uploaded PDFs are stored under `backend/data/uploads/`.
- The backend creates the `vector` extension and tables on startup.
- `pgvector` is now the source of truth for chunk embeddings instead of local FAISS folders.

## Suggested Next Improvements

- Add Alembic migrations instead of relying on startup table creation.
- Add a dedicated worker queue for indexing jobs.
- Add automated integration tests for upload, indexing, and chat flows.
- Retire the legacy prototype files once the new stack fully replaces them.
