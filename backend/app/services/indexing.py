from dataclasses import dataclass
from pathlib import Path
import time
from uuid import UUID

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document as LCDocument
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.models import DocumentChunkRecord, DocumentRecord
from app.services.documents import normalize_text, text_quality_score
from app.services.metrics import observe_index
from app.services.model_management import ModelManagementService


@dataclass(frozen=True)
class BuildIndexResult:
    document_id: UUID
    page_count: int
    chunk_count: int


def _get_embeddings(settings: Settings, model_name: str | None = None) -> GoogleGenerativeAIEmbeddings:
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is required to build or query the RAG index.")

    return GoogleGenerativeAIEmbeddings(
        model=model_name or settings.embedding_model,
        google_api_key=settings.google_api_key,
    )


def load_pdf_documents(document: DocumentRecord) -> list[LCDocument]:
    pdf_path = document.storage_path
    if not pdf_path:
        raise FileNotFoundError("Document has no stored PDF path.")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    cleaned_documents: list[LCDocument] = []
    for document in documents:
        cleaned = normalize_text(document.page_content)
        if not cleaned:
            continue

        page = document.metadata.get("page")
        page_number = page + 1 if isinstance(page, int) else None
        cleaned_documents.append(
            LCDocument(
                page_content=cleaned,
                metadata={
                    **document.metadata,
                    "source": Path(pdf_path).name,
                    "page_number": page_number,
                },
            )
        )

    return cleaned_documents


def split_documents(settings: Settings, documents: list[LCDocument]) -> list[LCDocument]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    chunks = splitter.split_documents(documents)
    enriched_chunks: list[LCDocument] = []

    for index, chunk in enumerate(chunks, start=1):
        cleaned_content = normalize_text(chunk.page_content)
        if text_quality_score(cleaned_content) < 0.18:
            continue

        enriched_chunks.append(
            LCDocument(
                page_content=cleaned_content,
                metadata={
                    **chunk.metadata,
                    "chunk_id": f"chunk-{index:04d}",
                },
            )
        )

    return enriched_chunks


def build_index(settings: Settings, session: Session, document: DocumentRecord) -> BuildIndexResult:
    started_at = time.perf_counter()
    documents = load_pdf_documents(document)
    chunks = split_documents(settings, documents)
    if not chunks:
        raise RuntimeError("The uploaded PDF did not produce any indexable text chunks.")

    embeddings = _get_embeddings(settings).embed_documents([chunk.page_content for chunk in chunks])

    session.execute(delete(DocumentChunkRecord).where(DocumentChunkRecord.document_id == document.id))

    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True), start=1):
        session.add(
            DocumentChunkRecord(
                document_id=document.id,
                chunk_index=index,
                page_number=chunk.metadata.get("page_number"),
                content=chunk.page_content,
                embedding=embedding,
            )
        )

    document.status = "ready"
    document.error_message = None
    document.page_count = len(documents)
    document.chunk_count = len(chunks)
    session.add(document)
    session.commit()

    ModelManagementService(settings, session).log_index_experiment(
        document_id=document.id,
        page_count=len(documents),
        chunk_count=len(chunks),
        duration_ms=(time.perf_counter() - started_at) * 1000.0,
        metadata_json={
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        },
    )
    observe_index(
        status="completed",
        embedding_model=settings.embedding_model,
        latency_seconds=max(time.perf_counter() - started_at, 0.0),
    )

    return BuildIndexResult(
        document_id=document.id,
        page_count=len(documents),
        chunk_count=len(chunks),
    )
