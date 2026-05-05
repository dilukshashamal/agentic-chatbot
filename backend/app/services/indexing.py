from dataclasses import dataclass
from pathlib import Path
import time
from uuid import UUID

from langchain_core.documents import Document as LCDocument
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from sqlalchemy import delete
from sqlalchemy.orm import Session
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

from app.core.config import Settings
from app.db.models import DocumentChunkRecord, DocumentRecord
from app.services.azure_search import AzureAISearchService, azure_ai_search_enabled
from app.services.documents import normalize_text, text_quality_score
from app.services.metrics import observe_index
from app.services.model_management import ModelManagementService


@dataclass(frozen=True)
class BuildIndexResult:
    document_id: UUID
    page_count: int
    chunk_count: int


def _get_embeddings(settings: Settings, model_name: str | None = None) -> GoogleGenerativeAIEmbeddings | AzureOpenAIEmbeddings:
    if settings.llm_provider == "azure_openai":
        if not settings.azure_openai_api_key:
            raise RuntimeError("AZURE_OPENAI_API_KEY is required to build or query the RAG index.")
        if not settings.azure_openai_endpoint:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT is required to build or query the RAG index.")

        deployment = model_name or settings.azure_openai_embedding_deployment or settings.embedding_model
        return AzureOpenAIEmbeddings(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_deployment=deployment,
            model=deployment,
            dimensions=settings.embedding_dimensions,
        )

    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is required to build or query the RAG index.")

    return GoogleGenerativeAIEmbeddings(
        model=model_name or settings.embedding_model,
        google_api_key=settings.google_api_key,
    )


def load_pdf_documents(document: DocumentRecord) -> tuple[list, int]:
    pdf_path = document.storage_path
    if not pdf_path:
        raise FileNotFoundError("Document has no stored PDF path.")

    elements = partition_pdf(
        filename=pdf_path,
        strategy="auto",
    )

    page_count = 1
    for el in elements:
        if hasattr(el, "metadata") and hasattr(el.metadata, "page_number") and el.metadata.page_number:
            page_count = max(page_count, el.metadata.page_number)

    return elements, page_count


_analyzer = None
_anonymizer = None

def redact_pii(text: str) -> str:
    global _analyzer, _anonymizer
    if _analyzer is None:
        _analyzer = AnalyzerEngine()
    if _anonymizer is None:
        _anonymizer = AnonymizerEngine()
        
    # Analyze text for specific PII entities
    results = _analyzer.analyze(
        text=text, 
        entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "US_SSN"], 
        language='en'
    )
    
    # Anonymize findings
    anonymized_result = _anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized_result.text


def split_documents(settings: Settings, elements: list, filename: str) -> list[LCDocument]:
    unstructured_chunks = chunk_by_title(
        elements,
        max_characters=settings.chunk_size,
        overlap=settings.chunk_overlap,
        combine_text_under_n_chars=300
    )

    enriched_chunks: list[LCDocument] = []

    for index, chunk in enumerate(unstructured_chunks, start=1):
        cleaned_content = normalize_text(str(chunk))
        if text_quality_score(cleaned_content) < 0.18:
            continue

        # Redact PII before creating the document chunk
        redacted_content = redact_pii(cleaned_content)

        page_number = None
        if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "page_number"):
            page_number = chunk.metadata.page_number

        enriched_chunks.append(
            LCDocument(
                page_content=redacted_content,
                metadata={
                    "source": filename,
                    "page_number": page_number,
                    "chunk_id": f"chunk-{index:04d}",
                },
            )
        )

    return enriched_chunks


def build_index(settings: Settings, session: Session, document: DocumentRecord) -> BuildIndexResult:
    started_at = time.perf_counter()
    try:
        elements, page_count = load_pdf_documents(document)
    except FileNotFoundError as exc:
        raise RuntimeError(f"PDF file not found at {document.storage_path}: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to load PDF: {exc}") from exc
    
    try:
        chunks = split_documents(settings, elements, Path(document.storage_path).name)
    except Exception as exc:
        raise RuntimeError(f"Failed to split documents into chunks: {exc}") from exc
    
    if not chunks:
        raise RuntimeError("The uploaded PDF did not produce any indexable text chunks. This may mean the PDF has no readable text or all content was filtered by quality checks.")

    try:
        embeddings = _get_embeddings(settings).embed_documents([chunk.page_content for chunk in chunks])
    except Exception as exc:
        raise RuntimeError(f"Failed to generate embeddings (check LLM provider credentials): {exc}") from exc

    if azure_ai_search_enabled(settings):
        try:
            AzureAISearchService(settings).upsert_chunks(
                document_id=document.id,
                source_name=document.file_name,
                chunks=chunks,
                embeddings=embeddings,
                allowed_groups=document.allowed_groups,
            )
        except Exception as exc:
            # In hybrid mode, log warning but continue; in azure-only mode, fail hard
            if settings.retrieval_backend == "azure_ai_search":
                raise RuntimeError(f"Failed to upsert chunks to Azure AI Search (strict mode): {exc}") from exc
            # Hybrid mode: continue with pgvector
            import logging
            logging.getLogger(__name__).warning(f"Azure AI Search upsert failed (hybrid mode, continuing with pgvector): {exc}")

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
    document.page_count = page_count
    document.chunk_count = len(chunks)
    session.add(document)
    session.commit()

    ModelManagementService(settings, session).log_index_experiment(
        document_id=document.id,
        page_count=page_count,
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
        page_count=page_count,
        chunk_count=len(chunks),
    )
