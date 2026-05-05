from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.models import DocumentChunkRecord, DocumentRecord
from app.services.azure_search import AzureAISearchService, azure_ai_search_enabled
from app.services.documents import RetrievedChunk, text_quality_score, tokenize
from app.services.indexing import _get_embeddings


@dataclass
class _Candidate:
    chunk_id: str
    content: str
    page_number: int | None
    source: str
    vector_score: float = 0.0
    bm25_score: float = 0.0
    overlap_score: float = 0.0


class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        top_k: int,
        document_id: UUID | None = None,
        *,
        embedding_model: str | None = None,
        retrieval_overrides: dict[str, float | int] | None = None,
        allowed_groups: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        ...


class PgVectorRetriever:
    def __init__(self, settings: Settings, session: Session) -> None:
        self.settings = settings
        self.session = session

    @staticmethod
    def _keyword_overlap(query_tokens: list[str], doc_tokens: list[str]) -> float:
        if not query_tokens or not doc_tokens:
            return 0.0

        query_set = set(query_tokens)
        doc_set = set(doc_tokens)
        return len(query_set & doc_set) / max(len(query_set), 1)

    @staticmethod
    def _chunk_id(document_id: UUID, chunk_index: int) -> str:
        return f"doc-{str(document_id)[:8]}-chunk-{chunk_index:04d}"

    def retrieve(
        self,
        query: str,
        top_k: int,
        document_id: UUID | None = None,
        *,
        embedding_model: str | None = None,
        retrieval_overrides: dict[str, float | int] | None = None,
        allowed_groups: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        overrides = retrieval_overrides or {}
        query_tokens = tokenize(query)
        query_embedding = _get_embeddings(self.settings, model_name=embedding_model).embed_query(query)
        distance = DocumentChunkRecord.embedding.cosine_distance(query_embedding)

        statement = (
            select(DocumentChunkRecord, DocumentRecord.file_name, distance.label("distance"))
            .join(DocumentRecord, DocumentRecord.id == DocumentChunkRecord.document_id)
            .where(DocumentRecord.status == "ready")
            .order_by(distance.asc())
            .limit(int(overrides.get("retriever_fetch_k", self.settings.retriever_fetch_k)))
        )
        if document_id is not None:
            statement = statement.where(DocumentChunkRecord.document_id == document_id)

        if allowed_groups is not None:
            from sqlalchemy.dialects.postgresql import ARRAY
            from sqlalchemy import cast, String, or_, func
            
            if not allowed_groups:
                statement = statement.where(func.jsonb_array_length(DocumentRecord.allowed_groups) == 0)
            else:
                statement = statement.where(
                    or_(
                        func.jsonb_array_length(DocumentRecord.allowed_groups) == 0,
                        DocumentRecord.allowed_groups.op('?|')(cast(allowed_groups, ARRAY(String)))
                    )
                )

        rows = self.session.execute(statement).all()

        retrieved: list[RetrievedChunk] = []
        for chunk, document_name, raw_distance in rows:
            vector_score = max(0.0, min(1.0, 1.0 - max(float(raw_distance), 0.0)))
            overlap_score = self._keyword_overlap(query_tokens, tokenize(chunk.content))
            quality_score = max(text_quality_score(chunk.content), 0.2)
            combined_score = (
                float(overrides.get("vector_weight", self.settings.vector_weight)) * vector_score
                + float(overrides.get("overlap_weight", self.settings.overlap_weight)) * overlap_score
            ) * quality_score
            retrieved.append(
                RetrievedChunk(
                    chunk_id=self._chunk_id(chunk.document_id, chunk.chunk_index),
                    document_id=chunk.document_id,
                    content=chunk.content,
                    page_number=chunk.page_number,
                    source=document_name,
                    score=round(min(combined_score, 1.0), 4),
                    vector_score=round(vector_score, 4),
                    bm25_score=0.0,
                    overlap_score=round(min(overlap_score, 1.0), 4),
                )
            )

        retrieved.sort(key=lambda chunk: chunk.score, reverse=True)
        return retrieved[:top_k]


class AzureSearchRetriever:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.search = AzureAISearchService(settings)

    @staticmethod
    def _keyword_overlap(query_tokens: list[str], doc_tokens: list[str]) -> float:
        if not query_tokens or not doc_tokens:
            return 0.0

        query_set = set(query_tokens)
        doc_set = set(doc_tokens)
        return len(query_set & doc_set) / max(len(query_set), 1)

    @staticmethod
    def _normalize_raw_score(raw_score: float) -> float:
        # Azure scores vary by ranking profile and query mode; this keeps values bounded for blending.
        if raw_score <= 0.0:
            return 0.0
        return max(0.0, min(1.0, raw_score / (raw_score + 1.0)))

    def retrieve(
        self,
        query: str,
        top_k: int,
        document_id: UUID | None = None,
        *,
        embedding_model: str | None = None,
        retrieval_overrides: dict[str, float | int] | None = None,
        allowed_groups: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        overrides = retrieval_overrides or {}
        fetch_k = int(overrides.get("retriever_fetch_k", self.settings.retriever_fetch_k))
        query_tokens = tokenize(query)
        query_embedding = _get_embeddings(self.settings, model_name=embedding_model).embed_query(query)
        rows = self.search.search_chunks(
            query=query,
            query_embedding=query_embedding,
            fetch_k=fetch_k,
            document_id=document_id,
            allowed_groups=allowed_groups,
        )

        retrieved: list[RetrievedChunk] = []
        for row in rows:
            content = str(row.get("content") or "")
            if not content:
                continue

            raw_score = float(row.get("raw_score") or 0.0)
            vector_score = self._normalize_raw_score(raw_score)
            overlap_score = self._keyword_overlap(query_tokens, tokenize(content))
            quality_score = max(text_quality_score(content), 0.2)
            combined_score = (
                float(overrides.get("vector_weight", self.settings.vector_weight)) * vector_score
                + float(overrides.get("overlap_weight", self.settings.overlap_weight)) * overlap_score
            ) * quality_score

            document_uuid = UUID(str(row.get("document_id")))
            retrieved.append(
                RetrievedChunk(
                    chunk_id=str(row.get("chunk_id") or ""),
                    document_id=document_uuid,
                    content=content,
                    page_number=row.get("page_number"),
                    source=str(row.get("source") or "Unknown"),
                    score=round(min(combined_score, 1.0), 4),
                    vector_score=round(vector_score, 4),
                    bm25_score=0.0,
                    overlap_score=round(min(overlap_score, 1.0), 4),
                )
            )

        retrieved.sort(key=lambda chunk: chunk.score, reverse=True)
        return retrieved[:top_k]


class HybridRetriever:
    def __init__(self, settings: Settings, session: Session) -> None:
        self.pg = PgVectorRetriever(settings, session)
        self.azure = AzureSearchRetriever(settings)

    def retrieve(
        self,
        query: str,
        top_k: int,
        document_id: UUID | None = None,
        *,
        embedding_model: str | None = None,
        retrieval_overrides: dict[str, float | int] | None = None,
        allowed_groups: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        pg_chunks = self.pg.retrieve(
            query=query,
            top_k=max(top_k, 1),
            document_id=document_id,
            embedding_model=embedding_model,
            retrieval_overrides=retrieval_overrides,
            allowed_groups=allowed_groups,
        )
        az_chunks = self.azure.retrieve(
            query=query,
            top_k=max(top_k, 1),
            document_id=document_id,
            embedding_model=embedding_model,
            retrieval_overrides=retrieval_overrides,
            allowed_groups=allowed_groups,
        )

        merged: dict[str, RetrievedChunk] = {}
        for chunk in [*pg_chunks, *az_chunks]:
            existing = merged.get(chunk.chunk_id)
            if existing is None:
                merged[chunk.chunk_id] = chunk
                continue

            merged[chunk.chunk_id] = RetrievedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                content=chunk.content if len(chunk.content) > len(existing.content) else existing.content,
                page_number=chunk.page_number if chunk.page_number is not None else existing.page_number,
                source=chunk.source or existing.source,
                score=round(max(existing.score, chunk.score), 4),
                vector_score=round(max(existing.vector_score, chunk.vector_score), 4),
                bm25_score=0.0,
                overlap_score=round(max(existing.overlap_score, chunk.overlap_score), 4),
            )

        ranked = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        return ranked[:top_k]


def build_retriever(settings: Settings, session: Session) -> Retriever:
    backend = settings.retrieval_backend
    if backend == "pgvector":
        return PgVectorRetriever(settings, session)
    if backend == "azure_ai_search":
        return AzureSearchRetriever(settings)
    if backend == "hybrid":
        return HybridRetriever(settings, session)

    # Safety fallback for unexpected values from external env edits.
    if azure_ai_search_enabled(settings):
        return HybridRetriever(settings, session)
    return PgVectorRetriever(settings, session)
