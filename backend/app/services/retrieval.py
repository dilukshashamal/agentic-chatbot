from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.models import DocumentChunkRecord, DocumentRecord
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
