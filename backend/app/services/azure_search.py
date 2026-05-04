from __future__ import annotations

from typing import Any
from uuid import UUID

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery
from langchain_core.documents import Document as LCDocument

from app.core.config import Settings


class AzureAISearchService:
    """Small wrapper around Azure AI Search for index bootstrap, upsert, and retrieval."""

    VECTOR_PROFILE_NAME = "vector-profile"
    VECTOR_ALGORITHM_NAME = "hnsw-default"

    FIELD_ID = "id"
    FIELD_DOCUMENT_ID = "document_id"
    FIELD_CHUNK_ID = "chunk_id"
    FIELD_CHUNK_INDEX = "chunk_index"
    FIELD_SOURCE = "source"
    FIELD_PAGE_NUMBER = "page_number"
    FIELD_CONTENT = "content"
    FIELD_CONTENT_VECTOR = "content_vector"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._validate_required_settings()

    def _validate_required_settings(self) -> None:
        if not self.settings.azure_ai_search_endpoint:
            raise RuntimeError("AZURE_AI_SEARCH_ENDPOINT is required when RETRIEVAL_BACKEND uses Azure AI Search.")
        if not self.settings.azure_ai_search_api_key:
            raise RuntimeError("AZURE_AI_SEARCH_API_KEY is required when RETRIEVAL_BACKEND uses Azure AI Search.")
        if not self.settings.azure_ai_search_index_name:
            raise RuntimeError("AZURE_AI_SEARCH_INDEX_NAME is required when RETRIEVAL_BACKEND uses Azure AI Search.")

    def _credential(self) -> AzureKeyCredential:
        return AzureKeyCredential(self.settings.azure_ai_search_api_key or "")

    def _search_client(self) -> SearchClient:
        return SearchClient(
            endpoint=self.settings.azure_ai_search_endpoint or "",
            index_name=self.settings.azure_ai_search_index_name or "",
            credential=self._credential(),
        )

    def _index_client(self) -> SearchIndexClient:
        return SearchIndexClient(
            endpoint=self.settings.azure_ai_search_endpoint or "",
            credential=self._credential(),
        )

    @staticmethod
    def _chunk_compound_id(document_id: UUID, chunk_index: int) -> str:
        return f"{document_id}:{chunk_index:04d}"

    @staticmethod
    def to_public_chunk_id(document_id: UUID, chunk_index: int) -> str:
        return f"doc-{str(document_id)[:8]}-chunk-{chunk_index:04d}"

    def ensure_index_exists(self) -> None:
        index_name = self.settings.azure_ai_search_index_name or ""
        index_client = self._index_client()
        try:
            index_client.get_index(index_name)
            return
        except ResourceNotFoundError:
            pass

        index = SearchIndex(
            name=index_name,
            fields=[
                SimpleField(name=self.FIELD_ID, type=SearchFieldDataType.String, key=True, filterable=True),
                SimpleField(name=self.FIELD_DOCUMENT_ID, type=SearchFieldDataType.String, filterable=True),
                SimpleField(name=self.FIELD_CHUNK_ID, type=SearchFieldDataType.String, filterable=True),
                SimpleField(name=self.FIELD_CHUNK_INDEX, type=SearchFieldDataType.Int32, filterable=True, sortable=True),
                SearchableField(name=self.FIELD_SOURCE, type=SearchFieldDataType.String, filterable=True),
                SimpleField(name=self.FIELD_PAGE_NUMBER, type=SearchFieldDataType.Int32, filterable=True, sortable=True),
                SearchableField(name=self.FIELD_CONTENT, type=SearchFieldDataType.String),
                SearchField(
                    name=self.FIELD_CONTENT_VECTOR,
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=self.settings.embedding_dimensions,
                    vector_search_profile_name=self.VECTOR_PROFILE_NAME,
                ),
            ],
            vector_search=VectorSearch(
                algorithms=[HnswAlgorithmConfiguration(name=self.VECTOR_ALGORITHM_NAME)],
                profiles=[
                    VectorSearchProfile(
                        name=self.VECTOR_PROFILE_NAME,
                        algorithm_configuration_name=self.VECTOR_ALGORITHM_NAME,
                    )
                ],
            ),
        )
        index_client.create_or_update_index(index)

    def delete_document_chunks(self, document_id: UUID) -> None:
        self.ensure_index_exists()
        client = self._search_client()
        filter_value = f"{self.FIELD_DOCUMENT_ID} eq '{document_id}'"
        ids: list[dict[str, str]] = []

        results = client.search(
            search_text="*",
            filter=filter_value,
            select=[self.FIELD_ID],
            top=1000,
        )
        for row in results:
            row_id = row.get(self.FIELD_ID) if isinstance(row, dict) else None
            if row_id:
                ids.append({self.FIELD_ID: str(row_id)})

        if ids:
            client.delete_documents(documents=ids)

    def upsert_chunks(
        self,
        document_id: UUID,
        source_name: str,
        chunks: list[LCDocument],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise RuntimeError("Chunk and embedding counts must match for Azure AI Search upsert.")

        self.ensure_index_exists()
        self.delete_document_chunks(document_id)
        client = self._search_client()
        payload: list[dict[str, Any]] = []

        for chunk_index, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True), start=1):
            payload.append(
                {
                    self.FIELD_ID: self._chunk_compound_id(document_id, chunk_index),
                    self.FIELD_DOCUMENT_ID: str(document_id),
                    self.FIELD_CHUNK_ID: self.to_public_chunk_id(document_id, chunk_index),
                    self.FIELD_CHUNK_INDEX: chunk_index,
                    self.FIELD_SOURCE: source_name,
                    self.FIELD_PAGE_NUMBER: chunk.metadata.get("page_number"),
                    self.FIELD_CONTENT: chunk.page_content,
                    self.FIELD_CONTENT_VECTOR: embedding,
                }
            )

        if payload:
            client.merge_or_upload_documents(documents=payload)

    def search_chunks(
        self,
        query: str,
        query_embedding: list[float],
        *,
        fetch_k: int,
        document_id: UUID | None = None,
    ) -> list[dict[str, Any]]:
        self.ensure_index_exists()
        client = self._search_client()

        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=max(fetch_k, 1),
            fields=self.FIELD_CONTENT_VECTOR,
        )

        filter_value = None
        if document_id is not None:
            filter_value = f"{self.FIELD_DOCUMENT_ID} eq '{document_id}'"

        try:
            results = client.search(
                search_text=query,
                vector_queries=[vector_query],
                filter=filter_value,
                select=[
                    self.FIELD_CHUNK_ID,
                    self.FIELD_DOCUMENT_ID,
                    self.FIELD_CONTENT,
                    self.FIELD_PAGE_NUMBER,
                    self.FIELD_SOURCE,
                ],
                top=max(fetch_k, 1),
            )
        except HttpResponseError as exc:
            raise RuntimeError(f"Azure AI Search query failed: {exc}") from exc

        rows: list[dict[str, Any]] = []
        for row in results:
            if isinstance(row, dict):
                score = float(row.get("@search.score", 0.0) or 0.0)
                rows.append(
                    {
                        "chunk_id": str(row.get(self.FIELD_CHUNK_ID) or ""),
                        "document_id": str(row.get(self.FIELD_DOCUMENT_ID) or ""),
                        "content": str(row.get(self.FIELD_CONTENT) or ""),
                        "page_number": row.get(self.FIELD_PAGE_NUMBER),
                        "source": str(row.get(self.FIELD_SOURCE) or ""),
                        "raw_score": score,
                    }
                )
        return rows


def azure_ai_search_enabled(settings: Settings) -> bool:
    return settings.retrieval_backend in {"azure_ai_search", "hybrid"}


def validate_azure_ai_search_runtime(settings: Settings) -> None:
    if not azure_ai_search_enabled(settings):
        return
    AzureAISearchService(settings).ensure_index_exists()
