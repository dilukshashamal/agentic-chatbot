from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


ExportFormat = Literal["json", "pdf", "docx"]


class ChatRequest(BaseModel):
    conversation_id: UUID | None = None
    document_id: UUID | None = None
    query: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=4, ge=2, le=8)
    include_sources: bool = True
    export_formats: list[ExportFormat] = Field(default_factory=list, max_length=3)
    user_preferences: dict[str, str] = Field(default_factory=dict)

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("Query cannot be empty.")
        return cleaned


class Citation(BaseModel):
    chunk_id: str
    source: str
    page: int | None = None
    score: float = Field(ge=0.0, le=1.0)
    excerpt: str = Field(min_length=1, max_length=400)


class ChatResponse(BaseModel):
    trace_id: str | None = None
    conversation_id: UUID
    answer: str
    grounded: bool
    confidence: float = Field(ge=0.0, le=1.0)
    answer_mode: Literal["grounded", "insufficient_context"]
    citations: list[Citation] = Field(default_factory=list)
    retrieved_chunks: int = Field(ge=0)
    system_notes: list[str] = Field(default_factory=list)
    route: str = "document_grounded"
    memory_summary: str | None = None
    short_term_memory: list[str] = Field(default_factory=list)
    long_term_memory: list["MemoryHit"] = Field(default_factory=list)
    memory_actions: list["MemoryAction"] = Field(default_factory=list)
    knowledge_graph_topics: list[str] = Field(default_factory=list)
    agent_trace: list["AgentTrace"] = Field(default_factory=list)
    exports: list["ExportArtifact"] = Field(default_factory=list)


class LLMAnswer(BaseModel):
    answer: str = Field(min_length=1, max_length=2000)
    grounded: bool
    confidence: float = Field(ge=0.0, le=1.0)
    cited_chunk_ids: list[str] = Field(default_factory=list, max_length=4)
    missing_information: list[str] = Field(default_factory=list, max_length=5)


class BuildIndexResponse(BaseModel):
    message: str
    page_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    document_id: UUID


class SystemStatus(BaseModel):
    status: Literal["ok"]
    api_name: str
    document_count: int = Field(ge=0)
    ready_document_count: int = Field(ge=0)
    chat_model: str
    embedding_model: str
    orchestration_enabled: bool = True
    conversation_count: int = Field(default=0, ge=0)
    memory_provider: str = "local"
    pipeline_version: str = "1.0.0"
    retrieval_config_version: str = "1.0.0"
    prompt_template_version: str = "1.0.0"
    model_registry_provider: str = "local"
    shadow_mode_enabled: bool = False


class DocumentSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    file_name: str
    status: Literal["uploaded", "processing", "ready", "failed"]
    page_count: int | None = None
    chunk_count: int | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class DocumentUploadResponse(BaseModel):
    message: str
    document: DocumentSummary


class AgentTrace(BaseModel):
    agent: str
    status: Literal["completed", "skipped", "fallback", "failed"]
    summary: str
    retries: int = Field(default=0, ge=0)


class ExportArtifact(BaseModel):
    format: ExportFormat
    path: str
    created_at: datetime


class MemoryHit(BaseModel):
    id: str
    memory_type: str
    content: str
    score: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryAction(BaseModel):
    action: Literal["remembered", "forgotten", "updated", "ignored"]
    target: str
    detail: str


class ConversationTurnSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    query: str
    answer: str
    route: str | None = None
    grounded: bool
    confidence: float
    created_at: datetime
    response_payload: dict[str, Any] = Field(default_factory=dict)


class ConversationDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str | None = None
    memory_summary: str | None = None
    user_preferences: dict[str, Any] = Field(default_factory=dict)
    interaction_patterns: dict[str, Any] = Field(default_factory=dict)
    query_refinement_history: list[dict[str, Any]] = Field(default_factory=list)
    custom_instructions: str | None = None
    active_document_id: UUID | None = None
    created_at: datetime
    updated_at: datetime
    turns: list[ConversationTurnSummary] = Field(default_factory=list)


class ModelRegistryEntry(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    model_kind: str
    provider: str
    model_name: str
    semantic_version: str
    external_version: str | None = None
    stage: str
    is_active: bool
    is_shadow: bool
    checkpoint_uri: str | None = None
    metadata_json: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class RetrievalConfigVersion(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    semantic_version: str
    is_active: bool
    metadata_json: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class PromptTemplateVersion(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    template_type: str
    semantic_version: str
    template_text: str | None = None
    is_active: bool
    metadata_json: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class FeatureFlagSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    enabled: bool
    rollout_percent: int
    metadata_json: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class ExperimentRunSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    conversation_id: UUID | None = None
    experiment_type: str
    experiment_name: str
    pipeline_version: str
    assignment_bucket: str | None = None
    query_type: str | None = None
    status: str
    latency_ms: float | None = None
    prompt_template_version: str | None = None
    retrieval_config_version: str | None = None
    chat_model_version: str | None = None
    embedding_model_version: str | None = None
    parameters_json: dict[str, Any] = Field(default_factory=dict)
    metrics_json: dict[str, Any] = Field(default_factory=dict)
    costs_json: dict[str, Any] = Field(default_factory=dict)
    metadata_json: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class ShadowEvaluationSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    experiment_run_id: UUID | None = None
    conversation_id: UUID | None = None
    candidate_chat_model: str | None = None
    candidate_embedding_model: str | None = None
    candidate_retrieval_config_version: str | None = None
    candidate_prompt_template_version: str | None = None
    assignment_bucket: str | None = None
    status: str
    latency_ms: float | None = None
    metrics_json: dict[str, Any] = Field(default_factory=dict)
    metadata_json: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ModelManagementStatus(BaseModel):
    pipeline_version: str
    registry_provider: str
    active_chat_model: str
    active_embedding_model: str
    retrieval_config_version: str
    prompt_template_version: str
    ab_test_enabled: bool
    ab_test_rollout_percent: int
    shadow_mode_enabled: bool
    shadow_sampling_percent: int
    feature_flags: list[FeatureFlagSummary] = Field(default_factory=list)
    registry_entries: list[ModelRegistryEntry] = Field(default_factory=list)


ChatResponse.model_rebuild()
