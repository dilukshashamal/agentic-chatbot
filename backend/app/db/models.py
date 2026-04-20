from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


VECTOR_DIMENSIONS = get_settings().embedding_dimensions


class DocumentRecord(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_name: Mapped[str] = mapped_column(String(255))
    storage_path: Mapped[str] = mapped_column(String(1024))
    status: Mapped[str] = mapped_column(String(32), default="uploaded")
    error_message: Mapped[str | None] = mapped_column(Text(), nullable=True)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    chunks: Mapped[list["DocumentChunkRecord"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class DocumentChunkRecord(Base):
    __tablename__ = "document_chunks"
    __table_args__ = (UniqueConstraint("document_id", "chunk_index", name="uq_document_chunk_index"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    content: Mapped[str] = mapped_column(Text())
    embedding: Mapped[list[float]] = mapped_column(Vector(VECTOR_DIMENSIONS))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    document: Mapped[DocumentRecord] = relationship(back_populates="chunks")


class ConversationRecord(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    memory_summary: Mapped[str | None] = mapped_column(Text(), nullable=True)
    user_preferences: Mapped[dict] = mapped_column(JSON, default=dict)
    interaction_patterns: Mapped[dict] = mapped_column(JSON, default=dict)
    query_refinement_history: Mapped[list] = mapped_column(JSON, default=list)
    custom_instructions: Mapped[str | None] = mapped_column(Text(), nullable=True)
    active_document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    turns: Mapped[list["ConversationTurnRecord"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    checkpoints: Mapped[list["ConversationCheckpointRecord"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    memories: Mapped[list["MemoryRecord"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    knowledge_nodes: Mapped[list["KnowledgeGraphNodeRecord"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    document_accesses: Mapped[list["DocumentAccessRecord"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class ConversationTurnRecord(Base):
    __tablename__ = "conversation_turns"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        index=True,
    )
    query: Mapped[str] = mapped_column(Text())
    answer: Mapped[str] = mapped_column(Text())
    route: Mapped[str | None] = mapped_column(String(64), nullable=True)
    grounded: Mapped[bool] = mapped_column(Boolean, default=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    response_payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    conversation: Mapped[ConversationRecord] = relationship(back_populates="turns")
    checkpoints: Mapped[list["ConversationCheckpointRecord"]] = relationship(
        back_populates="turn",
        passive_deletes=True,
    )


class ConversationCheckpointRecord(Base):
    __tablename__ = "conversation_checkpoints"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        index=True,
    )
    turn_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversation_turns.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    node_name: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="completed")
    state_payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    conversation: Mapped[ConversationRecord] = relationship(back_populates="checkpoints")
    turn: Mapped[ConversationTurnRecord | None] = relationship(back_populates="checkpoints")


class MemoryRecord(Base):
    __tablename__ = "memories"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        index=True,
    )
    turn_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversation_turns.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    memory_type: Mapped[str] = mapped_column(String(64), index=True)
    scope: Mapped[str] = mapped_column(String(32), default="conversation")
    content: Mapped[str] = mapped_column(Text())
    summary: Mapped[str | None] = mapped_column(Text(), nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(VECTOR_DIMENSIONS), nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    last_accessed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    conversation: Mapped[ConversationRecord] = relationship(back_populates="memories")


class KnowledgeGraphNodeRecord(Base):
    __tablename__ = "knowledge_graph_nodes"
    __table_args__ = (UniqueConstraint("conversation_id", "node_type", "label", name="uq_knowledge_node"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        index=True,
    )
    node_type: Mapped[str] = mapped_column(String(32), index=True)
    label: Mapped[str] = mapped_column(String(255), index=True)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    conversation: Mapped[ConversationRecord] = relationship(back_populates="knowledge_nodes")


class KnowledgeGraphEdgeRecord(Base):
    __tablename__ = "knowledge_graph_edges"
    __table_args__ = (
        UniqueConstraint("conversation_id", "source_node_id", "target_node_id", "relation", name="uq_knowledge_edge"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        index=True,
    )
    source_node_id: Mapped[int] = mapped_column(ForeignKey("knowledge_graph_nodes.id", ondelete="CASCADE"), index=True)
    target_node_id: Mapped[int] = mapped_column(ForeignKey("knowledge_graph_nodes.id", ondelete="CASCADE"), index=True)
    relation: Mapped[str] = mapped_column(String(64), default="related_to")
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class DocumentAccessRecord(Base):
    __tablename__ = "document_accesses"
    __table_args__ = (UniqueConstraint("conversation_id", "document_id", name="uq_document_access"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        index=True,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        index=True,
    )
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    last_accessed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    conversation: Mapped[ConversationRecord] = relationship(back_populates="document_accesses")


class ModelRegistryRecord(Base):
    __tablename__ = "model_registry"
    __table_args__ = (UniqueConstraint("model_kind", "semantic_version", name="uq_model_registry_version"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_kind: Mapped[str] = mapped_column(String(32), index=True)
    provider: Mapped[str] = mapped_column(String(32), default="local")
    model_name: Mapped[str] = mapped_column(String(255))
    semantic_version: Mapped[str] = mapped_column(String(32), index=True)
    external_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    stage: Mapped[str] = mapped_column(String(32), default="active", index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    is_shadow: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    checkpoint_uri: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class RetrievalConfigRecord(Base):
    __tablename__ = "retrieval_configs"
    __table_args__ = (UniqueConstraint("name", "semantic_version", name="uq_retrieval_config_version"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), default="default")
    semantic_version: Mapped[str] = mapped_column(String(32), index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class PromptTemplateRecord(Base):
    __tablename__ = "prompt_templates"
    __table_args__ = (UniqueConstraint("name", "semantic_version", name="uq_prompt_template_version"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), default="default")
    template_type: Mapped[str] = mapped_column(String(64), default="chat")
    semantic_version: Mapped[str] = mapped_column(String(32), index=True)
    template_text: Mapped[str | None] = mapped_column(Text(), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class FeatureFlagRecord(Base):
    __tablename__ = "feature_flags"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    rollout_percent: Mapped[int] = mapped_column(Integer, default=100)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class ExperimentRunRecord(Base):
    __tablename__ = "experiment_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    experiment_type: Mapped[str] = mapped_column(String(64), index=True)
    experiment_name: Mapped[str] = mapped_column(String(255), index=True)
    pipeline_version: Mapped[str] = mapped_column(String(32), index=True)
    assignment_bucket: Mapped[str | None] = mapped_column(String(32), nullable=True)
    query_type: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(32), default="completed", index=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    prompt_template_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    retrieval_config_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    chat_model_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    embedding_model_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    parameters_json: Mapped[dict] = mapped_column(JSON, default=dict)
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    costs_json: Mapped[dict] = mapped_column(JSON, default=dict)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class ShadowEvaluationRecord(Base):
    __tablename__ = "shadow_evaluations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    experiment_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("experiment_runs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    candidate_chat_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    candidate_embedding_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    candidate_retrieval_config_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    candidate_prompt_template_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    assignment_bucket: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="logged", index=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
