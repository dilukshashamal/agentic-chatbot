"""initial schema

Revision ID: 20260502_0001
Revises:
Create Date: 2026-05-02
"""

from alembic import op
from pgvector.sqlalchemy import Vector
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from app.core.config import get_settings

revision = "20260502_0001"
down_revision = None
branch_labels = None
depends_on = None

VECTOR_DIMENSIONS = get_settings().embedding_dimensions


def _created_at_column() -> sa.Column:
    return sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False)


def _updated_at_column() -> sa.Column:
    return sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False)


def _json_column(name: str) -> sa.Column:
    return sa.Column(name, sa.JSON(), nullable=False)


def _uuid_column(name: str, *, nullable: bool = False) -> sa.Column:
    return sa.Column(name, postgresql.UUID(as_uuid=True), nullable=nullable)


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        _uuid_column("id"),
        sa.Column("file_name", sa.String(length=255), nullable=False),
        sa.Column("storage_path", sa.String(length=1024), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("chunk_count", sa.Integer(), nullable=True),
        _created_at_column(),
        _updated_at_column(),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "model_registry",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("model_kind", sa.String(length=32), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("semantic_version", sa.String(length=32), nullable=False),
        sa.Column("external_version", sa.String(length=128), nullable=True),
        sa.Column("stage", sa.String(length=32), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("is_shadow", sa.Boolean(), nullable=False),
        sa.Column("checkpoint_uri", sa.String(length=1024), nullable=True),
        _json_column("metadata_json"),
        _created_at_column(),
        _updated_at_column(),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("model_kind", "semantic_version", name="uq_model_registry_version"),
    )
    op.create_index(op.f("ix_model_registry_model_kind"), "model_registry", ["model_kind"], unique=False)
    op.create_index(op.f("ix_model_registry_semantic_version"), "model_registry", ["semantic_version"], unique=False)
    op.create_index(op.f("ix_model_registry_stage"), "model_registry", ["stage"], unique=False)
    op.create_index(op.f("ix_model_registry_is_active"), "model_registry", ["is_active"], unique=False)
    op.create_index(op.f("ix_model_registry_is_shadow"), "model_registry", ["is_shadow"], unique=False)

    op.create_table(
        "retrieval_configs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("semantic_version", sa.String(length=32), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        _json_column("metadata_json"),
        _created_at_column(),
        _updated_at_column(),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", "semantic_version", name="uq_retrieval_config_version"),
    )
    op.create_index(op.f("ix_retrieval_configs_semantic_version"), "retrieval_configs", ["semantic_version"], unique=False)
    op.create_index(op.f("ix_retrieval_configs_is_active"), "retrieval_configs", ["is_active"], unique=False)

    op.create_table(
        "prompt_templates",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("template_type", sa.String(length=64), nullable=False),
        sa.Column("semantic_version", sa.String(length=32), nullable=False),
        sa.Column("template_text", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        _json_column("metadata_json"),
        _created_at_column(),
        _updated_at_column(),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", "semantic_version", name="uq_prompt_template_version"),
    )
    op.create_index(op.f("ix_prompt_templates_semantic_version"), "prompt_templates", ["semantic_version"], unique=False)
    op.create_index(op.f("ix_prompt_templates_is_active"), "prompt_templates", ["is_active"], unique=False)

    op.create_table(
        "feature_flags",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("rollout_percent", sa.Integer(), nullable=False),
        _json_column("metadata_json"),
        _created_at_column(),
        _updated_at_column(),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_feature_flags_name"), "feature_flags", ["name"], unique=True)
    op.create_index(op.f("ix_feature_flags_enabled"), "feature_flags", ["enabled"], unique=False)

    op.create_table(
        "conversations",
        _uuid_column("id"),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.Column("memory_summary", sa.Text(), nullable=True),
        _json_column("user_preferences"),
        _json_column("interaction_patterns"),
        _json_column("query_refinement_history"),
        sa.Column("custom_instructions", sa.Text(), nullable=True),
        _uuid_column("active_document_id", nullable=True),
        _created_at_column(),
        _updated_at_column(),
        sa.ForeignKeyConstraint(["active_document_id"], ["documents.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_conversations_active_document_id"), "conversations", ["active_document_id"], unique=False)

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        _uuid_column("document_id"),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(VECTOR_DIMENSIONS), nullable=False),
        _created_at_column(),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_document_chunk_index"),
    )
    op.create_index(op.f("ix_document_chunks_document_id"), "document_chunks", ["document_id"], unique=False)

    op.create_table(
        "conversation_turns",
        _uuid_column("id"),
        _uuid_column("conversation_id"),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("route", sa.String(length=64), nullable=True),
        sa.Column("grounded", sa.Boolean(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        _json_column("response_payload"),
        _created_at_column(),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_conversation_turns_conversation_id"), "conversation_turns", ["conversation_id"], unique=False)

    op.create_table(
        "knowledge_graph_nodes",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        _uuid_column("conversation_id"),
        sa.Column("node_type", sa.String(length=32), nullable=False),
        sa.Column("label", sa.String(length=255), nullable=False),
        sa.Column("weight", sa.Float(), nullable=False),
        _json_column("metadata_json"),
        sa.Column("is_deleted", sa.Boolean(), nullable=False),
        _created_at_column(),
        _updated_at_column(),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("conversation_id", "node_type", "label", name="uq_knowledge_node"),
    )
    op.create_index(op.f("ix_knowledge_graph_nodes_conversation_id"), "knowledge_graph_nodes", ["conversation_id"], unique=False)
    op.create_index(op.f("ix_knowledge_graph_nodes_node_type"), "knowledge_graph_nodes", ["node_type"], unique=False)
    op.create_index(op.f("ix_knowledge_graph_nodes_label"), "knowledge_graph_nodes", ["label"], unique=False)
    op.create_index(op.f("ix_knowledge_graph_nodes_is_deleted"), "knowledge_graph_nodes", ["is_deleted"], unique=False)

    op.create_table(
        "document_accesses",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        _uuid_column("conversation_id"),
        _uuid_column("document_id"),
        sa.Column("access_count", sa.Integer(), nullable=False),
        sa.Column("last_accessed_at", sa.DateTime(timezone=True), nullable=True),
        _json_column("metadata_json"),
        _created_at_column(),
        _updated_at_column(),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("conversation_id", "document_id", name="uq_document_access"),
    )
    op.create_index(op.f("ix_document_accesses_conversation_id"), "document_accesses", ["conversation_id"], unique=False)
    op.create_index(op.f("ix_document_accesses_document_id"), "document_accesses", ["document_id"], unique=False)

    op.create_table(
        "experiment_runs",
        _uuid_column("id"),
        _uuid_column("conversation_id", nullable=True),
        sa.Column("experiment_type", sa.String(length=64), nullable=False),
        sa.Column("experiment_name", sa.String(length=255), nullable=False),
        sa.Column("pipeline_version", sa.String(length=32), nullable=False),
        sa.Column("assignment_bucket", sa.String(length=32), nullable=True),
        sa.Column("query_type", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("prompt_template_version", sa.String(length=32), nullable=True),
        sa.Column("retrieval_config_version", sa.String(length=32), nullable=True),
        sa.Column("chat_model_version", sa.String(length=32), nullable=True),
        sa.Column("embedding_model_version", sa.String(length=32), nullable=True),
        _json_column("parameters_json"),
        _json_column("metrics_json"),
        _json_column("costs_json"),
        _json_column("metadata_json"),
        _created_at_column(),
        _updated_at_column(),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_experiment_runs_conversation_id"), "experiment_runs", ["conversation_id"], unique=False)
    op.create_index(op.f("ix_experiment_runs_experiment_type"), "experiment_runs", ["experiment_type"], unique=False)
    op.create_index(op.f("ix_experiment_runs_experiment_name"), "experiment_runs", ["experiment_name"], unique=False)
    op.create_index(op.f("ix_experiment_runs_pipeline_version"), "experiment_runs", ["pipeline_version"], unique=False)
    op.create_index(op.f("ix_experiment_runs_query_type"), "experiment_runs", ["query_type"], unique=False)
    op.create_index(op.f("ix_experiment_runs_status"), "experiment_runs", ["status"], unique=False)

    op.create_table(
        "conversation_checkpoints",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        _uuid_column("conversation_id"),
        _uuid_column("turn_id", nullable=True),
        sa.Column("node_name", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        _json_column("state_payload"),
        _created_at_column(),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["turn_id"], ["conversation_turns.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_conversation_checkpoints_conversation_id"),
        "conversation_checkpoints",
        ["conversation_id"],
        unique=False,
    )
    op.create_index(op.f("ix_conversation_checkpoints_turn_id"), "conversation_checkpoints", ["turn_id"], unique=False)

    op.create_table(
        "memories",
        _uuid_column("id"),
        _uuid_column("conversation_id"),
        _uuid_column("turn_id", nullable=True),
        sa.Column("memory_type", sa.String(length=64), nullable=False),
        sa.Column("scope", sa.String(length=32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("embedding", Vector(VECTOR_DIMENSIONS), nullable=True),
        _json_column("metadata_json"),
        sa.Column("access_count", sa.Integer(), nullable=False),
        sa.Column("last_accessed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_deleted", sa.Boolean(), nullable=False),
        _created_at_column(),
        _updated_at_column(),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["turn_id"], ["conversation_turns.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_memories_conversation_id"), "memories", ["conversation_id"], unique=False)
    op.create_index(op.f("ix_memories_turn_id"), "memories", ["turn_id"], unique=False)
    op.create_index(op.f("ix_memories_memory_type"), "memories", ["memory_type"], unique=False)
    op.create_index(op.f("ix_memories_is_deleted"), "memories", ["is_deleted"], unique=False)

    op.create_table(
        "knowledge_graph_edges",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        _uuid_column("conversation_id"),
        sa.Column("source_node_id", sa.Integer(), nullable=False),
        sa.Column("target_node_id", sa.Integer(), nullable=False),
        sa.Column("relation", sa.String(length=64), nullable=False),
        sa.Column("weight", sa.Float(), nullable=False),
        _json_column("metadata_json"),
        sa.Column("is_deleted", sa.Boolean(), nullable=False),
        _created_at_column(),
        _updated_at_column(),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_node_id"], ["knowledge_graph_nodes.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["target_node_id"], ["knowledge_graph_nodes.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "conversation_id",
            "source_node_id",
            "target_node_id",
            "relation",
            name="uq_knowledge_edge",
        ),
    )
    op.create_index(op.f("ix_knowledge_graph_edges_conversation_id"), "knowledge_graph_edges", ["conversation_id"], unique=False)
    op.create_index(
        op.f("ix_knowledge_graph_edges_source_node_id"),
        "knowledge_graph_edges",
        ["source_node_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_knowledge_graph_edges_target_node_id"),
        "knowledge_graph_edges",
        ["target_node_id"],
        unique=False,
    )
    op.create_index(op.f("ix_knowledge_graph_edges_is_deleted"), "knowledge_graph_edges", ["is_deleted"], unique=False)

    op.create_table(
        "shadow_evaluations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        _uuid_column("experiment_run_id", nullable=True),
        _uuid_column("conversation_id", nullable=True),
        sa.Column("candidate_chat_model", sa.String(length=255), nullable=True),
        sa.Column("candidate_embedding_model", sa.String(length=255), nullable=True),
        sa.Column("candidate_retrieval_config_version", sa.String(length=32), nullable=True),
        sa.Column("candidate_prompt_template_version", sa.String(length=32), nullable=True),
        sa.Column("assignment_bucket", sa.String(length=32), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        _json_column("metrics_json"),
        _json_column("metadata_json"),
        _created_at_column(),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["experiment_run_id"], ["experiment_runs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_shadow_evaluations_experiment_run_id"), "shadow_evaluations", ["experiment_run_id"], unique=False)
    op.create_index(op.f("ix_shadow_evaluations_conversation_id"), "shadow_evaluations", ["conversation_id"], unique=False)
    op.create_index(op.f("ix_shadow_evaluations_status"), "shadow_evaluations", ["status"], unique=False)


def downgrade() -> None:
    op.drop_table("shadow_evaluations")
    op.drop_table("knowledge_graph_edges")
    op.drop_table("memories")
    op.drop_table("conversation_checkpoints")
    op.drop_table("experiment_runs")
    op.drop_table("document_accesses")
    op.drop_table("knowledge_graph_nodes")
    op.drop_table("conversation_turns")
    op.drop_table("document_chunks")
    op.drop_table("conversations")
    op.drop_table("feature_flags")
    op.drop_table("prompt_templates")
    op.drop_table("retrieval_configs")
    op.drop_table("model_registry")
    op.drop_table("documents")
    op.execute("DROP EXTENSION IF EXISTS vector")
