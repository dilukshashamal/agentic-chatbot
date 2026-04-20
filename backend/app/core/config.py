from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    app_name: str = "Document RAG API"
    api_prefix: str = "/api/v1"
    environment: Literal["development", "production", "test"] = "development"

    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    database_url: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/rag_chatbot",
        alias="DATABASE_URL",
    )

    upload_dir: Path = BACKEND_ROOT / "data" / "uploads"

    embedding_model: str = "models/gemini-embedding-001"
    embedding_dimensions: int = Field(default=3072, ge=128, le=8192)
    chat_model: str = "gemini-2.5-flash"

    chunk_size: int = Field(default=1000, ge=300, le=2000)
    chunk_overlap: int = Field(default=150, ge=0, le=400)

    retriever_k: int = Field(default=4, ge=2, le=8)
    retriever_fetch_k: int = Field(default=10, ge=4, le=24)
    max_context_chunks: int = Field(default=4, ge=2, le=8)

    vector_weight: float = Field(default=0.65, ge=0.0, le=1.0)
    bm25_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    overlap_weight: float = Field(default=0.10, ge=0.0, le=1.0)
    min_retrieval_score: float = Field(default=0.18, ge=0.0, le=1.0)

    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    export_dir: Path = BACKEND_ROOT / "data" / "exports"

    orchestration_model: str = "gemini-2.5-flash"
    supervisor_max_steps: int = Field(default=4, ge=1, le=8)
    conversation_history_window: int = Field(default=10, ge=2, le=20)
    max_agent_retries: int = Field(default=2, ge=0, le=5)
    web_search_enabled: bool = True
    calculator_enabled: bool = True
    code_interpreter_enabled: bool = True
    chart_generation_enabled: bool = True
    export_enabled: bool = True

    memory_provider: Literal["local", "mem0", "zep"] = "local"
    memory_search_k: int = Field(default=5, ge=1, le=12)
    memory_similarity_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    query_refinement_window: int = Field(default=10, ge=3, le=30)
    memory_summary_window: int = Field(default=4, ge=2, le=10)
    faq_min_count: int = Field(default=2, ge=1, le=20)
    knowledge_graph_max_entities: int = Field(default=8, ge=2, le=20)
    enable_semantic_memory_search: bool = True

    mem0_api_key: str | None = Field(default=None, alias="MEM0_API_KEY")
    mem0_base_url: str | None = Field(default=None, alias="MEM0_BASE_URL")
    zep_api_key: str | None = Field(default=None, alias="ZEP_API_KEY")
    zep_base_url: str | None = Field(default=None, alias="ZEP_BASE_URL")

    pipeline_version: str = "1.0.0"
    chunking_strategy_version: str = "1.0.0"
    retrieval_config_version: str = "1.0.0"
    prompt_template_version: str = "1.0.0"

    model_registry_provider: Literal["local", "mlflow", "wandb"] = "local"
    mlflow_tracking_uri: str | None = Field(default=None, alias="MLFLOW_TRACKING_URI")
    mlflow_registry_uri: str | None = Field(default=None, alias="MLFLOW_REGISTRY_URI")
    mlflow_experiment_name: str = "rag-chatbot"
    mlflow_registry_prefix: str = "rag"
    wandb_project: str | None = Field(default=None, alias="WANDB_PROJECT")
    wandb_entity: str | None = Field(default=None, alias="WANDB_ENTITY")
    wandb_api_key: str | None = Field(default=None, alias="WANDB_API_KEY")

    model_ab_test_enabled: bool = False
    model_ab_test_rollout_percent: int = Field(default=10, ge=0, le=100)
    experimental_chat_model: str | None = None
    experimental_embedding_model: str | None = None
    experimental_retrieval_config_version: str | None = None
    experimental_prompt_template_version: str | None = None

    shadow_mode_enabled: bool = False
    shadow_sampling_percent: int = Field(default=10, ge=0, le=100)
    shadow_chat_model: str | None = None
    shadow_embedding_model: str | None = None
    shadow_retrieval_config_version: str | None = None
    shadow_prompt_template_version: str | None = None

    estimated_input_token_cost_per_1k: float = Field(default=0.0003, ge=0.0)
    estimated_output_token_cost_per_1k: float = Field(default=0.0006, ge=0.0)
    estimated_embedding_cost_per_1k: float = Field(default=0.0001, ge=0.0)

    metrics_enabled: bool = True
    metrics_namespace: str = "rag"

    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )

    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
