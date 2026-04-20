from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.models import (
    ExperimentRunRecord,
    FeatureFlagRecord,
    ModelRegistryRecord,
    PromptTemplateRecord,
    RetrievalConfigRecord,
    ShadowEvaluationRecord,
)
from app.services.metrics import observe_experiment, observe_provider_publish, observe_shadow

try:  # pragma: no cover - optional dependency
    import mlflow
    from mlflow import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - graceful degradation
    mlflow = None
    MlflowClient = None
    MLFLOW_AVAILABLE = False


@dataclass(frozen=True)
class RuntimeModelProfile:
    chat_model: str
    embedding_model: str
    retrieval_config_version: str
    prompt_template_version: str
    assignment_bucket: str
    shadow_enabled: bool
    shadow_profile: dict[str, str]


class BaseRegistryProvider:
    def __init__(self, name: str, available: bool, note: str | None = None) -> None:
        self.name = name
        self.available = available
        self.note = note

    def log_registry_entry(self, _payload: dict[str, Any]) -> dict[str, Any]:
        return {"provider_note": self.note} if self.note else {}

    def log_experiment(self, _payload: dict[str, Any]) -> dict[str, Any]:
        return {"provider_note": self.note} if self.note else {}


class MlflowRegistryProvider(BaseRegistryProvider):
    def __init__(self, settings: Settings, available: bool, note: str | None = None) -> None:
        super().__init__("mlflow", available, note)
        self.settings = settings

    def _configure(self) -> MlflowClient:
        if not self.available or not MLFLOW_AVAILABLE or mlflow is None or MlflowClient is None:
            raise RuntimeError(self.note or "MLflow is not available.")
        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
        if self.settings.mlflow_registry_uri:
            mlflow.set_registry_uri(self.settings.mlflow_registry_uri)
        mlflow.set_experiment(self.settings.mlflow_experiment_name)
        return MlflowClient()

    def log_registry_entry(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            self._configure()
            with mlflow.start_run(run_name=f"registry-{payload['model_kind']}-{payload['semantic_version']}") as run:
                mlflow.set_tags(
                    {
                        "rag.event_type": "registry_entry",
                        "rag.model_kind": payload["model_kind"],
                        "rag.stage": payload["stage"],
                        "rag.semantic_version": payload["semantic_version"],
                    }
                )
                mlflow.log_params(
                    {
                        "model_kind": payload["model_kind"],
                        "model_name": payload["model_name"],
                        "semantic_version": payload["semantic_version"],
                        "stage": payload["stage"],
                        "is_shadow": payload.get("is_shadow", False),
                    }
                )
                mlflow.log_dict(payload, "artifacts/registry_entry.json")
                if payload.get("checkpoint_uri"):
                    mlflow.log_text(str(payload["checkpoint_uri"]), "artifacts/checkpoint_uri.txt")
                observe_provider_publish(self.name, "registry_entry", True)
                return {"mlflow_run_id": run.info.run_id}
        except Exception as exc:
            observe_provider_publish(self.name, "registry_entry", False)
            return {"provider_note": f"MLflow publish failed: {exc}"}

    def log_experiment(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            self._configure()
            with mlflow.start_run(run_name=payload["experiment_name"]) as run:
                mlflow.set_tags(
                    {
                        "rag.event_type": payload["experiment_type"],
                        "rag.pipeline_version": payload["pipeline_version"],
                        "rag.query_type": payload.get("query_type", ""),
                        "rag.assignment_bucket": payload.get("assignment_bucket", ""),
                    }
                )
                mlflow.log_params(payload.get("parameters", {}))
                metrics = payload.get("metrics", {})
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, float(value))
                costs = payload.get("costs", {})
                for key, value in costs.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, float(value))
                mlflow.log_dict(payload, "artifacts/experiment.json")
                observe_provider_publish(self.name, "experiment", True)
                return {"mlflow_run_id": run.info.run_id}
        except Exception as exc:
            observe_provider_publish(self.name, "experiment", False)
            return {"provider_note": f"MLflow publish failed: {exc}"}


class ModelManagementService:
    def __init__(self, settings: Settings, session: Session) -> None:
        self.settings = settings
        self.session = session
        self.provider = self._build_provider()

    def _build_provider(self) -> BaseRegistryProvider:
        if self.settings.model_registry_provider == "mlflow":
            available = bool(self.settings.mlflow_tracking_uri and MLFLOW_AVAILABLE)
            note = None if available else "MLflow selected but not configured; local registry remains active."
            return MlflowRegistryProvider(self.settings, available, note)
        if self.settings.model_registry_provider == "wandb":
            available = bool(self.settings.wandb_project and self.settings.wandb_api_key)
            note = None if available else "Weights & Biases selected but not configured; local registry remains active."
            return BaseRegistryProvider("wandb", available, note)
        return BaseRegistryProvider("local", True, None)

    def bootstrap_defaults(self) -> None:
        chat_active_exists = self._has_active_registry("chat")
        embedding_active_exists = self._has_active_registry("embedding")
        self._upsert_model_registry(
            model_kind="chat",
            model_name=self.settings.chat_model,
            semantic_version=self.settings.pipeline_version,
            stage="active" if not chat_active_exists else "registered",
            is_active=not chat_active_exists,
            is_shadow=False,
            metadata_json={"external_version": self.settings.chat_model},
        )
        self._upsert_model_registry(
            model_kind="embedding",
            model_name=self.settings.embedding_model,
            semantic_version=self.settings.pipeline_version,
            stage="active" if not embedding_active_exists else "registered",
            is_active=not embedding_active_exists,
            is_shadow=False,
            metadata_json={"external_version": self.settings.embedding_model},
        )
        if self.settings.experimental_chat_model:
            self._upsert_model_registry(
                model_kind="chat",
                model_name=self.settings.experimental_chat_model,
                semantic_version=f"{self.settings.pipeline_version}-exp",
                stage="canary",
                is_active=False,
                is_shadow=False,
                metadata_json={"external_version": self.settings.experimental_chat_model},
            )
        if self.settings.experimental_embedding_model:
            self._upsert_model_registry(
                model_kind="embedding",
                model_name=self.settings.experimental_embedding_model,
                semantic_version=f"{self.settings.pipeline_version}-exp",
                stage="canary",
                is_active=False,
                is_shadow=False,
                metadata_json={"external_version": self.settings.experimental_embedding_model},
            )
        if self.settings.shadow_chat_model:
            self._upsert_model_registry(
                model_kind="chat",
                model_name=self.settings.shadow_chat_model,
                semantic_version=f"{self.settings.pipeline_version}-shadow",
                stage="shadow",
                is_active=False,
                is_shadow=True,
                metadata_json={"external_version": self.settings.shadow_chat_model},
            )
        if self.settings.shadow_embedding_model:
            self._upsert_model_registry(
                model_kind="embedding",
                model_name=self.settings.shadow_embedding_model,
                semantic_version=f"{self.settings.pipeline_version}-shadow",
                stage="shadow",
                is_active=False,
                is_shadow=True,
                metadata_json={"external_version": self.settings.shadow_embedding_model},
            )

        self._upsert_retrieval_config(
            name="default",
            semantic_version=self.settings.retrieval_config_version,
            is_active=not self._has_active_retrieval_config(),
            metadata_json=self._retrieval_config_payload(),
        )
        self._upsert_prompt_template(
            name="multi-agent-rag",
            template_type="chat",
            semantic_version=self.settings.prompt_template_version,
            is_active=not self._has_active_prompt_template(),
            template_text=None,
            metadata_json={"pipeline_version": self.settings.pipeline_version},
        )
        self._upsert_feature_flag(
            "model_ab_test",
            enabled=self.settings.model_ab_test_enabled,
            rollout_percent=self.settings.model_ab_test_rollout_percent,
            metadata_json={
                "experimental_chat_model": self.settings.experimental_chat_model,
                "experimental_embedding_model": self.settings.experimental_embedding_model,
                "experimental_retrieval_config_version": self.settings.experimental_retrieval_config_version,
                "experimental_prompt_template_version": self.settings.experimental_prompt_template_version,
            },
        )
        self._upsert_feature_flag(
            "shadow_mode",
            enabled=self.settings.shadow_mode_enabled,
            rollout_percent=self.settings.shadow_sampling_percent,
            metadata_json={
                "shadow_chat_model": self.settings.shadow_chat_model,
                "shadow_embedding_model": self.settings.shadow_embedding_model,
                "shadow_retrieval_config_version": self.settings.shadow_retrieval_config_version,
                "shadow_prompt_template_version": self.settings.shadow_prompt_template_version,
            },
        )
        self.session.commit()

    def _retrieval_config_payload(self) -> dict[str, Any]:
        return {
            "vector_weight": self.settings.vector_weight,
            "bm25_weight": self.settings.bm25_weight,
            "overlap_weight": self.settings.overlap_weight,
            "min_retrieval_score": self.settings.min_retrieval_score,
            "retriever_fetch_k": self.settings.retriever_fetch_k,
            "max_context_chunks": self.settings.max_context_chunks,
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
            "chunking_strategy_version": self.settings.chunking_strategy_version,
        }

    def runtime_profile(self, assignment_seed: str) -> RuntimeModelProfile:
        ab_flag = self._feature_flag("model_ab_test")
        shadow_flag = self._feature_flag("shadow_mode")

        assignment_bucket = "control"
        chat_model = self.active_model_name("chat", self.settings.chat_model)
        embedding_model = self.active_model_name("embedding", self.settings.embedding_model)
        retrieval_config_version = self.active_retrieval_config_version()
        prompt_template_version = self.active_prompt_template_version()

        if ab_flag and ab_flag.enabled and self._is_in_rollout(assignment_seed, ab_flag.rollout_percent):
            assignment_bucket = "candidate"
            chat_model = self.settings.experimental_chat_model or chat_model
            embedding_model = self.settings.experimental_embedding_model or embedding_model
            retrieval_config_version = self.settings.experimental_retrieval_config_version or retrieval_config_version
            prompt_template_version = self.settings.experimental_prompt_template_version or prompt_template_version

        shadow_enabled = bool(
            shadow_flag
            and shadow_flag.enabled
            and self._is_in_rollout(f"shadow:{assignment_seed}", shadow_flag.rollout_percent)
        )
        shadow_profile = {
            "chat_model": self.settings.shadow_chat_model or self.settings.chat_model,
            "embedding_model": self.settings.shadow_embedding_model or self.settings.embedding_model,
            "retrieval_config_version": self.settings.shadow_retrieval_config_version or self.settings.retrieval_config_version,
            "prompt_template_version": self.settings.shadow_prompt_template_version or self.settings.prompt_template_version,
        }

        return RuntimeModelProfile(
            chat_model=chat_model,
            embedding_model=embedding_model,
            retrieval_config_version=retrieval_config_version,
            prompt_template_version=prompt_template_version,
            assignment_bucket=assignment_bucket,
            shadow_enabled=shadow_enabled,
            shadow_profile=shadow_profile,
        )

    def log_query_experiment(
        self,
        *,
        conversation_id: UUID | None,
        experiment_name: str,
        query_type: str,
        runtime_profile: RuntimeModelProfile,
        parameters_json: dict[str, Any],
        metrics_json: dict[str, Any],
        costs_json: dict[str, Any],
        latency_ms: float,
        metadata_json: dict[str, Any] | None = None,
    ) -> ExperimentRunRecord:
        record = ExperimentRunRecord(
            conversation_id=conversation_id,
            experiment_type="query",
            experiment_name=experiment_name,
            pipeline_version=self.settings.pipeline_version,
            assignment_bucket=runtime_profile.assignment_bucket,
            query_type=query_type,
            status="completed",
            latency_ms=latency_ms,
            prompt_template_version=runtime_profile.prompt_template_version,
            retrieval_config_version=runtime_profile.retrieval_config_version,
            chat_model_version=runtime_profile.chat_model,
            embedding_model_version=runtime_profile.embedding_model,
            parameters_json=parameters_json,
            metrics_json=metrics_json,
            costs_json=costs_json,
            metadata_json=metadata_json or {},
        )
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)
        provider_metadata = self.provider.log_experiment(
            {
                "experiment_type": record.experiment_type,
                "experiment_name": record.experiment_name,
                "pipeline_version": record.pipeline_version,
                "assignment_bucket": record.assignment_bucket,
                "query_type": record.query_type,
                "parameters": parameters_json,
                "metrics": metrics_json,
                "costs": costs_json,
                "latency_ms": latency_ms,
                "metadata": metadata_json or {},
            }
        )
        observe_experiment(record.experiment_type, record.status, record.assignment_bucket or "control")
        if provider_metadata:
            record.metadata_json = {**(record.metadata_json or {}), **provider_metadata}
            self.session.add(record)
            self.session.commit()
            self.session.refresh(record)
        return record

    def log_index_experiment(
        self,
        *,
        document_id: UUID,
        page_count: int,
        chunk_count: int,
        duration_ms: float,
        metadata_json: dict[str, Any] | None = None,
    ) -> ExperimentRunRecord:
        record = ExperimentRunRecord(
            conversation_id=None,
            experiment_type="indexing",
            experiment_name=f"index-document-{str(document_id)[:8]}",
            pipeline_version=self.settings.pipeline_version,
            assignment_bucket="active",
            query_type="indexing",
            status="completed",
            latency_ms=duration_ms,
            prompt_template_version=None,
            retrieval_config_version=self.settings.retrieval_config_version,
            chat_model_version=None,
            embedding_model_version=self.settings.embedding_model,
            parameters_json={
                "chunk_size": self.settings.chunk_size,
                "chunk_overlap": self.settings.chunk_overlap,
                "chunking_strategy_version": self.settings.chunking_strategy_version,
            },
            metrics_json={"page_count": page_count, "chunk_count": chunk_count},
            costs_json={"estimated_embedding_cost_usd": self.estimate_embedding_cost(page_count + chunk_count)},
            metadata_json={**(metadata_json or {}), "document_id": str(document_id)},
        )
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)
        provider_metadata = self.provider.log_experiment(
            {
                "experiment_type": record.experiment_type,
                "experiment_name": record.experiment_name,
                "pipeline_version": record.pipeline_version,
                "assignment_bucket": record.assignment_bucket,
                "query_type": record.query_type,
                "parameters": record.parameters_json,
                "metrics": record.metrics_json,
                "costs": record.costs_json,
                "latency_ms": record.latency_ms,
                "metadata": record.metadata_json,
            }
        )
        observe_experiment(record.experiment_type, record.status, record.assignment_bucket or "active")
        if provider_metadata:
            record.metadata_json = {**(record.metadata_json or {}), **provider_metadata}
            self.session.add(record)
            self.session.commit()
            self.session.refresh(record)
        return record

    def log_shadow_evaluation(
        self,
        *,
        experiment_run_id: UUID | None,
        conversation_id: UUID | None,
        runtime_profile: RuntimeModelProfile,
        latency_ms: float,
        metrics_json: dict[str, Any],
        metadata_json: dict[str, Any] | None = None,
    ) -> ShadowEvaluationRecord:
        record = ShadowEvaluationRecord(
            experiment_run_id=experiment_run_id,
            conversation_id=conversation_id,
            candidate_chat_model=runtime_profile.shadow_profile.get("chat_model"),
            candidate_embedding_model=runtime_profile.shadow_profile.get("embedding_model"),
            candidate_retrieval_config_version=runtime_profile.shadow_profile.get("retrieval_config_version"),
            candidate_prompt_template_version=runtime_profile.shadow_profile.get("prompt_template_version"),
            assignment_bucket=runtime_profile.assignment_bucket,
            status="logged",
            latency_ms=latency_ms,
            metrics_json=metrics_json,
            metadata_json=metadata_json or {},
        )
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)
        observe_shadow(record.status, record.assignment_bucket or "control")
        return record

    def estimate_query_cost(
        self,
        *,
        query_text: str,
        answer_text: str,
        retrieved_chunks: int,
        query_type: str,
    ) -> dict[str, float]:
        input_tokens = max(len(query_text) / 4.0 + retrieved_chunks * 180.0, 1.0)
        output_tokens = max(len(answer_text) / 4.0, 1.0)
        total_cost = (
            (input_tokens / 1000.0) * self.settings.estimated_input_token_cost_per_1k
            + (output_tokens / 1000.0) * self.settings.estimated_output_token_cost_per_1k
        )
        if query_type == "retrieval_only":
            total_cost *= 0.4
        return {
            "estimated_input_tokens": round(input_tokens, 2),
            "estimated_output_tokens": round(output_tokens, 2),
            "estimated_total_cost_usd": round(total_cost, 6),
        }

    def estimate_embedding_cost(self, item_count: int) -> float:
        return round((max(item_count, 1) / 1000.0) * self.settings.estimated_embedding_cost_per_1k, 6)

    def retrieval_metrics(
        self,
        *,
        retrieved_chunks: list[Any],
        cited_chunk_ids: set[str],
        top_k: int,
        grounding_threshold: float,
    ) -> dict[str, float]:
        if not retrieved_chunks:
            return {"mrr": 0.0, "ndcg": 0.0, "precision_at_k": 0.0, "recall_at_k": 0.0}

        first_relevant_rank = 0
        gains: list[float] = []
        ideal_gains: list[float] = []
        relevant_total = 0
        for idx, chunk in enumerate(retrieved_chunks[:top_k], start=1):
            relevant = chunk.chunk_id in cited_chunk_ids or chunk.score >= grounding_threshold
            if relevant:
                relevant_total += 1
                if first_relevant_rank == 0:
                    first_relevant_rank = idx
            gain = chunk.score if relevant else 0.0
            gains.append(gain / math.log2(idx + 1))

        ideal_scores = sorted([chunk.score for chunk in retrieved_chunks[:top_k]], reverse=True)
        for idx, score in enumerate(ideal_scores, start=1):
            ideal_gains.append(score / math.log2(idx + 1))

        dcg = sum(gains)
        idcg = sum(ideal_gains) or 1.0
        precision = relevant_total / max(min(top_k, len(retrieved_chunks)), 1)
        recall = relevant_total / max(sum(1 for chunk in retrieved_chunks if chunk.score >= grounding_threshold), 1)
        return {
            "mrr": round(1.0 / first_relevant_rank, 4) if first_relevant_rank else 0.0,
            "ndcg": round(min(dcg / idcg, 1.0), 4),
            "precision_at_k": round(min(max(precision, 0.0), 1.0), 4),
            "recall_at_k": round(min(max(recall, 0.0), 1.0), 4),
        }

    def llm_metrics(self, *, grounded: bool, confidence: float, latency_ms: float) -> dict[str, float]:
        hallucination_rate = 0.0 if grounded else 1.0
        return {
            "accuracy_proxy": round(confidence if grounded else confidence * 0.35, 4),
            "hallucination_rate": round(hallucination_rate, 4),
            "latency_ms": round(latency_ms, 2),
        }

    def list_registry_entries(self) -> list[ModelRegistryRecord]:
        statement = select(ModelRegistryRecord).order_by(
            ModelRegistryRecord.model_kind.asc(),
            ModelRegistryRecord.created_at.desc(),
        )
        return list(self.session.scalars(statement))

    def list_retrieval_configs(self) -> list[RetrievalConfigRecord]:
        statement = select(RetrievalConfigRecord).order_by(RetrievalConfigRecord.created_at.desc())
        return list(self.session.scalars(statement))

    def list_prompt_templates(self) -> list[PromptTemplateRecord]:
        statement = select(PromptTemplateRecord).order_by(PromptTemplateRecord.created_at.desc())
        return list(self.session.scalars(statement))

    def list_feature_flags(self) -> list[FeatureFlagRecord]:
        statement = select(FeatureFlagRecord).order_by(FeatureFlagRecord.name.asc())
        return list(self.session.scalars(statement))

    def list_recent_experiments(self, limit: int = 25) -> list[ExperimentRunRecord]:
        statement = select(ExperimentRunRecord).order_by(ExperimentRunRecord.created_at.desc()).limit(limit)
        return list(self.session.scalars(statement))

    def list_recent_shadow_evaluations(self, limit: int = 25) -> list[ShadowEvaluationRecord]:
        statement = select(ShadowEvaluationRecord).order_by(ShadowEvaluationRecord.created_at.desc()).limit(limit)
        return list(self.session.scalars(statement))

    def active_model_name(self, model_kind: str, default: str) -> str:
        record = self.session.scalar(
            select(ModelRegistryRecord).where(
                ModelRegistryRecord.model_kind == model_kind,
                ModelRegistryRecord.is_active.is_(True),
            )
        )
        return record.model_name if record is not None else default

    def active_retrieval_config_version(self) -> str:
        record = self.session.scalar(select(RetrievalConfigRecord).where(RetrievalConfigRecord.is_active.is_(True)))
        return record.semantic_version if record is not None else self.settings.retrieval_config_version

    def active_prompt_template_version(self) -> str:
        record = self.session.scalar(select(PromptTemplateRecord).where(PromptTemplateRecord.is_active.is_(True)))
        return record.semantic_version if record is not None else self.settings.prompt_template_version

    def activate_registry_version(self, model_kind: str, semantic_version: str) -> None:
        records = list(
            self.session.scalars(select(ModelRegistryRecord).where(ModelRegistryRecord.model_kind == model_kind))
        )
        for record in records:
            record.is_active = record.semantic_version == semantic_version
            record.stage = "active" if record.is_active else ("shadow" if record.is_shadow else "archived")
            self.session.add(record)
        self.session.commit()

    def rollback_to_version(self, model_kind: str, semantic_version: str) -> None:
        self.activate_registry_version(model_kind, semantic_version)

    def _feature_flag(self, name: str) -> FeatureFlagRecord | None:
        return self.session.scalar(select(FeatureFlagRecord).where(FeatureFlagRecord.name == name))

    def _has_active_registry(self, model_kind: str) -> bool:
        return bool(
            self.session.scalar(
                select(ModelRegistryRecord.id).where(
                    ModelRegistryRecord.model_kind == model_kind,
                    ModelRegistryRecord.is_active.is_(True),
                )
            )
        )

    def _has_active_retrieval_config(self) -> bool:
        return bool(self.session.scalar(select(RetrievalConfigRecord.id).where(RetrievalConfigRecord.is_active.is_(True))))

    def _has_active_prompt_template(self) -> bool:
        return bool(self.session.scalar(select(PromptTemplateRecord.id).where(PromptTemplateRecord.is_active.is_(True))))

    @staticmethod
    def _is_in_rollout(seed: str, rollout_percent: int) -> bool:
        if rollout_percent <= 0:
            return False
        if rollout_percent >= 100:
            return True
        hashed = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        bucket = int(hashed[:8], 16) % 100
        return bucket < rollout_percent

    def _upsert_model_registry(
        self,
        *,
        model_kind: str,
        model_name: str,
        semantic_version: str,
        stage: str,
        is_active: bool,
        is_shadow: bool,
        metadata_json: dict[str, Any],
    ) -> ModelRegistryRecord:
        record = self.session.scalar(
            select(ModelRegistryRecord).where(
                ModelRegistryRecord.model_kind == model_kind,
                ModelRegistryRecord.semantic_version == semantic_version,
            )
        )
        if record is None:
            record = ModelRegistryRecord(
                model_kind=model_kind,
                provider=self.provider.name,
                model_name=model_name,
                semantic_version=semantic_version,
                external_version=model_name,
                stage=stage,
                is_active=is_active,
                is_shadow=is_shadow,
                metadata_json=metadata_json,
            )
        else:
            record.provider = self.provider.name
            record.model_name = model_name
            record.external_version = model_name
            record.stage = stage
            record.is_active = is_active
            record.is_shadow = is_shadow
            record.metadata_json = {**(record.metadata_json or {}), **metadata_json}
        self.session.add(record)
        provider_metadata = self.provider.log_registry_entry(
            {
                "model_kind": model_kind,
                "model_name": model_name,
                "semantic_version": semantic_version,
                "stage": stage,
                "is_shadow": is_shadow,
                "checkpoint_uri": record.checkpoint_uri,
                "metadata_json": metadata_json,
            }
        )
        if provider_metadata:
            record.metadata_json = {**(record.metadata_json or {}), **provider_metadata}
        self.session.flush()
        return record

    def _upsert_retrieval_config(
        self,
        *,
        name: str,
        semantic_version: str,
        is_active: bool,
        metadata_json: dict[str, Any],
    ) -> RetrievalConfigRecord:
        record = self.session.scalar(
            select(RetrievalConfigRecord).where(
                RetrievalConfigRecord.name == name,
                RetrievalConfigRecord.semantic_version == semantic_version,
            )
        )
        if record is None:
            record = RetrievalConfigRecord(
                name=name,
                semantic_version=semantic_version,
                is_active=is_active,
                metadata_json=metadata_json,
            )
        else:
            record.is_active = is_active
            record.metadata_json = {**(record.metadata_json or {}), **metadata_json}
        self.session.add(record)
        self.session.flush()
        return record

    def _upsert_prompt_template(
        self,
        *,
        name: str,
        template_type: str,
        semantic_version: str,
        is_active: bool,
        template_text: str | None,
        metadata_json: dict[str, Any],
    ) -> PromptTemplateRecord:
        record = self.session.scalar(
            select(PromptTemplateRecord).where(
                PromptTemplateRecord.name == name,
                PromptTemplateRecord.semantic_version == semantic_version,
            )
        )
        if record is None:
            record = PromptTemplateRecord(
                name=name,
                template_type=template_type,
                semantic_version=semantic_version,
                template_text=template_text,
                is_active=is_active,
                metadata_json=metadata_json,
            )
        else:
            record.template_type = template_type
            record.template_text = template_text
            record.is_active = is_active
            record.metadata_json = {**(record.metadata_json or {}), **metadata_json}
        self.session.add(record)
        self.session.flush()
        return record

    def _upsert_feature_flag(
        self,
        name: str,
        *,
        enabled: bool,
        rollout_percent: int,
        metadata_json: dict[str, Any],
    ) -> FeatureFlagRecord:
        record = self._feature_flag(name)
        if record is None:
            record = FeatureFlagRecord(
                name=name,
                enabled=enabled,
                rollout_percent=rollout_percent,
                metadata_json=metadata_json,
            )
        else:
            record.enabled = enabled
            record.rollout_percent = rollout_percent
            record.metadata_json = {**(record.metadata_json or {}), **metadata_json}
        self.session.add(record)
        self.session.flush()
        return record
