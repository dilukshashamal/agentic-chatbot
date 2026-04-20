from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_db_session
from app.models.schemas import (
    ExperimentRunSummary,
    FeatureFlagSummary,
    ModelManagementStatus,
    ModelRegistryEntry,
    PromptTemplateVersion,
    RetrievalConfigVersion,
    ShadowEvaluationSummary,
)
from app.services.model_management import ModelManagementService

router = APIRouter(prefix="/model-management", tags=["model-management"])


@router.get("/status", response_model=ModelManagementStatus)
def model_management_status(
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> ModelManagementStatus:
    service = ModelManagementService(settings, session)
    service.bootstrap_defaults()
    return ModelManagementStatus(
        pipeline_version=settings.pipeline_version,
        registry_provider=settings.model_registry_provider,
        active_chat_model=service.active_model_name("chat", settings.chat_model),
        active_embedding_model=service.active_model_name("embedding", settings.embedding_model),
        retrieval_config_version=service.active_retrieval_config_version(),
        prompt_template_version=service.active_prompt_template_version(),
        ab_test_enabled=settings.model_ab_test_enabled,
        ab_test_rollout_percent=settings.model_ab_test_rollout_percent,
        shadow_mode_enabled=settings.shadow_mode_enabled,
        shadow_sampling_percent=settings.shadow_sampling_percent,
        feature_flags=[FeatureFlagSummary.model_validate(item) for item in service.list_feature_flags()],
        registry_entries=[ModelRegistryEntry.model_validate(item) for item in service.list_registry_entries()],
    )


@router.get("/registry", response_model=list[ModelRegistryEntry])
def list_registry(
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> list[ModelRegistryEntry]:
    service = ModelManagementService(settings, session)
    return [ModelRegistryEntry.model_validate(item) for item in service.list_registry_entries()]


@router.get("/retrieval-configs", response_model=list[RetrievalConfigVersion])
def list_retrieval_configs(
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> list[RetrievalConfigVersion]:
    service = ModelManagementService(settings, session)
    return [RetrievalConfigVersion.model_validate(item) for item in service.list_retrieval_configs()]


@router.get("/prompt-templates", response_model=list[PromptTemplateVersion])
def list_prompt_templates(
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> list[PromptTemplateVersion]:
    service = ModelManagementService(settings, session)
    return [PromptTemplateVersion.model_validate(item) for item in service.list_prompt_templates()]


@router.get("/feature-flags", response_model=list[FeatureFlagSummary])
def list_feature_flags(
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> list[FeatureFlagSummary]:
    service = ModelManagementService(settings, session)
    return [FeatureFlagSummary.model_validate(item) for item in service.list_feature_flags()]


@router.get("/experiments", response_model=list[ExperimentRunSummary])
def list_experiments(
    limit: int = 25,
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> list[ExperimentRunSummary]:
    service = ModelManagementService(settings, session)
    return [ExperimentRunSummary.model_validate(item) for item in service.list_recent_experiments(limit=limit)]


@router.get("/shadow-evaluations", response_model=list[ShadowEvaluationSummary])
def list_shadow_evaluations(
    limit: int = 25,
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> list[ShadowEvaluationSummary]:
    service = ModelManagementService(settings, session)
    return [ShadowEvaluationSummary.model_validate(item) for item in service.list_recent_shadow_evaluations(limit=limit)]


@router.post("/rollback/{model_kind}/{semantic_version}")
def rollback_model_version(
    model_kind: str,
    semantic_version: str,
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, str]:
    if model_kind not in {"chat", "embedding"}:
        raise HTTPException(status_code=400, detail="model_kind must be 'chat' or 'embedding'.")

    service = ModelManagementService(settings, session)
    matching_versions = [
        item
        for item in service.list_registry_entries()
        if item.model_kind == model_kind and item.semantic_version == semantic_version
    ]
    if not matching_versions:
        raise HTTPException(status_code=404, detail="Requested model version was not found in the registry.")

    service.rollback_to_version(model_kind, semantic_version)
    return {"message": f"Rolled back {model_kind} registry stage to version {semantic_version}."}
