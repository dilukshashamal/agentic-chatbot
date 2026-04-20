from fastapi import APIRouter, Depends, Response
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_db_session
from app.models.schemas import SystemStatus
from app.services.metrics import CONTENT_TYPE_LATEST, render_metrics
from app.services.rag import RAGService

router = APIRouter(tags=["health"])


@router.get("/health", response_model=SystemStatus)
def health_check(
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> SystemStatus:
    return RAGService(settings, session).system_status()


@router.get("/metrics", include_in_schema=False)
def metrics(settings: Settings = Depends(get_settings)) -> Response:
    if not settings.metrics_enabled:
        return Response(status_code=404)
    return Response(content=render_metrics(), media_type=CONTENT_TYPE_LATEST)
