from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_db_session
from app.models.schemas import SystemStatus
from app.services.rag import RAGService

router = APIRouter(tags=["health"])


@router.get("/health", response_model=SystemStatus)
def health_check(
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> SystemStatus:
    return RAGService(settings, session).system_status()
