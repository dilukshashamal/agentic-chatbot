from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_db_session
from app.models.schemas import ChatRequest, ChatResponse
from app.services.rag import RAGService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/query", response_model=ChatResponse)
async def query_chat(
    payload: ChatRequest,
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> ChatResponse:
    try:
        service = RAGService(settings, session)
        return await run_in_threadpool(service.answer_question, payload)
    except Exception as exc:  # pragma: no cover - thin transport layer
        raise HTTPException(status_code=500, detail=str(exc)) from exc
