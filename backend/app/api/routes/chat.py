from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_db_session
from app.models.schemas import ChatRequest, ChatResponse, ConversationDetail, ExportFormat
from app.services.conversations import ConversationService
from app.services.exports import ExportService
from app.services.orchestration import MultiAgentOrchestrator

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/query", response_model=ChatResponse)
async def query_chat(
    payload: ChatRequest,
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> ChatResponse:
    try:
        service = MultiAgentOrchestrator(settings, session)
        return await run_in_threadpool(service.answer_question, payload)
    except Exception as exc:  # pragma: no cover - thin transport layer
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(
    conversation_id: UUID,
    session: Session = Depends(get_db_session),
) -> ConversationDetail:
    conversation = ConversationService(session).get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    ordered_turns = sorted(conversation.turns, key=lambda turn: turn.created_at)
    return ConversationDetail(
        id=conversation.id,
        title=conversation.title,
        memory_summary=conversation.memory_summary,
        user_preferences=conversation.user_preferences or {},
        interaction_patterns=conversation.interaction_patterns or {},
        query_refinement_history=conversation.query_refinement_history or [],
        custom_instructions=conversation.custom_instructions,
        active_document_id=conversation.active_document_id,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        turns=[turn for turn in ordered_turns],
    )


@router.get("/conversations/{conversation_id}/export/{export_format}")
def export_conversation(
    conversation_id: UUID,
    export_format: ExportFormat,
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    conversation = ConversationService(session).get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    ordered_turns = sorted(conversation.turns, key=lambda turn: turn.created_at)
    detail = ConversationDetail(
        id=conversation.id,
        title=conversation.title,
        memory_summary=conversation.memory_summary,
        user_preferences=conversation.user_preferences or {},
        interaction_patterns=conversation.interaction_patterns or {},
        query_refinement_history=conversation.query_refinement_history or [],
        custom_instructions=conversation.custom_instructions,
        active_document_id=conversation.active_document_id,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        turns=[turn for turn in ordered_turns],
    )
    output_path = ExportService(settings).export_conversation(detail, export_format)
    media_type = {
        "json": "application/json",
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }[export_format]
    return FileResponse(
        path=Path(output_path),
        media_type=media_type,
        filename=Path(output_path).name,
    )
