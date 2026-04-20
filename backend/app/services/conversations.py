from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.db.models import (
    ConversationCheckpointRecord,
    ConversationRecord,
    ConversationTurnRecord,
    DocumentAccessRecord,
    KnowledgeGraphEdgeRecord,
    KnowledgeGraphNodeRecord,
    MemoryRecord,
)


class ConversationService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_or_create_conversation(
        self,
        conversation_id: UUID | None,
        user_preferences: dict[str, str] | None = None,
    ) -> ConversationRecord:
        conversation: ConversationRecord | None = None
        if conversation_id is not None:
            conversation = self.session.get(ConversationRecord, conversation_id)

        if conversation is None:
            conversation = ConversationRecord(
                user_preferences=dict(user_preferences or {}),
                interaction_patterns={},
                query_refinement_history=[],
            )
            self.session.add(conversation)
            self.session.commit()
            self.session.refresh(conversation)
            return conversation

        if user_preferences:
            conversation.user_preferences = {
                **(conversation.user_preferences or {}),
                **user_preferences,
            }
            self.session.add(conversation)
            self.session.commit()
            self.session.refresh(conversation)
        return conversation

    def get_conversation(self, conversation_id: UUID) -> ConversationRecord | None:
        statement = (
            select(ConversationRecord)
            .options(selectinload(ConversationRecord.turns))
            .where(ConversationRecord.id == conversation_id)
        )
        return self.session.scalar(statement)

    def get_memories(self, conversation_id: UUID) -> list[MemoryRecord]:
        statement = (
            select(MemoryRecord)
            .where(MemoryRecord.conversation_id == conversation_id, MemoryRecord.is_deleted.is_(False))
            .order_by(MemoryRecord.created_at.desc())
        )
        return list(self.session.scalars(statement))

    def get_graph_nodes(self, conversation_id: UUID) -> list[KnowledgeGraphNodeRecord]:
        statement = (
            select(KnowledgeGraphNodeRecord)
            .where(KnowledgeGraphNodeRecord.conversation_id == conversation_id, KnowledgeGraphNodeRecord.is_deleted.is_(False))
            .order_by(KnowledgeGraphNodeRecord.weight.desc(), KnowledgeGraphNodeRecord.label.asc())
        )
        return list(self.session.scalars(statement))

    def count_conversations(self) -> int:
        return self.session.scalar(select(func.count()).select_from(ConversationRecord)) or 0

    def recent_turns(self, conversation_id: UUID, limit: int = 6) -> list[ConversationTurnRecord]:
        statement = (
            select(ConversationTurnRecord)
            .where(ConversationTurnRecord.conversation_id == conversation_id)
            .order_by(ConversationTurnRecord.created_at.desc())
            .limit(limit)
        )
        turns = list(self.session.scalars(statement))
        turns.reverse()
        return turns

    def add_turn(
        self,
        conversation_id: UUID,
        query: str,
        answer: str,
        route: str | None,
        grounded: bool,
        confidence: float,
        response_payload: dict[str, Any],
    ) -> ConversationTurnRecord:
        turn = ConversationTurnRecord(
            conversation_id=conversation_id,
            query=query,
            answer=answer,
            route=route,
            grounded=grounded,
            confidence=confidence,
            response_payload=response_payload,
        )
        conversation = self.session.get(ConversationRecord, conversation_id)
        if conversation is not None:
            conversation.updated_at = datetime.now(timezone.utc)
            if not conversation.title:
                conversation.title = query[:80]
            self.session.add(conversation)

        self.session.add(turn)
        self.session.commit()
        self.session.refresh(turn)
        return turn

    def update_memory(
        self,
        conversation_id: UUID,
        memory_summary: str | None,
        user_preferences: dict[str, Any] | None = None,
        active_document_id: UUID | None = None,
        query_refinement_history: list[dict[str, Any]] | None = None,
        interaction_patterns: dict[str, Any] | None = None,
        custom_instructions: str | None = None,
    ) -> None:
        conversation = self.session.get(ConversationRecord, conversation_id)
        if conversation is None:
            return

        conversation.memory_summary = memory_summary
        if user_preferences:
            conversation.user_preferences = {
                **(conversation.user_preferences or {}),
                **user_preferences,
            }
        if active_document_id is not None:
            conversation.active_document_id = active_document_id
        if query_refinement_history is not None:
            conversation.query_refinement_history = query_refinement_history
        if interaction_patterns is not None:
            conversation.interaction_patterns = interaction_patterns
        if custom_instructions is not None:
            conversation.custom_instructions = custom_instructions
        self.session.add(conversation)
        self.session.commit()

    def add_checkpoint(
        self,
        conversation_id: UUID,
        node_name: str,
        status: str,
        state_payload: dict[str, Any],
        turn_id: UUID | None = None,
    ) -> None:
        checkpoint = ConversationCheckpointRecord(
            conversation_id=conversation_id,
            turn_id=turn_id,
            node_name=node_name,
            status=status,
            state_payload=state_payload,
        )
        self.session.add(checkpoint)
        self.session.commit()

    def upsert_document_access(
        self,
        conversation_id: UUID,
        document_id: UUID,
        metadata_json: dict[str, Any] | None = None,
    ) -> DocumentAccessRecord:
        statement = select(DocumentAccessRecord).where(
            DocumentAccessRecord.conversation_id == conversation_id,
            DocumentAccessRecord.document_id == document_id,
        )
        record = self.session.scalar(statement)
        now = datetime.now(timezone.utc)
        if record is None:
            record = DocumentAccessRecord(
                conversation_id=conversation_id,
                document_id=document_id,
                access_count=1,
                last_accessed_at=now,
                metadata_json=metadata_json or {},
            )
        else:
            record.access_count += 1
            record.last_accessed_at = now
            record.metadata_json = {
                **(record.metadata_json or {}),
                **(metadata_json or {}),
            }
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)
        return record
