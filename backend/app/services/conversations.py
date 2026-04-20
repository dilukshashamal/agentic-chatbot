from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.db.models import ConversationCheckpointRecord, ConversationRecord, ConversationTurnRecord


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
