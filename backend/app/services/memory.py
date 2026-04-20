from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import or_, select, update
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.models import (
    ConversationRecord,
    ConversationTurnRecord,
    DocumentAccessRecord,
    KnowledgeGraphEdgeRecord,
    KnowledgeGraphNodeRecord,
    MemoryRecord,
)
from app.models.schemas import MemoryAction, MemoryHit
from app.services.documents import RetrievedChunk
from app.services.indexing import _get_embeddings

STOPWORDS = {
    "about",
    "across",
    "after",
    "also",
    "answer",
    "answers",
    "before",
    "between",
    "could",
    "document",
    "documents",
    "explain",
    "focus",
    "from",
    "have",
    "into",
    "like",
    "more",
    "please",
    "query",
    "remember",
    "search",
    "should",
    "that",
    "their",
    "them",
    "these",
    "this",
    "those",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
}
ENTITY_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z0-9_-]{3,}\b")


@dataclass(frozen=True)
class MemoryCommand:
    action: str
    target: str | None = None


class BaseMemoryProvider:
    def __init__(self, name: str, available: bool, note: str | None = None) -> None:
        self.name = name
        self.available = available
        self.note = note

    def sync_write(self, _memory: MemoryRecord | None = None) -> str | None:
        return self.note

    def sync_delete(self, _conversation_id: UUID, _target: str | None = None) -> str | None:
        return self.note


class MemoryService:
    def __init__(self, settings: Settings, session: Session) -> None:
        self.settings = settings
        self.session = session
        self.provider = self._build_provider()

    def _build_provider(self) -> BaseMemoryProvider:
        if self.settings.memory_provider == "mem0":
            available = bool(self.settings.mem0_api_key or self.settings.mem0_base_url)
            note = None if available else "Mem0 selected but not configured; local memory store remains active."
            return BaseMemoryProvider("mem0", available, note)
        if self.settings.memory_provider == "zep":
            available = bool(self.settings.zep_api_key or self.settings.zep_base_url)
            note = None if available else "Zep selected but not configured; local memory store remains active."
            return BaseMemoryProvider("zep", available, note)
        return BaseMemoryProvider("local", True, None)

    def parse_command(self, query: str) -> MemoryCommand | None:
        cleaned = query.strip()
        lowered = cleaned.lower()

        if lowered in {"forget everything", "forget all memories", "delete my memory", "delete all memory"}:
            return MemoryCommand(action="forget_all")

        patterns = [
            (r"^remember this[:\-\s]+(.+)$", "remember"),
            (r"^remember that[:\-\s]+(.+)$", "remember"),
            (r"^remember[:\-\s]+(.+)$", "remember"),
            (r"^forget this[:\-\s]+(.+)$", "forget"),
            (r"^forget that[:\-\s]+(.+)$", "forget"),
            (r"^forget[:\-\s]+(.+)$", "forget"),
        ]
        for pattern, action in patterns:
            match = re.match(pattern, cleaned, flags=re.IGNORECASE)
            if match:
                return MemoryCommand(action=action, target=match.group(1).strip())

        if lowered == "forget that":
            return MemoryCommand(action="forget_last")
        return None

    def load_short_term_context(self, conversation: ConversationRecord) -> dict[str, Any]:
        turns = sorted(conversation.turns, key=lambda turn: turn.created_at)[-self.settings.conversation_history_window :]
        short_term = [
            f"User: {turn.query} | Assistant: {turn.answer[:160]}"
            for turn in turns
        ]
        return {
            "recent_turns": turns,
            "short_term_memory": short_term,
            "active_document_id": conversation.active_document_id,
            "query_refinement_history": conversation.query_refinement_history or [],
            "custom_instructions": conversation.custom_instructions,
            "interaction_patterns": conversation.interaction_patterns or {},
        }

    def semantic_search(self, conversation_id: UUID, query: str) -> list[MemoryHit]:
        if not self.settings.enable_semantic_memory_search:
            return []

        try:
            query_embedding = _get_embeddings(self.settings).embed_query(query)
        except Exception:
            return self.keyword_search(conversation_id, query)

        distance = MemoryRecord.embedding.cosine_distance(query_embedding)
        statement = (
            select(MemoryRecord, distance.label("distance"))
            .where(
                MemoryRecord.conversation_id == conversation_id,
                MemoryRecord.is_deleted.is_(False),
                MemoryRecord.embedding.is_not(None),
            )
            .order_by(distance.asc(), MemoryRecord.updated_at.desc())
            .limit(self.settings.memory_search_k)
        )
        rows = self.session.execute(statement).all()

        hits: list[MemoryHit] = []
        for memory, raw_distance in rows:
            score = max(0.0, min(1.0, 1.0 - max(float(raw_distance or 0.0), 0.0)))
            if score < self.settings.memory_similarity_threshold:
                continue
            hits.append(
                MemoryHit(
                    id=str(memory.id),
                    memory_type=memory.memory_type,
                    content=memory.summary or memory.content,
                    score=round(score, 4),
                    metadata=memory.metadata_json or {},
                )
            )
            memory.access_count += 1
            memory.last_accessed_at = datetime.now(timezone.utc)
            self.session.add(memory)
        self.session.commit()
        return hits

    def keyword_search(self, conversation_id: UUID, query: str) -> list[MemoryHit]:
        statement = (
            select(MemoryRecord)
            .where(
                MemoryRecord.conversation_id == conversation_id,
                MemoryRecord.is_deleted.is_(False),
                or_(
                    MemoryRecord.content.ilike(f"%{query}%"),
                    MemoryRecord.summary.ilike(f"%{query}%"),
                ),
            )
            .order_by(MemoryRecord.updated_at.desc())
            .limit(self.settings.memory_search_k)
        )
        records = list(self.session.scalars(statement))
        return [
            MemoryHit(
                id=str(record.id),
                memory_type=record.memory_type,
                content=record.summary or record.content,
                score=0.6,
                metadata=record.metadata_json or {},
            )
            for record in records
        ]

    def remember(self, conversation_id: UUID, content: str, turn_id: UUID | None = None) -> list[MemoryAction]:
        memory_type = "manual_fact"
        metadata: dict[str, Any] = {"source": "remember_command"}

        lowered = content.lower()
        conversation = self.session.get(ConversationRecord, conversation_id)
        if "tone" in lowered or "style" in lowered:
            memory_type = "tone_preference"
            metadata["category"] = "tone"
            if conversation is not None:
                conversation.custom_instructions = content
                self.session.add(conversation)
        elif any(token in lowered for token in ["prefer", "always", "never", "please call me"]):
            memory_type = "preference"
            metadata["category"] = "preference"

        memory = self._upsert_memory_record(
            conversation_id=conversation_id,
            memory_type=memory_type,
            content=content,
            summary=content[:240],
            metadata_json=metadata,
            turn_id=turn_id,
        )
        provider_note = self.provider.sync_write(memory)
        self._update_knowledge_graph(conversation_id, content, [])
        actions = [
            MemoryAction(
                action="remembered",
                target=content[:80],
                detail="Stored in long-term memory.",
            )
        ]
        if provider_note:
            actions.append(MemoryAction(action="ignored", target=self.provider.name, detail=provider_note))
        self.session.commit()
        return actions

    def forget(
        self,
        conversation_id: UUID,
        target: str | None = None,
        forget_all: bool = False,
    ) -> list[MemoryAction]:
        actions: list[MemoryAction] = []
        conversation = self.session.get(ConversationRecord, conversation_id)

        if forget_all:
            self.session.execute(
                update(MemoryRecord)
                .where(MemoryRecord.conversation_id == conversation_id)
                .values(is_deleted=True, updated_at=datetime.now(timezone.utc))
            )
            self.session.execute(
                update(KnowledgeGraphNodeRecord)
                .where(KnowledgeGraphNodeRecord.conversation_id == conversation_id)
                .values(is_deleted=True, updated_at=datetime.now(timezone.utc))
            )
            self.session.execute(
                update(KnowledgeGraphEdgeRecord)
                .where(KnowledgeGraphEdgeRecord.conversation_id == conversation_id)
                .values(is_deleted=True, updated_at=datetime.now(timezone.utc))
            )
            self.session.execute(
                update(DocumentAccessRecord)
                .where(DocumentAccessRecord.conversation_id == conversation_id)
                .values(access_count=0, last_accessed_at=None, metadata_json={})
            )
            if conversation is not None:
                conversation.memory_summary = None
                conversation.user_preferences = {}
                conversation.interaction_patterns = {}
                conversation.query_refinement_history = []
                conversation.custom_instructions = None
                conversation.active_document_id = None
                self.session.add(conversation)
            actions.append(MemoryAction(action="forgotten", target="all memory", detail="Cleared conversation memory for GDPR compliance."))
        else:
            if target is None:
                target = self._last_manual_memory(conversation_id)
            if not target:
                return [MemoryAction(action="ignored", target="memory", detail="There was nothing specific to forget.")]

            query = (
                select(MemoryRecord)
                .where(
                    MemoryRecord.conversation_id == conversation_id,
                    MemoryRecord.is_deleted.is_(False),
                    or_(
                        MemoryRecord.content.ilike(f"%{target}%"),
                        MemoryRecord.summary.ilike(f"%{target}%"),
                    ),
                )
            )
            records = list(self.session.scalars(query))
            for record in records:
                record.is_deleted = True
                record.updated_at = datetime.now(timezone.utc)
                self.session.add(record)

            node_query = (
                select(KnowledgeGraphNodeRecord)
                .where(
                    KnowledgeGraphNodeRecord.conversation_id == conversation_id,
                    KnowledgeGraphNodeRecord.is_deleted.is_(False),
                    KnowledgeGraphNodeRecord.label.ilike(f"%{target}%"),
                )
            )
            for node in self.session.scalars(node_query):
                node.is_deleted = True
                self.session.add(node)

            actions.append(
                MemoryAction(
                    action="forgotten",
                    target=target[:80],
                    detail="Removed matching stored memories and related graph items.",
                )
            )

        provider_note = self.provider.sync_delete(conversation_id, target)
        if provider_note:
            actions.append(MemoryAction(action="ignored", target=self.provider.name, detail=provider_note))
        self.session.commit()
        return actions

    def update_after_response(
        self,
        conversation: ConversationRecord,
        query: str,
        response_text: str,
        route: str,
        retrieved_chunks: list[RetrievedChunk],
        user_preferences: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        history = list(conversation.query_refinement_history or [])
        history.append({"query": query, "timestamp": now.isoformat(), "route": route})
        history = history[-self.settings.query_refinement_window :]

        interaction_patterns = dict(conversation.interaction_patterns or {})
        interaction_patterns["total_queries"] = int(interaction_patterns.get("total_queries", 0)) + 1
        route_counts = dict(interaction_patterns.get("route_counts", {}))
        route_counts[route] = int(route_counts.get(route, 0)) + 1
        interaction_patterns["route_counts"] = route_counts
        interaction_patterns["last_route"] = route
        interaction_patterns["last_seen_at"] = now.isoformat()

        active_document_id = retrieved_chunks[0].document_id if retrieved_chunks else conversation.active_document_id
        if active_document_id is not None:
            for chunk in {item.document_id: item for item in retrieved_chunks}.values():
                self._record_document_access(conversation.id, chunk.document_id, chunk.source)

        if route == "memory_command":
            self.session.flush()
            return {
                "memory_summary": conversation.memory_summary,
                "active_document_id": active_document_id,
                "query_refinement_history": history,
                "interaction_patterns": interaction_patterns,
                "faq_count": 0,
            }

        summary = self._build_summary(conversation.id, query, response_text)
        self._upsert_memory_record(
            conversation_id=conversation.id,
            memory_type="conversation_summary",
            content=f"Question: {query}\nAnswer: {response_text}",
            summary=summary,
            metadata_json={"route": route, "timestamp": now.isoformat()},
        )

        faq_count = self._increment_faq(conversation.id, query)
        self._update_knowledge_graph(conversation.id, f"{query}\n{response_text}", retrieved_chunks)

        if user_preferences:
            for key, value in user_preferences.items():
                self._upsert_memory_record(
                    conversation_id=conversation.id,
                    memory_type="preference",
                    content=f"{key}: {value}",
                    summary=f"User preference for {key} is {value}.",
                    metadata_json={"preference_key": key},
                )

        self.session.flush()
        return {
            "memory_summary": summary,
            "active_document_id": active_document_id,
            "query_refinement_history": history,
            "interaction_patterns": interaction_patterns,
            "faq_count": faq_count,
        }

    def top_topics(self, conversation_id: UUID, limit: int = 5) -> list[str]:
        statement = (
            select(KnowledgeGraphNodeRecord)
            .where(
                KnowledgeGraphNodeRecord.conversation_id == conversation_id,
                KnowledgeGraphNodeRecord.is_deleted.is_(False),
            )
            .order_by(KnowledgeGraphNodeRecord.weight.desc(), KnowledgeGraphNodeRecord.label.asc())
            .limit(limit)
        )
        return [node.label for node in self.session.scalars(statement)]

    def _build_summary(self, conversation_id: UUID, query: str, response_text: str) -> str:
        statement = (
            select(ConversationTurnRecord)
            .where(ConversationTurnRecord.conversation_id == conversation_id)
            .order_by(ConversationTurnRecord.created_at.desc())
            .limit(self.settings.memory_summary_window - 1)
        )
        turns = list(self.session.scalars(statement))
        turns.reverse()
        parts = [f"Q: {turn.query} A: {turn.answer[:120]}" for turn in turns]
        parts.append(f"Q: {query} A: {response_text[:180]}")
        return " | ".join(parts)[-1500:]

    def _record_document_access(self, conversation_id: UUID, document_id: UUID, source: str) -> None:
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
                metadata_json={"source": source},
            )
        else:
            record.access_count += 1
            record.last_accessed_at = now
            record.metadata_json = {
                **(record.metadata_json or {}),
                "source": source,
            }
        self.session.add(record)

    def _increment_faq(self, conversation_id: UUID, query: str) -> int:
        normalized = re.sub(r"\s+", " ", query.strip().lower())
        statement = select(MemoryRecord).where(
            MemoryRecord.conversation_id == conversation_id,
            MemoryRecord.memory_type == "faq",
            MemoryRecord.is_deleted.is_(False),
            MemoryRecord.content == normalized,
        )
        record = self.session.scalar(statement)
        if record is None:
            record = MemoryRecord(
                conversation_id=conversation_id,
                memory_type="faq",
                scope="user",
                content=normalized,
                summary=f"FAQ: {query}",
                metadata_json={"count": 1, "original_query": query},
            )
        else:
            count = int((record.metadata_json or {}).get("count", 0)) + 1
            record.metadata_json = {
                **(record.metadata_json or {}),
                "count": count,
                "original_query": query,
            }
        self._set_embedding(record, record.summary or query)
        self.session.add(record)
        self.session.flush()
        return int((record.metadata_json or {}).get("count", 1))

    def _update_knowledge_graph(self, conversation_id: UUID, text: str, retrieved_chunks: list[RetrievedChunk]) -> None:
        labels = self._extract_entities(text)
        labels.extend(chunk.source for chunk in retrieved_chunks[:3])

        cleaned_labels: list[str] = []
        seen: set[str] = set()
        for label in labels:
            normalized = label.strip()
            key = normalized.lower()
            if not normalized or key in seen:
                continue
            seen.add(key)
            cleaned_labels.append(normalized)
        cleaned_labels = cleaned_labels[: self.settings.knowledge_graph_max_entities]

        node_ids: list[int] = []
        for label in cleaned_labels:
            node = self._upsert_graph_node(conversation_id, label)
            node_ids.append(node.id)

        for idx, source_id in enumerate(node_ids):
            for target_id in node_ids[idx + 1 :]:
                self._upsert_graph_edge(conversation_id, source_id, target_id)

    def _extract_entities(self, text: str) -> list[str]:
        entities: list[str] = []
        for match in ENTITY_PATTERN.findall(text):
            lowered = match.lower()
            if lowered in STOPWORDS or lowered.isdigit():
                continue
            entities.append(match)
        return entities

    def _upsert_graph_node(self, conversation_id: UUID, label: str) -> KnowledgeGraphNodeRecord:
        statement = select(KnowledgeGraphNodeRecord).where(
            KnowledgeGraphNodeRecord.conversation_id == conversation_id,
            KnowledgeGraphNodeRecord.label == label,
            KnowledgeGraphNodeRecord.is_deleted.is_(False),
        )
        node = self.session.scalar(statement)
        if node is None:
            node = KnowledgeGraphNodeRecord(
                conversation_id=conversation_id,
                node_type="topic",
                label=label,
                weight=1.0,
                metadata_json={},
            )
        else:
            node.weight += 1.0
        self.session.add(node)
        self.session.flush()
        return node

    def _upsert_graph_edge(self, conversation_id: UUID, source_node_id: int, target_node_id: int) -> None:
        if source_node_id == target_node_id:
            return
        left_id, right_id = sorted([source_node_id, target_node_id])
        statement = select(KnowledgeGraphEdgeRecord).where(
            KnowledgeGraphEdgeRecord.conversation_id == conversation_id,
            KnowledgeGraphEdgeRecord.source_node_id == left_id,
            KnowledgeGraphEdgeRecord.target_node_id == right_id,
            KnowledgeGraphEdgeRecord.relation == "related_to",
            KnowledgeGraphEdgeRecord.is_deleted.is_(False),
        )
        edge = self.session.scalar(statement)
        if edge is None:
            edge = KnowledgeGraphEdgeRecord(
                conversation_id=conversation_id,
                source_node_id=left_id,
                target_node_id=right_id,
                relation="related_to",
                weight=1.0,
                metadata_json={},
            )
        else:
            edge.weight += 1.0
        self.session.add(edge)

    def _last_manual_memory(self, conversation_id: UUID) -> str | None:
        statement = (
            select(MemoryRecord)
            .where(
                MemoryRecord.conversation_id == conversation_id,
                MemoryRecord.is_deleted.is_(False),
                MemoryRecord.memory_type.in_(["manual_fact", "preference", "tone_preference"]),
            )
            .order_by(MemoryRecord.created_at.desc())
            .limit(1)
        )
        record = self.session.scalar(statement)
        return record.content if record is not None else None

    def _upsert_memory_record(
        self,
        conversation_id: UUID,
        memory_type: str,
        content: str,
        summary: str,
        metadata_json: dict[str, Any],
        turn_id: UUID | None = None,
    ) -> MemoryRecord:
        statement = select(MemoryRecord).where(
            MemoryRecord.conversation_id == conversation_id,
            MemoryRecord.memory_type == memory_type,
            MemoryRecord.content == content,
            MemoryRecord.is_deleted.is_(False),
        )
        memory = self.session.scalar(statement)
        if memory is None:
            memory = MemoryRecord(
                conversation_id=conversation_id,
                turn_id=turn_id,
                memory_type=memory_type,
                scope="conversation",
                content=content,
                summary=summary,
                metadata_json=metadata_json,
                access_count=0,
            )
        else:
            memory.summary = summary
            memory.metadata_json = {
                **(memory.metadata_json or {}),
                **metadata_json,
            }
            memory.updated_at = datetime.now(timezone.utc)
        self._set_embedding(memory, summary or content)
        self.session.add(memory)
        self.session.flush()
        return memory

    def _set_embedding(self, memory: MemoryRecord, text: str) -> None:
        try:
            memory.embedding = _get_embeddings(self.settings).embed_query(text)
        except Exception:
            memory.embedding = None
