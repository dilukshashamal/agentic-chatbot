from __future__ import annotations

from functools import lru_cache
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings
from app.db.models import Base


@lru_cache
def get_engine():
    settings = get_settings()
    return create_engine(settings.database_url, pool_pre_ping=True)


@lru_cache
def _get_session_factory():
    return sessionmaker(bind=get_engine(), autoflush=False, autocommit=False, expire_on_commit=False)


def get_db_session() -> Iterator[Session]:
    session = _get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def new_session() -> Session:
    return _get_session_factory()()


COMPATIBILITY_MIGRATIONS: tuple[str, ...] = (
    """
    ALTER TABLE IF EXISTS documents
    ADD COLUMN IF NOT EXISTS error_message TEXT
    """,
    """
    ALTER TABLE IF EXISTS documents
    ADD COLUMN IF NOT EXISTS page_count INTEGER
    """,
    """
    ALTER TABLE IF EXISTS documents
    ADD COLUMN IF NOT EXISTS chunk_count INTEGER
    """,
    """
    ALTER TABLE IF EXISTS conversations
    ADD COLUMN IF NOT EXISTS memory_summary TEXT
    """,
    """
    ALTER TABLE IF EXISTS conversations
    ADD COLUMN IF NOT EXISTS user_preferences JSON DEFAULT '{}'::json
    """,
    """
    ALTER TABLE IF EXISTS conversations
    ADD COLUMN IF NOT EXISTS interaction_patterns JSON DEFAULT '{}'::json
    """,
    """
    ALTER TABLE IF EXISTS conversations
    ADD COLUMN IF NOT EXISTS query_refinement_history JSON DEFAULT '[]'::json
    """,
    """
    ALTER TABLE IF EXISTS conversations
    ADD COLUMN IF NOT EXISTS custom_instructions TEXT
    """,
    """
    ALTER TABLE IF EXISTS conversations
    ADD COLUMN IF NOT EXISTS active_document_id UUID
    """,
    """
    CREATE INDEX IF NOT EXISTS ix_conversations_active_document_id
    ON conversations (active_document_id)
    """,
    """
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1
            FROM pg_constraint
            WHERE conname = 'fk_conversations_active_document_id_documents'
        ) THEN
            ALTER TABLE conversations
            ADD CONSTRAINT fk_conversations_active_document_id_documents
            FOREIGN KEY (active_document_id) REFERENCES documents(id) ON DELETE SET NULL;
        END IF;
    END
    $$;
    """,
    """
    ALTER TABLE IF EXISTS conversation_turns
    ADD COLUMN IF NOT EXISTS route VARCHAR(64)
    """,
    """
    ALTER TABLE IF EXISTS conversation_turns
    ADD COLUMN IF NOT EXISTS grounded BOOLEAN DEFAULT FALSE
    """,
    """
    ALTER TABLE IF EXISTS conversation_turns
    ADD COLUMN IF NOT EXISTS confidence DOUBLE PRECISION DEFAULT 0.0
    """,
    """
    ALTER TABLE IF EXISTS conversation_turns
    ADD COLUMN IF NOT EXISTS response_payload JSON DEFAULT '{}'::json
    """,
)


def _run_compatibility_migrations() -> None:
    engine = get_engine()
    with engine.begin() as connection:
        for statement in COMPATIBILITY_MIGRATIONS:
            connection.execute(text(statement))


def init_database(settings: Settings | None = None) -> None:
    engine = get_engine()
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.create_all(bind=connection)
    _run_compatibility_migrations()
