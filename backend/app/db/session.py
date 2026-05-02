from __future__ import annotations

from functools import lru_cache
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings


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


def init_database(settings: Settings | None = None) -> None:
    """Database schema is managed via Alembic migrations."""
    _ = settings
    return None
