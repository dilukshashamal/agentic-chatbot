from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterator

from alembic import command
from alembic.config import Config
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


def _alembic_config(settings: Settings | None = None) -> Config:
    backend_root = Path(__file__).resolve().parents[2]
    config = Config(str(backend_root / "alembic.ini"))
    config.set_main_option("script_location", str(backend_root / "alembic"))
    chosen_settings = settings or get_settings()
    config.set_main_option("sqlalchemy.url", chosen_settings.database_url.replace("%", "%%"))
    return config


def init_database(settings: Settings | None = None) -> None:
    """Apply pending Alembic migrations during startup."""
    command.upgrade(_alembic_config(settings), "head")
