"""Database engine and session plumbing (SQLite locally, PostgreSQL in Docker)."""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings, running_on_vercel


class Base(DeclarativeBase):
    """Declarative base for every ORM model."""


def _resolve_url() -> str:
    settings = get_settings()
    url = settings.database_url

    # Serverless filesystems are read-only apart from /tmp. Pointing SQLite at
    # the project directory there produces an unhelpful "unable to open
    # database file" at import time, so rescue the obvious mistake.
    if running_on_vercel() and url.startswith("sqlite") and "/tmp" not in url:
        return "sqlite:////tmp/maidan.db"
    return url


def _build_engine():
    url = _resolve_url()
    kwargs: dict = {"pool_pre_ping": True, "future": True}
    if url.startswith("sqlite"):
        # check_same_thread=False is required because FastAPI may touch the
        # session from a different thread than the one that created it.
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kwargs)


engine = _build_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create tables. Safe to call repeatedly."""
    from app import models  # noqa: F401  (import registers the mappers)

    Base.metadata.create_all(bind=engine)
