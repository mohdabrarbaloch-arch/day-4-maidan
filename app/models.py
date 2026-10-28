"""ORM models: users and the runs they generate."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(120), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    runs: Mapped[list[Run]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Run(Base):
    """One generation — the prompt, the parameters, and what came back.

    Storing the full parameter set is what makes a run reproducible: the seed
    is saved, so the same prompt + parameters + model can be replayed exactly.
    """

    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)

    model_id: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    backend: Mapped[str] = mapped_column(String(16), nullable=False, default="local")

    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    completion: Mapped[str] = mapped_column(Text, nullable=False, default="")

    temperature: Mapped[float] = mapped_column(default=0.8)
    top_p: Mapped[float] = mapped_column(default=0.95)
    top_k: Mapped[int] = mapped_column(default=40)
    max_tokens: Mapped[int] = mapped_column(default=80)
    repetition_penalty: Mapped[float] = mapped_column(default=1.15)
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)

    tokens_generated: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    stop_reason: Mapped[str] = mapped_column(String(32), default="max_tokens")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )

    user: Mapped[User] = relationship(back_populates="runs")
