"""SQLAlchemy 2.0 ORM models for capture sessions, challenges, and validation outcomes."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    """Return timezone-aware UTC timestamp."""
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class CaptureSessionRecord(Base):
    """Persisted metadata and state for a facial liveness capture session.

    Note on privacy:
    - Raw video streams and intermediate challenge frames are NEVER persisted here.
    - Session nonces and photo retrieval tokens are stored only as SHA-256 digests.
    - The internal photo_storage_key is populated ONLY when outcome == 'VALIDATED'
      and is never returned directly in API responses.
    """

    __tablename__ = "capture_sessions"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    registration_ref: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    nonce_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    outcome: Mapped[str] = mapped_column(String(32), nullable=False, default="PENDING", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    current_step_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_steps: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    last_frame_timestamp_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    quality_passed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    challenges_passed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    passive_spoof_passed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    aggregate_real_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    min_frame_real_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    frames_evaluated: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    passive_model_name: Mapped[str | None] = mapped_column(String(128), nullable=True)

    reason: Mapped[str] = mapped_column(
        Text, nullable=False, default="Session initialized and awaiting active challenges."
    )
    quality_issues_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")

    photo_storage_key: Mapped[str | None] = mapped_column(String(256), nullable=True)
    photo_token_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    verification_signature: Mapped[str | None] = mapped_column(String(64), nullable=True)

    challenges: Mapped[list[ChallengeStepRecord]] = relationship(
        "ChallengeStepRecord",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChallengeStepRecord.step_index",
    )


class ChallengeStepRecord(Base):
    """Individual randomized challenge step belonging to a capture session."""

    __tablename__ = "challenge_steps"

    challenge_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("capture_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    step_index: Mapped[int] = mapped_column(Integer, nullable=False)
    challenge_type: Mapped[str] = mapped_column(String(32), nullable=False)
    instruction: Mapped[str] = mapped_column(String(256), nullable=False)
    completed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    session: Mapped[CaptureSessionRecord] = relationship("CaptureSessionRecord", back_populates="challenges")
