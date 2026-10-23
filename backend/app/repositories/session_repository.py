"""Repository for persisting and querying capture sessions and challenge steps."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models.session import CaptureSessionRecord, ChallengeStepRecord


class SessionRepository:
    """Data-access layer for CaptureSessionRecord and ChallengeStepRecord."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create_session(
        self,
        session_record: CaptureSessionRecord,
        challenge_records: list[ChallengeStepRecord],
    ) -> CaptureSessionRecord:
        """Persist a new capture session and its ordered challenge sequence."""
        session_record.challenges = challenge_records
        self.db.add(session_record)
        self.db.commit()
        self.db.refresh(session_record)
        return session_record

    def get_session(self, session_id: str) -> CaptureSessionRecord | None:
        """Fetch a capture session and eagerly load its ordered challenges."""
        stmt = (
            select(CaptureSessionRecord)
            .where(CaptureSessionRecord.session_id == session_id)
            .options(selectinload(CaptureSessionRecord.challenges))
        )
        return self.db.scalar(stmt)

    def save(self, session_record: CaptureSessionRecord) -> CaptureSessionRecord:
        """Commit updates to an existing capture session record."""
        self.db.add(session_record)
        self.db.commit()
        self.db.refresh(session_record)
        return session_record
