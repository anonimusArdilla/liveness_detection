"""Capture-session lifecycle, progressive challenge disclosure, and transient in-memory buffer management."""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import numpy as np

from app.core.config import Settings
from app.core.errors import AppError, ErrorCode
from app.models.session import CaptureSessionRecord, ChallengeStepRecord, utc_now
from app.repositories.session_repository import SessionRepository
from app.schemas.liveness import (
    ChallengeInstructionSchema,
    ChallengeType,
    CreateSessionResponse,
    CurrentChallengeResponse,
    LivenessOutcome,
    PassiveSpoofReportSchema,
    QualityIssue,
    SessionResultResponse,
)
from app.services.challenge import ActiveChallengeService
from app.vision.face_landmarks import BoundingBox, FaceAnalysis

TERMINAL_OUTCOMES: frozenset[str] = frozenset(
    {
        LivenessOutcome.QUALITY_FAILED.value,
        LivenessOutcome.CHALLENGE_FAILED.value,
        LivenessOutcome.SPOOF_SUSPECTED.value,
        LivenessOutcome.VALIDATION_UNAVAILABLE.value,
        LivenessOutcome.VALIDATED.value,
        LivenessOutcome.EXPIRED.value,
    }
)

MAX_ATTEMPTS_PER_CHALLENGE = 8


def hash_secret(value: str) -> str:
    """Compute SHA-256 hex digest of a session nonce or access token."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def compute_verification_signature(
    session_id: str,
    registration_ref: str | None,
    outcome: str,
    completed_at: datetime,
    nonce_hash: str,
) -> str:
    """Compute deterministic integrity digest binding session outcome to registration reference."""
    payload = f"{session_id}|{registration_ref or ''}|{outcome}|{completed_at.isoformat()}|{nonce_hash}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class SessionTransientBuffers:
    """Bounded in-memory holder for representative frames of an active session.

    Never persisted to disk as raw video or multi-frame dumps. Explicitly purged
    upon session completion, failure, or expiration.
    """

    expires_at: datetime
    frontal_frame: np.ndarray | None = None
    sampled_frames: list[tuple[np.ndarray, BoundingBox]] = field(default_factory=list)
    active_challenge_id: str | None = None
    step_history: list[tuple[int, FaceAnalysis]] = field(default_factory=list)

    def clear(self) -> None:
        if self.frontal_frame is not None:
            del self.frontal_frame
            self.frontal_frame = None
        for frame, _ in self.sampled_frames:
            del frame
        self.sampled_frames.clear()
        self.step_history.clear()
        self.active_challenge_id = None

    def get_step_history(self, challenge_id: str) -> list[tuple[int, FaceAnalysis]]:
        if self.active_challenge_id != challenge_id:
            self.step_history.clear()
            self.active_challenge_id = challenge_id
        return list(self.step_history)

    def append_step_history(
        self,
        challenge_id: str,
        analyses: list[tuple[int, FaceAnalysis]],
        max_history: int = 8,
    ) -> None:
        if self.active_challenge_id != challenge_id:
            self.step_history.clear()
            self.active_challenge_id = challenge_id
        self.step_history.extend(analyses)
        if len(self.step_history) > max_history:
            self.step_history = self.step_history[-max_history:]

    def clear_step_history(self) -> None:
        self.step_history.clear()
        self.active_challenge_id = None


class TransientSessionBufferStore:
    """Thread-safe in-memory store for transient session frames with automatic expiration cleanup."""

    def __init__(self, max_sampled_frames: int = 5) -> None:
        self.max_sampled_frames = max_sampled_frames
        self._lock = threading.Lock()
        self._buffers: dict[str, SessionTransientBuffers] = {}

    def init_session(self, session_id: str, expires_at: datetime) -> None:
        with self._lock:
            self._purge_expired_locked()
            self._buffers[session_id] = SessionTransientBuffers(expires_at=expires_at)

    def record_challenge_frame(
        self,
        session_id: str,
        bgr_frame: np.ndarray,
        bbox: BoundingBox,
        is_frontal: bool = False,
    ) -> None:
        with self._lock:
            self._purge_expired_locked()
            buf = self._buffers.get(session_id)
            if buf is None:
                return
            if is_frontal or buf.frontal_frame is None:
                buf.frontal_frame = bgr_frame.copy()
            if len(buf.sampled_frames) < self.max_sampled_frames:
                buf.sampled_frames.append((bgr_frame.copy(), bbox))

    def pop_buffers(self, session_id: str) -> SessionTransientBuffers | None:
        with self._lock:
            self._purge_expired_locked()
            return self._buffers.pop(session_id, None)

    def purge(self, session_id: str) -> None:
        with self._lock:
            buf = self._buffers.pop(session_id, None)
            if buf is not None:
                buf.clear()

    def get_step_history(self, session_id: str, challenge_id: str) -> list[tuple[int, FaceAnalysis]]:
        with self._lock:
            buf = self._buffers.get(session_id)
            if buf is None:
                return []
            return buf.get_step_history(challenge_id)

    def append_step_history(
        self,
        session_id: str,
        challenge_id: str,
        analyses: list[tuple[int, FaceAnalysis]],
        max_history: int = 8,
    ) -> None:
        with self._lock:
            buf = self._buffers.get(session_id)
            if buf is not None:
                buf.append_step_history(challenge_id, analyses, max_history)

    def clear_step_history(self, session_id: str) -> None:
        with self._lock:
            buf = self._buffers.get(session_id)
            if buf is not None:
                buf.clear_step_history()

    def _purge_expired_locked(self) -> None:
        now = utc_now()
        expired_ids = [sid for sid, buf in self._buffers.items() if ensure_utc(buf.expires_at) <= now]
        for sid in expired_ids:
            buf = self._buffers.pop(sid, None)
            if buf is not None:
                buf.clear()


def ensure_utc(dt: datetime) -> datetime:
    """Ensure a datetime loaded from SQLite or PostgreSQL is timezone-aware UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


class CaptureSessionService:
    """Manages session creation, state transitions, nonce verification, and challenge progression."""

    def __init__(
        self,
        settings: Settings,
        challenge_service: ActiveChallengeService,
        buffer_store: TransientSessionBufferStore,
    ) -> None:
        self.settings = settings
        self.challenge_service = challenge_service
        self.buffer_store = buffer_store

    def create_session(
        self,
        repo: SessionRepository,
        registration_ref: str | None = None,
    ) -> CreateSessionResponse:
        """Create a new short-lived session with a one-time nonce and randomized challenge sequence."""
        session_id = str(uuid.uuid4())
        raw_nonce = secrets.token_urlsafe(32)
        nonce_hash = hash_secret(raw_nonce)

        now = utc_now()
        expires_at = now + timedelta(seconds=self.settings.session_ttl_seconds)
        challenge_specs = self.challenge_service.generate_challenge_sequence()

        session_record = CaptureSessionRecord(
            session_id=session_id,
            registration_ref=registration_ref,
            nonce_hash=nonce_hash,
            outcome=LivenessOutcome.PENDING.value,
            created_at=now,
            expires_at=expires_at,
            current_step_index=0,
            total_steps=len(challenge_specs),
            last_frame_timestamp_ms=0,
            reason="Session initialized and awaiting active challenges.",
        )

        challenge_records = [
            ChallengeStepRecord(
                challenge_id=str(uuid.uuid4()),
                session_id=session_id,
                step_index=idx,
                challenge_type=ctype.value,
                instruction=instruction,
                completed=False,
                attempts=0,
            )
            for idx, (ctype, instruction) in enumerate(challenge_specs)
        ]

        persisted = repo.create_session(session_record, challenge_records)
        self.buffer_store.init_session(session_id, expires_at)

        first_step = persisted.challenges[0]
        return CreateSessionResponse(
            session_id=session_id,
            session_nonce=raw_nonce,
            outcome=LivenessOutcome.PENDING,
            expires_at=ensure_utc(persisted.expires_at),
            total_challenges=persisted.total_steps,
            current_challenge=self._to_instruction_schema(first_step, persisted.total_steps),
        )

    def get_active_session_or_raise(
        self,
        repo: SessionRepository,
        session_id: str,
        session_nonce: str | None = None,
    ) -> CaptureSessionRecord:
        """Retrieve a session, enforce expiration and terminal state rules, and verify nonce if supplied."""
        record = repo.get_session(session_id)
        if record is None:
            raise AppError(
                code=ErrorCode.SESSION_NOT_FOUND,
                message="Capture session not found.",
                status_code=404,
            )

        now = utc_now()
        if record.outcome == LivenessOutcome.PENDING.value and ensure_utc(record.expires_at) <= now:
            record.outcome = LivenessOutcome.EXPIRED.value
            record.completed_at = now
            record.reason = "Capture session expired before completion."
            repo.save(record)
            self.buffer_store.purge(session_id)

        if record.outcome == LivenessOutcome.EXPIRED.value:
            raise AppError(
                code=ErrorCode.SESSION_EXPIRED,
                message="Capture session has expired. Please start a new session.",
                status_code=410,
            )

        if record.outcome in TERMINAL_OUTCOMES:
            raise AppError(
                code=ErrorCode.SESSION_INVALID_STATE,
                message=f"Session is already finalized with outcome '{record.outcome}' and cannot be reused.",
                status_code=409,
            )

        if session_nonce is not None:
            candidate_hash = hash_secret(session_nonce)
            if not secrets.compare_digest(candidate_hash, record.nonce_hash):
                raise AppError(
                    code=ErrorCode.SESSION_NONCE_MISMATCH,
                    message="Invalid session nonce.",
                    status_code=403,
                )

        return record

    def get_current_challenge(
        self,
        repo: SessionRepository,
        session_id: str,
    ) -> CurrentChallengeResponse:
        """Return only the current challenge instruction (progressive disclosure)."""
        record = self.get_active_session_or_raise(repo, session_id)
        current_step = self._current_step(record)
        return CurrentChallengeResponse(
            session_id=record.session_id,
            outcome=LivenessOutcome(record.outcome),
            expires_at=ensure_utc(record.expires_at),
            all_challenges_completed=current_step is None,
            current_challenge=(
                self._to_instruction_schema(current_step, record.total_steps)
                if current_step is not None
                else None
            ),
        )

    def build_session_result(
        self,
        repo: SessionRepository,
        session_id: str,
        photo_access_token: str | None = None,
    ) -> SessionResultResponse:
        """Return the current or finalized outcome of a session."""
        record = repo.get_session(session_id)
        if record is None:
            raise AppError(
                code=ErrorCode.SESSION_NOT_FOUND,
                message="Capture session not found.",
                status_code=404,
            )

        now = utc_now()
        if record.outcome == LivenessOutcome.PENDING.value and ensure_utc(record.expires_at) <= now:
            record.outcome = LivenessOutcome.EXPIRED.value
            record.completed_at = now
            record.reason = "Capture session expired before completion."
            repo.save(record)
            self.buffer_store.purge(session_id)

        passive_report: PassiveSpoofReportSchema | None = None
        if record.aggregate_real_score is not None and record.min_frame_real_score is not None:
            passive_report = PassiveSpoofReportSchema(
                passed=record.passive_spoof_passed,
                aggregate_real_score=record.aggregate_real_score,
                min_frame_real_score=record.min_frame_real_score,
                frames_evaluated=record.frames_evaluated,
                threshold_applied=self.settings.spoof_real_score_threshold,
                model_name=record.passive_model_name or "MiniFASNet",
            )

        issues_raw = json.loads(record.quality_issues_json or "[]")
        issues = [QualityIssue(**item) for item in issues_raw]

        return SessionResultResponse(
            session_id=record.session_id,
            registration_ref=record.registration_ref,
            outcome=LivenessOutcome(record.outcome),
            validated=record.outcome == LivenessOutcome.VALIDATED.value,
            created_at=ensure_utc(record.created_at),
            expires_at=ensure_utc(record.expires_at),
            completed_at=ensure_utc(record.completed_at) if record.completed_at else None,
            reason=record.reason,
            quality_passed=record.quality_passed,
            challenges_passed=record.challenges_passed,
            passive_spoof_passed=record.passive_spoof_passed,
            passive_report=passive_report,
            quality_issues=issues,
            photo_available=bool(record.photo_storage_key),
            photo_access_token=photo_access_token,
            verification_signature=record.verification_signature,
        )

    @staticmethod
    def _current_step(record: CaptureSessionRecord) -> ChallengeStepRecord | None:
        for step in record.challenges:
            if not step.completed and step.step_index == record.current_step_index:
                return step
        return None

    @staticmethod
    def _to_instruction_schema(step: ChallengeStepRecord, total_steps: int) -> ChallengeInstructionSchema:
        return ChallengeInstructionSchema(
            challenge_id=step.challenge_id,
            step_index=step.step_index,
            total_steps=total_steps,
            challenge_type=ChallengeType(step.challenge_type),
            instruction=step.instruction,
            completed=step.completed,
        )
