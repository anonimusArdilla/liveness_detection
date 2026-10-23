"""Liveness pipeline coordinating quality checks, active challenges, passive PAD, and persistence."""

from __future__ import annotations

import json
import secrets

import numpy as np

from app.core.config import Settings
from app.core.errors import AppError, ErrorCode
from app.core.logging import get_logger
from app.models.session import utc_now
from app.repositories.session_repository import SessionRepository
from app.schemas.liveness import (
    ChallengeType,
    CompleteSessionRequest,
    LivenessOutcome,
    QualityReportSchema,
    SessionResultResponse,
    SubmitChallengeFramesRequest,
    SubmitChallengeFramesResponse,
)
from app.services.capture_session import (
    MAX_ATTEMPTS_PER_CHALLENGE,
    CaptureSessionService,
    compute_verification_signature,
    hash_secret,
)
from app.services.challenge import ActiveChallengeService
from app.services.image_quality import ImageQualityService, decode_base64_image
from app.storage.base import PhotoStorageProvider
from app.vision.anti_spoofing import AntiSpoofingProvider
from app.vision.face_detector import FaceDetectorProvider
from app.vision.face_landmarks import FaceAnalysis

logger = get_logger(__name__)


class LivenessPipelineService:
    """Orchestrates face detection, image quality, active challenges, and passive anti-spoofing."""

    def __init__(
        self,
        settings: Settings,
        session_service: CaptureSessionService,
        quality_service: ImageQualityService,
        challenge_service: ActiveChallengeService,
        face_detector: FaceDetectorProvider,
        anti_spoofing: AntiSpoofingProvider,
        photo_storage: PhotoStorageProvider,
    ) -> None:
        self.settings = settings
        self.session_service = session_service
        self.quality_service = quality_service
        self.challenge_service = challenge_service
        self.face_detector = face_detector
        self.anti_spoofing = anti_spoofing
        self.photo_storage = photo_storage

    def is_ready(self) -> bool:
        """Return True only when face detector, anti-spoofing models, and photo storage are ready."""
        return (
            self.face_detector.status().ready
            and self.anti_spoofing.is_ready()
            and self.photo_storage.is_ready()
        )

    def submit_challenge_frames(
        self,
        repo: SessionRepository,
        session_id: str,
        request: SubmitChallengeFramesRequest,
    ) -> SubmitChallengeFramesResponse:
        """Validate a frame batch for the current active challenge step."""
        record = self.session_service.get_active_session_or_raise(
            repo=repo,
            session_id=session_id,
            session_nonce=request.session_nonce,
        )

        # Fail closed if vision models are not ready
        if not self.face_detector.status().ready or not self.anti_spoofing.is_ready():
            record.outcome = LivenessOutcome.VALIDATION_UNAVAILABLE.value
            record.completed_at = utc_now()
            record.reason = "Required computer-vision or anti-spoofing models are unavailable."
            repo.save(record)
            self.session_service.buffer_store.purge(session_id)
            raise AppError(
                code=ErrorCode.SERVICE_UNAVAILABLE,
                message=record.reason,
                status_code=503,
            )

        if (
            len(request.frames) < self.settings.min_frames_per_batch
            or len(request.frames) > self.settings.max_frames_per_batch
        ):
            raise AppError(
                code=ErrorCode.VALIDATION_ERROR,
                message=(
                    f"Frame batch must contain between {self.settings.min_frames_per_batch} "
                    f"and {self.settings.max_frames_per_batch} frames."
                ),
                status_code=400,
            )

        current_step = self.session_service._current_step(record)
        if current_step is None or current_step.challenge_id != request.challenge_id:
            raise AppError(
                code=ErrorCode.CHALLENGE_MISMATCH,
                message="Submitted challenge_id does not match the current active challenge step.",
                status_code=409,
            )

        # Validate strictly increasing frame timestamps
        new_last_ts = self.challenge_service.validate_frame_timestamps(
            timestamps_ms=[f.timestamp_ms for f in request.frames],
            last_session_timestamp_ms=record.last_frame_timestamp_ms,
        )
        record.last_frame_timestamp_ms = new_last_ts
        current_step.attempts += 1

        challenge_type = ChallengeType(current_step.challenge_type)
        require_centered = challenge_type == ChallengeType.CENTER_FACE

        decoded_frames: list[np.ndarray] = []
        valid_analyses: list[tuple[int, FaceAnalysis]] = []
        latest_quality_report: QualityReportSchema | None = None

        try:
            for idx, frame_input in enumerate(request.frames):
                bgr = decode_base64_image(frame_input.image_base64)
                decoded_frames.append(bgr)
                faces = self.face_detector.detect_faces(bgr)
                q_report = self.quality_service.evaluate_frame(
                    bgr_image=bgr,
                    faces=faces,
                    require_centered=require_centered,
                )
                latest_quality_report = q_report
                if q_report.passed and len(faces) == 1:
                    valid_analyses.append((idx, faces[0]))

            assert latest_quality_report is not None

            # If no frames in the batch passed basic image quality
            if not valid_analyses:
                record.quality_issues_json = json.dumps(
                    [issue.model_dump() for issue in latest_quality_report.issues]
                )
                first_issue_msg = (
                    latest_quality_report.issues[0].message
                    if latest_quality_report.issues
                    else "Image quality requirements were not met."
                )
                if current_step.attempts >= MAX_ATTEMPTS_PER_CHALLENGE:
                    record.outcome = LivenessOutcome.QUALITY_FAILED.value
                    record.completed_at = utc_now()
                    record.reason = f"Maximum attempts exceeded due to image quality: {first_issue_msg}"
                    repo.save(record)
                    self.session_service.buffer_store.purge(session_id)
                else:
                    repo.save(record)

                return SubmitChallengeFramesResponse(
                    session_id=session_id,
                    outcome=LivenessOutcome(record.outcome),
                    challenge_passed=False,
                    step_index=current_step.step_index,
                    total_steps=record.total_steps,
                    feedback_message=first_issue_msg,
                    quality_report=latest_quality_report,
                    all_challenges_completed=False,
                    next_challenge=self.session_service._to_instruction_schema(
                        current_step, record.total_steps
                    )
                    if record.outcome == LivenessOutcome.PENDING.value
                    else None,
                )

            # Evaluate active challenge pose requirements across quality-passing frames
            eval_result = self.challenge_service.evaluate_challenge_batch(
                challenge_type=challenge_type,
                valid_face_analyses=valid_analyses,
            )

            if not eval_result.passed:
                if current_step.attempts >= MAX_ATTEMPTS_PER_CHALLENGE:
                    record.outcome = LivenessOutcome.CHALLENGE_FAILED.value
                    record.completed_at = utc_now()
                    record.reason = (
                        f"Failed to complete active challenge '{challenge_type.value}' "
                        f"within maximum allowed attempts."
                    )
                    repo.save(record)
                    self.session_service.buffer_store.purge(session_id)
                else:
                    repo.save(record)

                return SubmitChallengeFramesResponse(
                    session_id=session_id,
                    outcome=LivenessOutcome(record.outcome),
                    challenge_passed=False,
                    step_index=current_step.step_index,
                    total_steps=record.total_steps,
                    feedback_message=eval_result.feedback_message,
                    quality_report=latest_quality_report,
                    all_challenges_completed=False,
                    next_challenge=self.session_service._to_instruction_schema(
                        current_step, record.total_steps
                    )
                    if record.outcome == LivenessOutcome.PENDING.value
                    else None,
                )

            # Challenge step succeeded! Cache representative frame in transient memory
            best_idx = (
                eval_result.best_frame_index
                if eval_result.best_frame_index is not None
                else valid_analyses[-1][0]
            )
            best_analysis = next(a for i, a in valid_analyses if i == best_idx)
            self.session_service.buffer_store.record_challenge_frame(
                session_id=session_id,
                bgr_frame=decoded_frames[best_idx],
                bbox=best_analysis.bounding_box,
                is_frontal=(challenge_type == ChallengeType.CENTER_FACE),
            )

            current_step.completed = True
            current_step.completed_at = utc_now()
            record.quality_passed = True
            record.quality_issues_json = "[]"
            record.current_step_index += 1

            all_done = record.current_step_index >= record.total_steps
            if all_done:
                record.challenges_passed = True
                record.reason = "All active challenges passed. Ready for final anti-spoofing verification."

            repo.save(record)
            next_step = self.session_service._current_step(record)

            return SubmitChallengeFramesResponse(
                session_id=session_id,
                outcome=LivenessOutcome(record.outcome),
                challenge_passed=True,
                step_index=current_step.step_index,
                total_steps=record.total_steps,
                feedback_message=eval_result.feedback_message,
                quality_report=latest_quality_report,
                all_challenges_completed=all_done,
                next_challenge=self.session_service._to_instruction_schema(next_step, record.total_steps)
                if next_step is not None
                else None,
            )
        finally:
            for arr in decoded_frames:
                del arr
            decoded_frames.clear()

    def complete_session(
        self,
        repo: SessionRepository,
        session_id: str,
        request: CompleteSessionRequest,
    ) -> SessionResultResponse:
        """Run final passive anti-spoofing verification, persist photo if valid, and purge buffers."""
        record = self.session_service.get_active_session_or_raise(
            repo=repo,
            session_id=session_id,
            session_nonce=request.session_nonce,
        )

        transient = self.session_service.buffer_store.pop_buffers(session_id)
        try:
            now = utc_now()
            record.completed_at = now

            # 1. Verify all active challenges were completed
            if not record.challenges_passed or any(not c.completed for c in record.challenges):
                record.outcome = LivenessOutcome.CHALLENGE_FAILED.value
                record.reason = "Capture finalized before all active challenges were completed."
                record.verification_signature = compute_verification_signature(
                    session_id=record.session_id,
                    registration_ref=record.registration_ref,
                    outcome=record.outcome,
                    completed_at=now,
                    nonce_hash=record.nonce_hash,
                )
                repo.save(record)
                return self.session_service.build_session_result(repo, session_id)

            # 2. Verify model readiness (fail closed)
            if not self.anti_spoofing.is_ready() or not self.photo_storage.is_ready():
                record.outcome = LivenessOutcome.VALIDATION_UNAVAILABLE.value
                record.reason = "Anti-spoofing model or storage service is unavailable. Failing closed."
                record.verification_signature = compute_verification_signature(
                    session_id=record.session_id,
                    registration_ref=record.registration_ref,
                    outcome=record.outcome,
                    completed_at=now,
                    nonce_hash=record.nonce_hash,
                )
                repo.save(record)
                return self.session_service.build_session_result(repo, session_id)

            if transient is None or not transient.sampled_frames or transient.frontal_frame is None:
                record.outcome = LivenessOutcome.QUALITY_FAILED.value
                record.reason = "No validated session frames available for anti-spoofing analysis."
                record.verification_signature = compute_verification_signature(
                    session_id=record.session_id,
                    registration_ref=record.registration_ref,
                    outcome=record.outcome,
                    completed_at=now,
                    nonce_hash=record.nonce_hash,
                )
                repo.save(record)
                return self.session_service.build_session_result(repo, session_id)

            # 3. Execute real ONNX MiniFASNet passive anti-spoofing inference across sampled frames
            try:
                pad_eval = self.anti_spoofing.evaluate_frames(
                    frames_with_boxes=transient.sampled_frames,
                    real_score_threshold=self.settings.spoof_real_score_threshold,
                    min_frame_score_threshold=self.settings.spoof_min_frame_score,
                )
            except Exception as exc:
                logger.error("Passive anti-spoofing inference failed for session %s: %s", session_id, exc)
                record.outcome = LivenessOutcome.VALIDATION_UNAVAILABLE.value
                record.reason = "Passive anti-spoofing model inference failed. Failing closed."
                record.verification_signature = compute_verification_signature(
                    session_id=record.session_id,
                    registration_ref=record.registration_ref,
                    outcome=record.outcome,
                    completed_at=now,
                    nonce_hash=record.nonce_hash,
                )
                repo.save(record)
                return self.session_service.build_session_result(repo, session_id)

            record.aggregate_real_score = pad_eval.aggregate_real_score
            record.min_frame_real_score = pad_eval.min_frame_real_score
            record.frames_evaluated = pad_eval.frames_evaluated
            record.passive_model_name = pad_eval.model_name
            record.passive_spoof_passed = pad_eval.passed

            if not pad_eval.passed:
                record.outcome = LivenessOutcome.SPOOF_SUSPECTED.value
                record.reason = (
                    "Passive anti-spoofing check failed: potential presentation attack "
                    "(printed photo or screen replay) detected."
                )
                record.verification_signature = compute_verification_signature(
                    session_id=record.session_id,
                    registration_ref=record.registration_ref,
                    outcome=record.outcome,
                    completed_at=now,
                    nonce_hash=record.nonce_hash,
                )
                repo.save(record)
                return self.session_service.build_session_result(repo, session_id)

            # 4. All checks passed! Persist accepted frontal photograph & generate protected token
            storage_key = self.photo_storage.save_validated_photo(
                session_id=session_id,
                bgr_image=transient.frontal_frame,
            )
            raw_photo_token = secrets.token_urlsafe(32)
            record.photo_storage_key = storage_key
            record.photo_token_hash = hash_secret(raw_photo_token)
            record.outcome = LivenessOutcome.VALIDATED.value
            record.reason = (
                "Facial liveness validated: image quality, randomized active challenges, "
                "and passive anti-spoofing checks all passed."
            )
            record.verification_signature = compute_verification_signature(
                session_id=record.session_id,
                registration_ref=record.registration_ref,
                outcome=record.outcome,
                completed_at=now,
                nonce_hash=record.nonce_hash,
            )
            repo.save(record)

            logger.info(
                "Session %s VALIDATED (aggregate_real_score=%.4f, min_frame_score=%.4f)",
                session_id,
                pad_eval.aggregate_real_score,
                pad_eval.min_frame_real_score,
            )
            return self.session_service.build_session_result(
                repo, session_id, photo_access_token=raw_photo_token
            )
        finally:
            if transient is not None:
                transient.clear()

    def get_validated_photo_bytes(
        self,
        repo: SessionRepository,
        session_id: str,
        access_token: str,
    ) -> bytes:
        """Retrieve accepted photograph bytes only when outcome == VALIDATED and token matches."""
        record = repo.get_session(session_id)
        if record is None:
            raise AppError(
                code=ErrorCode.SESSION_NOT_FOUND,
                message="Capture session not found.",
                status_code=404,
            )

        if (
            record.outcome != LivenessOutcome.VALIDATED.value
            or not record.photo_storage_key
            or not record.photo_token_hash
        ):
            raise AppError(
                code=ErrorCode.PHOTO_NOT_FOUND,
                message="No validated photograph is available for this session.",
                status_code=404,
            )

        candidate_hash = hash_secret(access_token)
        if not secrets.compare_digest(candidate_hash, record.photo_token_hash):
            raise AppError(
                code=ErrorCode.UNAUTHORIZED_ACCESS,
                message="Invalid or missing photograph access token.",
                status_code=403,
            )

        return self.photo_storage.load_validated_photo(record.photo_storage_key)
