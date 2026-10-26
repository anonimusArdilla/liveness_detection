"""Unit tests for fail-closed model behavior, spoof rejection, and privacy redaction."""

from __future__ import annotations

from pathlib import Path

import pytest
from app.core.config import Settings
from app.core.errors import AppError, ErrorCode
from app.core.logging import PrivacyRedactingFilter, configure_logging
from app.repositories.database import DatabaseManager
from app.repositories.session_repository import SessionRepository
from app.schemas.liveness import (
    ChallengeInstructionSchema,
    ChallengeType,
    CompleteSessionRequest,
    FrameInputSchema,
    LivenessOutcome,
    SubmitChallengeFramesRequest,
)
from app.services.capture_session import CaptureSessionService, TransientSessionBufferStore
from app.services.challenge import ActiveChallengeService
from app.services.image_quality import ImageQualityService
from app.services.liveness_pipeline import LivenessPipelineService
from app.storage.local_storage import LocalPrivatePhotoStorage
from app.vision.face_landmarks import BoundingBox, FaceAnalysis, FacialExpression, HeadPose
from app.vision.providers.minifasnet_onnx import MiniFASNetOnnxProvider

from tests.conftest import (
    StubAntiSpoofingProvider,
    StubFaceDetector,
    make_encoded_test_frame,
)

NEUTRAL_EXPR = FacialExpression(smile=0.05, eyes_closed=0.05, eyes_open_max=0.08, brow_raise=0.1)


def _face(yaw: float = 0.0, expression: FacialExpression = NEUTRAL_EXPR) -> FaceAnalysis:
    return FaceAnalysis(
        bounding_box=BoundingBox(x=200, y=120, width=220, height=220),
        face_area_ratio=0.18,
        center_offset_ratio=0.03,
        head_pose=HeadPose(yaw=yaw, pitch=0.0, roll=0.0),
        landmarks_count=478,
        expression=expression,
    )


def _faces_for_challenge(ctype: ChallengeType) -> list[list[FaceAnalysis]]:
    """Per-frame detector output satisfying the given challenge (cycled by StubFaceDetector)."""
    if ctype == ChallengeType.TURN_LEFT:
        return [[_face(yaw=-20.0)]]
    if ctype == ChallengeType.TURN_RIGHT:
        return [[_face(yaw=20.0)]]
    if ctype == ChallengeType.SMILE:
        return [[_face(expression=FacialExpression(0.8, 0.05, 0.08, 0.1))]]
    if ctype == ChallengeType.RAISE_EYEBROWS:
        return [[_face(expression=FacialExpression(0.05, 0.05, 0.08, 0.7))]]
    if ctype == ChallengeType.BLINK:
        return [[_face()], [_face(expression=FacialExpression(0.05, 0.85, 0.9, 0.1))]]
    return [[_face()]]


def test_missing_or_corrupt_onnx_model_reports_not_ready(tmp_path: Path) -> None:
    missing_path = tmp_path / "nonexistent_model.onnx"
    provider = MiniFASNetOnnxProvider(primary_model_path=missing_path, secondary_model_path=None)
    assert provider.is_ready() is False
    assert provider.model_statuses()[0].ready is False

    corrupt_path = tmp_path / "corrupt.onnx"
    corrupt_path.write_bytes(b"not-a-valid-onnx-protobuf")
    provider_corrupt = MiniFASNetOnnxProvider(
        primary_model_path=corrupt_path,
        primary_sha256="0" * 64,
        secondary_model_path=None,
        verify_checksum=True,
    )
    assert provider_corrupt.is_ready() is False
    assert "checksum mismatch" in (provider_corrupt.model_statuses()[0].reason or "").lower()


def test_pipeline_fails_closed_when_passive_model_detects_spoof(
    test_settings: Settings,
    photo_storage: LocalPrivatePhotoStorage,
) -> None:
    db_mgr = DatabaseManager(test_settings.database_url)
    db_mgr.init_schema()
    detector = StubFaceDetector(ready=True)
    # Configure low real score (0.12 < 0.65 threshold) to simulate a printed photo / screen replay
    spoof_pad = StubAntiSpoofingProvider(ready=True, real_score=0.12)
    quality_svc = ImageQualityService(test_settings)
    challenge_svc = ActiveChallengeService(test_settings)
    buf_store = TransientSessionBufferStore()
    session_svc = CaptureSessionService(test_settings, challenge_svc, buf_store)
    pipeline = LivenessPipelineService(
        settings=test_settings,
        session_service=session_svc,
        quality_service=quality_svc,
        challenge_service=challenge_svc,
        face_detector=detector,
        anti_spoofing=spoof_pad,
        photo_storage=photo_storage,
    )

    _, b64 = make_encoded_test_frame(640, 480)
    with next(db_mgr.get_session()) as db:
        repo = SessionRepository(db)
        session = session_svc.create_session(repo, registration_ref="spoof-attempt-1")
        ts = 1000
        challenge: ChallengeInstructionSchema | None = session.current_challenge

        while challenge is not None:
            detector.set_face_sequence(_faces_for_challenge(challenge.challenge_type))
            resp = pipeline.submit_challenge_frames(
                repo=repo,
                session_id=session.session_id,
                request=SubmitChallengeFramesRequest(
                    session_nonce=session.session_nonce,
                    challenge_id=challenge.challenge_id,
                    frames=[
                        FrameInputSchema(timestamp_ms=ts + 100, image_base64=b64),
                        FrameInputSchema(timestamp_ms=ts + 200, image_base64=b64),
                    ],
                ),
            )
            ts += 300
            assert resp.challenge_passed is True
            challenge = resp.next_challenge

        final = pipeline.complete_session(
            repo=repo,
            session_id=session.session_id,
            request=CompleteSessionRequest(session_nonce=session.session_nonce),
        )
        assert final.outcome == LivenessOutcome.SPOOF_SUSPECTED
        assert final.validated is False
        assert final.photo_available is False
        assert final.photo_access_token is None
        # Verify no photograph was written to storage
        assert list(test_settings.photo_storage_dir.glob("*.jpg")) == []


def test_pipeline_fails_closed_when_inference_throws_error(
    test_settings: Settings,
    photo_storage: LocalPrivatePhotoStorage,
) -> None:
    db_mgr = DatabaseManager(test_settings.database_url)
    db_mgr.init_schema()
    detector = StubFaceDetector(ready=True)
    failing_pad = StubAntiSpoofingProvider(ready=True, raise_on_infer=True)
    quality_svc = ImageQualityService(test_settings)
    challenge_svc = ActiveChallengeService(test_settings)
    buf_store = TransientSessionBufferStore()
    session_svc = CaptureSessionService(test_settings, challenge_svc, buf_store)
    pipeline = LivenessPipelineService(
        settings=test_settings,
        session_service=session_svc,
        quality_service=quality_svc,
        challenge_service=challenge_svc,
        face_detector=detector,
        anti_spoofing=failing_pad,
        photo_storage=photo_storage,
    )

    _, b64 = make_encoded_test_frame(640, 480)
    with next(db_mgr.get_session()) as db:
        repo = SessionRepository(db)
        session = session_svc.create_session(repo)
        ts = 1000
        challenge: ChallengeInstructionSchema | None = session.current_challenge

        while challenge is not None:
            detector.set_face_sequence(_faces_for_challenge(challenge.challenge_type))
            resp = pipeline.submit_challenge_frames(
                repo=repo,
                session_id=session.session_id,
                request=SubmitChallengeFramesRequest(
                    session_nonce=session.session_nonce,
                    challenge_id=challenge.challenge_id,
                    frames=[
                        FrameInputSchema(timestamp_ms=ts + 100, image_base64=b64),
                        FrameInputSchema(timestamp_ms=ts + 200, image_base64=b64),
                    ],
                ),
            )
            ts += 300
            challenge = resp.next_challenge

        final = pipeline.complete_session(
            repo=repo,
            session_id=session.session_id,
            request=CompleteSessionRequest(session_nonce=session.session_nonce),
        )
        assert final.outcome == LivenessOutcome.VALIDATION_UNAVAILABLE
        assert final.validated is False
        assert final.photo_available is False


def test_incomplete_challenges_cannot_be_validated(
    test_settings: Settings,
    photo_storage: LocalPrivatePhotoStorage,
) -> None:
    db_mgr = DatabaseManager(test_settings.database_url)
    db_mgr.init_schema()
    detector = StubFaceDetector(ready=True)
    pad = StubAntiSpoofingProvider(ready=True, real_score=0.95)
    quality_svc = ImageQualityService(test_settings)
    challenge_svc = ActiveChallengeService(test_settings)
    buf_store = TransientSessionBufferStore()
    session_svc = CaptureSessionService(test_settings, challenge_svc, buf_store)
    pipeline = LivenessPipelineService(
        settings=test_settings,
        session_service=session_svc,
        quality_service=quality_svc,
        challenge_service=challenge_svc,
        face_detector=detector,
        anti_spoofing=pad,
        photo_storage=photo_storage,
    )

    with next(db_mgr.get_session()) as db:
        repo = SessionRepository(db)
        session = session_svc.create_session(repo)
        # Immediately call complete_session without completing active challenges
        final = pipeline.complete_session(
            repo=repo,
            session_id=session.session_id,
            request=CompleteSessionRequest(session_nonce=session.session_nonce),
        )
        assert final.outcome == LivenessOutcome.CHALLENGE_FAILED
        assert final.validated is False

        # Subsequent reuse of finalized session is rejected with 409 SESSION_INVALID_STATE
        with pytest.raises(AppError) as exc_info:
            pipeline.complete_session(
                repo=repo,
                session_id=session.session_id,
                request=CompleteSessionRequest(session_nonce=session.session_nonce),
            )
        assert exc_info.value.code == ErrorCode.SESSION_INVALID_STATE


def test_privacy_redacting_log_filter() -> None:
    configure_logging("INFO")
    flt = PrivacyRedactingFilter()
    import logging

    rec = logging.LogRecord(
        name="liveness",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='Payload {"session_nonce": "super-secret-nonce", "image_base64": "data:image/jpeg;base64,AAAA"}',
        args=(),
        exc_info=None,
    )
    flt.filter(rec)
    assert "super-secret-nonce" not in rec.getMessage()
    assert "AAAA" not in rec.getMessage()
    assert "[REDACTED]" in rec.getMessage()
