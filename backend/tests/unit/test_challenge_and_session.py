"""Unit tests for randomized active challenges, timestamp ordering, and session expiration."""

from __future__ import annotations

from datetime import timedelta

import pytest
from app.core.config import Settings
from app.core.errors import AppError, ErrorCode
from app.models.session import utc_now
from app.repositories.database import DatabaseManager
from app.repositories.session_repository import SessionRepository
from app.schemas.liveness import ChallengeType, LivenessOutcome
from app.services.capture_session import CaptureSessionService, TransientSessionBufferStore
from app.services.challenge import ActiveChallengeService
from app.vision.face_landmarks import BoundingBox, FaceAnalysis, HeadPose


def _pose_analysis(yaw: float, pitch: float = 0.0) -> FaceAnalysis:
    return FaceAnalysis(
        bounding_box=BoundingBox(x=200, y=120, width=220, height=220),
        face_area_ratio=0.18,
        center_offset_ratio=0.04,
        head_pose=HeadPose(yaw=yaw, pitch=pitch, roll=0.0),
        landmarks_count=478,
    )


def test_challenge_sequence_randomization_and_structure(test_settings: Settings) -> None:
    svc = ActiveChallengeService(test_settings)
    sequences = [tuple(ctype for ctype, _ in svc.generate_challenge_sequence()) for _ in range(30)]

    for seq in sequences:
        assert len(seq) == 3
        assert seq[0] == ChallengeType.CENTER_FACE
        assert set(seq[1:]) == {ChallengeType.TURN_LEFT, ChallengeType.TURN_RIGHT}

    # Verify both random permutations (LEFT->RIGHT and RIGHT->LEFT) occur
    assert len(set(sequences)) == 2


def test_frame_timestamp_monotonic_validation(test_settings: Settings) -> None:
    svc = ActiveChallengeService(test_settings)
    last_ts = svc.validate_frame_timestamps([1000, 1150, 1300], last_session_timestamp_ms=900)
    assert last_ts == 1300

    # Reject non-increasing or replayed timestamps
    with pytest.raises(AppError) as exc_info:
        svc.validate_frame_timestamps([1300, 1400], last_session_timestamp_ms=1300)
    assert exc_info.value.code == ErrorCode.FRAME_ORDER_INVALID

    with pytest.raises(AppError) as exc_info:
        svc.validate_frame_timestamps([1500, 1450], last_session_timestamp_ms=1300)
    assert exc_info.value.code == ErrorCode.FRAME_ORDER_INVALID


def test_challenge_pose_evaluation(test_settings: Settings) -> None:
    svc = ActiveChallengeService(test_settings)

    # CENTER_FACE passes when yaw & pitch are near 0
    res_center = svc.evaluate_challenge_batch(
        ChallengeType.CENTER_FACE,
        [(0, _pose_analysis(yaw=2.0)), (1, _pose_analysis(yaw=-3.0))],
    )
    assert res_center.passed is True

    # TURN_LEFT requires negative yaw <= -13 deg
    res_left_fail = svc.evaluate_challenge_batch(
        ChallengeType.TURN_LEFT,
        [(0, _pose_analysis(yaw=-5.0)), (1, _pose_analysis(yaw=18.0))],
    )
    assert res_left_fail.passed is False

    res_left_pass = svc.evaluate_challenge_batch(
        ChallengeType.TURN_LEFT,
        [(0, _pose_analysis(yaw=-18.0)), (1, _pose_analysis(yaw=-22.0))],
    )
    assert res_left_pass.passed is True

    # TURN_RIGHT requires positive yaw >= +13 deg
    res_right_pass = svc.evaluate_challenge_batch(
        ChallengeType.TURN_RIGHT,
        [(0, _pose_analysis(yaw=17.5)), (1, _pose_analysis(yaw=21.0))],
    )
    assert res_right_pass.passed is True


def test_session_expiration_and_nonce_verification(test_settings: Settings) -> None:
    db_mgr = DatabaseManager(test_settings.database_url)
    db_mgr.init_schema()
    challenge_svc = ActiveChallengeService(test_settings)
    buf_store = TransientSessionBufferStore()
    session_svc = CaptureSessionService(test_settings, challenge_svc, buf_store)

    with next(db_mgr.get_session()) as db:
        repo = SessionRepository(db)
        created = session_svc.create_session(repo, registration_ref="user-123")
        assert created.outcome == LivenessOutcome.PENDING

        # Wrong nonce is rejected with 403 SESSION_NONCE_MISMATCH
        with pytest.raises(AppError) as exc_info:
            session_svc.get_active_session_or_raise(
                repo, created.session_id, session_nonce="wrong-nonce-value-123456"
            )
        assert exc_info.value.code == ErrorCode.SESSION_NONCE_MISMATCH

        # Correct nonce succeeds
        record = session_svc.get_active_session_or_raise(
            repo, created.session_id, session_nonce=created.session_nonce
        )
        assert record.session_id == created.session_id

        # Force expiration and verify 410 SESSION_EXPIRED
        record.expires_at = utc_now() - timedelta(seconds=5)
        repo.save(record)

        with pytest.raises(AppError) as exp_exc:
            session_svc.get_active_session_or_raise(
                repo, created.session_id, session_nonce=created.session_nonce
            )
        assert exp_exc.value.code == ErrorCode.SESSION_EXPIRED
