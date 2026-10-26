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
from app.services.challenge import (
    EXPRESSION_UNAVAILABLE_HINT,
    RANDOM_CHALLENGE_POOL,
    TOTAL_CHALLENGE_STEPS,
    ActiveChallengeService,
)
from app.vision.face_landmarks import (
    BoundingBox,
    FaceAnalysis,
    FacialExpression,
    HeadPose,
    expression_from_blendshapes,
)
from pydantic import ValidationError


def _pose_analysis(
    yaw: float,
    pitch: float = 0.0,
    expression: FacialExpression | None = None,
) -> FaceAnalysis:
    return FaceAnalysis(
        bounding_box=BoundingBox(x=200, y=120, width=220, height=220),
        face_area_ratio=0.18,
        center_offset_ratio=0.04,
        head_pose=HeadPose(yaw=yaw, pitch=pitch, roll=0.0),
        landmarks_count=478,
        expression=expression,
    )


def _expr(
    smile: float = 0.05,
    eyes_closed: float = 0.05,
    eyes_open_max: float = 0.08,
    brow_raise: float = 0.1,
) -> FacialExpression:
    return FacialExpression(
        smile=smile,
        eyes_closed=eyes_closed,
        eyes_open_max=eyes_open_max,
        brow_raise=brow_raise,
    )


NEUTRAL = _expr()
EYES_CLOSED = _expr(eyes_closed=0.82, eyes_open_max=0.88)


def test_challenge_sequence_randomization_and_structure(test_settings: Settings) -> None:
    svc = ActiveChallengeService(test_settings)
    sequences = [tuple(ctype for ctype, _ in svc.generate_challenge_sequence()) for _ in range(400)]

    for seq in sequences:
        # Mandatory CENTER_FACE followed by exactly one randomly drawn challenge
        assert len(seq) == TOTAL_CHALLENGE_STEPS == 2
        assert seq[0] == ChallengeType.CENTER_FACE
        assert seq[1] in RANDOM_CHALLENGE_POOL
        assert seq[1] != ChallengeType.CENTER_FACE

    # Every challenge in the pool (turns + expressions) is reachable
    assert {seq[1] for seq in sequences} == set(RANDOM_CHALLENGE_POOL)
    assert set(RANDOM_CHALLENGE_POOL) == {
        ChallengeType.TURN_LEFT,
        ChallengeType.TURN_RIGHT,
        ChallengeType.SMILE,
        ChallengeType.BLINK,
        ChallengeType.RAISE_EYEBROWS,
    }


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


def test_center_face_requires_neutral_expression_when_available(test_settings: Settings) -> None:
    svc = ActiveChallengeService(test_settings)

    res_neutral = svc.evaluate_challenge_batch(
        ChallengeType.CENTER_FACE,
        [(0, _pose_analysis(yaw=1.0, expression=NEUTRAL)), (1, _pose_analysis(yaw=-2.0, expression=NEUTRAL))],
    )
    assert res_neutral.passed is True

    # A smiling (or brow-raised) frontal face must not satisfy CENTER_FACE, so a single
    # static photo cannot pass both CENTER_FACE and SMILE / RAISE_EYEBROWS.
    smiling = _expr(smile=0.8)
    res_smiling = svc.evaluate_challenge_batch(
        ChallengeType.CENTER_FACE,
        [(0, _pose_analysis(yaw=1.0, expression=smiling)), (1, _pose_analysis(yaw=0.0, expression=smiling))],
    )
    assert res_smiling.passed is False
    assert "neutral" in res_smiling.feedback_message.lower()

    brows = _expr(brow_raise=0.75)
    res_brows = svc.evaluate_challenge_batch(
        ChallengeType.CENTER_FACE,
        [(0, _pose_analysis(yaw=1.0, expression=brows)), (1, _pose_analysis(yaw=0.0, expression=brows))],
    )
    assert res_brows.passed is False


def test_smile_challenge_evaluation(test_settings: Settings) -> None:
    svc = ActiveChallengeService(test_settings)

    res_fail = svc.evaluate_challenge_batch(
        ChallengeType.SMILE,
        [(0, _pose_analysis(yaw=0.0, expression=NEUTRAL)), (1, _pose_analysis(yaw=0.0, expression=_expr(smile=0.3)))],
    )
    assert res_fail.passed is False

    res_pass = svc.evaluate_challenge_batch(
        ChallengeType.SMILE,
        [
            (0, _pose_analysis(yaw=0.0, expression=_expr(smile=0.7))),
            (1, _pose_analysis(yaw=2.0, expression=_expr(smile=0.9))),
        ],
    )
    assert res_pass.passed is True
    assert res_pass.best_frame_index == 1  # strongest smile


def test_raise_eyebrows_challenge_evaluation(test_settings: Settings) -> None:
    svc = ActiveChallengeService(test_settings)

    res_fail = svc.evaluate_challenge_batch(
        ChallengeType.RAISE_EYEBROWS,
        [(0, _pose_analysis(yaw=0.0, expression=NEUTRAL)), (1, _pose_analysis(yaw=0.0, expression=NEUTRAL))],
    )
    assert res_fail.passed is False

    res_pass = svc.evaluate_challenge_batch(
        ChallengeType.RAISE_EYEBROWS,
        [
            (0, _pose_analysis(yaw=0.0, expression=_expr(brow_raise=0.65))),
            (1, _pose_analysis(yaw=0.0, expression=_expr(brow_raise=0.7))),
        ],
    )
    assert res_pass.passed is True


def test_blink_requires_open_closed_transition_within_batch(test_settings: Settings) -> None:
    svc = ActiveChallengeService(test_settings)

    # Static open eyes (e.g. printed photo) -> fail
    res_open = svc.evaluate_challenge_batch(
        ChallengeType.BLINK,
        [(i, _pose_analysis(yaw=0.0, expression=NEUTRAL)) for i in range(4)],
    )
    assert res_open.passed is False

    # Static closed eyes (e.g. photo with eyes shut) -> fail
    res_closed = svc.evaluate_challenge_batch(
        ChallengeType.BLINK,
        [(i, _pose_analysis(yaw=0.0, expression=EYES_CLOSED)) for i in range(4)],
    )
    assert res_closed.passed is False

    # Wink (only one eye closed) does not count as closed
    wink = _expr(eyes_closed=0.1, eyes_open_max=0.9)
    res_wink = svc.evaluate_challenge_batch(
        ChallengeType.BLINK,
        [(0, _pose_analysis(yaw=0.0, expression=NEUTRAL)), (1, _pose_analysis(yaw=0.0, expression=wink))],
    )
    assert res_wink.passed is False

    # Closed-eye frame while head is turned is ignored (must stay roughly frontal)
    res_turned = svc.evaluate_challenge_batch(
        ChallengeType.BLINK,
        [(0, _pose_analysis(yaw=0.0, expression=NEUTRAL)), (1, _pose_analysis(yaw=30.0, expression=EYES_CLOSED))],
    )
    assert res_turned.passed is False

    # Genuine blink: open -> closed -> open
    res_blink = svc.evaluate_challenge_batch(
        ChallengeType.BLINK,
        [
            (0, _pose_analysis(yaw=0.0, expression=NEUTRAL)),
            (1, _pose_analysis(yaw=0.0, expression=EYES_CLOSED)),
            (2, _pose_analysis(yaw=0.0, expression=_expr(eyes_open_max=0.04))),
        ],
    )
    assert res_blink.passed is True
    assert res_blink.best_frame_index == 2  # most clearly open-eyed frame


@pytest.mark.parametrize(
    "ctype",
    [ChallengeType.SMILE, ChallengeType.BLINK, ChallengeType.RAISE_EYEBROWS],
)
def test_expression_challenges_fail_closed_without_blendshapes(
    test_settings: Settings, ctype: ChallengeType
) -> None:
    svc = ActiveChallengeService(test_settings)
    res = svc.evaluate_challenge_batch(
        ctype,
        [(i, _pose_analysis(yaw=0.0, expression=None)) for i in range(4)],
    )
    assert res.passed is False
    assert res.feedback_message == EXPRESSION_UNAVAILABLE_HINT


def test_expression_from_blendshapes_mapping_and_fail_closed() -> None:
    scores = {
        "mouthSmileLeft": 0.8,
        "mouthSmileRight": 0.6,
        "eyeBlinkLeft": 0.9,
        "eyeBlinkRight": 0.2,
        "browInnerUp": 0.6,
        "browOuterUpLeft": 0.4,
        "browOuterUpRight": float("nan"),
    }
    expr = expression_from_blendshapes(scores)
    assert expr is not None
    assert expr.smile == pytest.approx(0.7)
    assert expr.eyes_closed == pytest.approx(0.2)
    assert expr.eyes_open_max == pytest.approx(0.9)
    # NaN is clamped to 0.0 -> (0.6 + (0.4 + 0.0) / 2) / 2
    assert expr.brow_raise == pytest.approx(0.4)

    del scores["browInnerUp"]
    assert expression_from_blendshapes(scores) is None


def test_expression_threshold_settings_validation(test_settings: Settings) -> None:
    with pytest.raises(ValidationError):
        Settings(blink_open_max_score=0.5, blink_closed_min_score=0.5)
    with pytest.raises(ValidationError):
        Settings(expression_neutral_max_score=0.6, smile_min_score=0.55)


def test_created_session_has_exactly_two_steps(test_settings: Settings) -> None:
    db_mgr = DatabaseManager(test_settings.database_url)
    db_mgr.init_schema()
    session_svc = CaptureSessionService(
        test_settings, ActiveChallengeService(test_settings), TransientSessionBufferStore()
    )
    with next(db_mgr.get_session()) as db:
        repo = SessionRepository(db)
        created = session_svc.create_session(repo)
        assert created.total_challenges == 2
        assert created.current_challenge.challenge_type == ChallengeType.CENTER_FACE
        record = repo.get_session(created.session_id)
        assert record is not None
        assert len(record.challenges) == 2
        assert ChallengeType(record.challenges[1].challenge_type) in RANDOM_CHALLENGE_POOL


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
