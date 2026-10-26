"""Integration tests for the /api/v1 REST endpoints, photo persistence, and security controls."""

from __future__ import annotations

import pytest
from app.api.dependencies import build_service_container, set_container
from app.core.config import Settings
from app.main import create_app
from app.schemas.liveness import ChallengeType, LivenessOutcome
from app.services.challenge import RANDOM_CHALLENGE_POOL, TOTAL_CHALLENGE_STEPS
from app.vision.face_landmarks import BoundingBox, FaceAnalysis, FacialExpression, HeadPose
from fastapi.testclient import TestClient

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
        center_offset_ratio=0.02,
        head_pose=HeadPose(yaw=yaw, pitch=0.0, roll=0.0),
        landmarks_count=478,
        expression=expression,
    )


def _face_sequence_for_type(ctype: str) -> list[list[FaceAnalysis]]:
    """Per-frame detector output that satisfies the given challenge type."""
    if ctype == ChallengeType.TURN_LEFT.value:
        return [[_face(yaw=-21.0)]]
    if ctype == ChallengeType.TURN_RIGHT.value:
        return [[_face(yaw=21.0)]]
    if ctype == ChallengeType.SMILE.value:
        return [[_face(expression=FacialExpression(0.82, 0.05, 0.08, 0.1))]]
    if ctype == ChallengeType.RAISE_EYEBROWS.value:
        return [[_face(expression=FacialExpression(0.05, 0.05, 0.08, 0.72))]]
    if ctype == ChallengeType.BLINK.value:
        # open -> closed transition across consecutive frames
        return [[_face()], [_face(expression=FacialExpression(0.05, 0.85, 0.9, 0.1))]]
    return [[_face()]]


@pytest.mark.parametrize("forced_follow_up", RANDOM_CHALLENGE_POOL)
def test_full_api_capture_workflow_persists_photo_and_zero_raw_video(
    test_settings: Settings,
    forced_follow_up: ChallengeType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = StubFaceDetector(ready=True)
    pad = StubAntiSpoofingProvider(ready=True, real_score=0.94)
    container = build_service_container(
        settings=test_settings,
        face_detector=detector,
        anti_spoofing=pad,
    )
    # Force the CSPRNG sample so forced_follow_up is always included in the 3 random follow-ups
    other_pool = [c for c in RANDOM_CHALLENGE_POOL if c != forced_follow_up]
    monkeypatch.setattr(
        container.challenge_service._rng,
        "sample",
        lambda pool, k: [forced_follow_up, other_pool[0], other_pool[1]],
    )
    set_container(container)
    app = create_app(test_settings)

    try:
        with TestClient(app) as client:
            # 1. Health & readiness checks
            health_resp = client.get("/api/v1/health/ready")
            assert health_resp.status_code == 200
            assert health_resp.json()["ready"] is True

            # 2. Create session
            create_resp = client.post(
                "/api/v1/sessions",
                json={"registration_ref": "reg-user-99"},
            )
            assert create_resp.status_code == 201
            session_data = create_resp.json()
            session_id = session_data["session_id"]
            nonce = session_data["session_nonce"]
            current_challenge = session_data["current_challenge"]
            assert session_data["outcome"] == LivenessOutcome.PENDING.value
            assert session_data["total_challenges"] == TOTAL_CHALLENGE_STEPS == 4
            assert current_challenge["challenge_type"] == ChallengeType.CENTER_FACE.value
            seen_types: list[str] = []

            # 3. Progressive challenge execution
            _, b64 = make_encoded_test_frame(640, 480)
            ts = 1000
            steps_completed = 0

            while current_challenge is not None:
                # Verify GET /challenge matches progressive disclosure
                get_chal = client.get(f"/api/v1/sessions/{session_id}/challenge")
                assert get_chal.status_code == 200
                assert (
                    get_chal.json()["current_challenge"]["challenge_id"] == current_challenge["challenge_id"]
                )

                seen_types.append(current_challenge["challenge_type"])
                detector.set_face_sequence(_face_sequence_for_type(current_challenge["challenge_type"]))
                submit_resp = client.post(
                    f"/api/v1/sessions/{session_id}/challenge/frames",
                    json={
                        "session_nonce": nonce,
                        "challenge_id": current_challenge["challenge_id"],
                        "frames": [
                            {"timestamp_ms": ts + 100, "image_base64": b64},
                            {"timestamp_ms": ts + 200, "image_base64": b64},
                        ],
                    },
                )
                ts += 300
                assert submit_resp.status_code == 200
                sub_json = submit_resp.json()
                assert sub_json["challenge_passed"] is True
                steps_completed += 1
                current_challenge = sub_json["next_challenge"]

            assert steps_completed == TOTAL_CHALLENGE_STEPS == 4
            assert seen_types[0] == ChallengeType.CENTER_FACE.value
            assert forced_follow_up.value in seen_types[1:]
            # Verify transient memory buffers still exist prior to completion, and no disk photo yet
            assert list(test_settings.photo_storage_dir.glob("*")) == []

            # 4. Complete session
            complete_resp = client.post(
                f"/api/v1/sessions/{session_id}/complete",
                json={"session_nonce": nonce},
            )
            assert complete_resp.status_code == 200
            result = complete_resp.json()
            assert result["outcome"] == LivenessOutcome.VALIDATED.value
            assert result["validated"] is True
            assert result["registration_ref"] == "reg-user-99"
            assert result["photo_available"] is True
            assert result["photo_access_token"] is not None
            assert result["verification_signature"] is not None

            # 5. Confirm transient in-memory frame buffers were purged and zero raw video exists
            assert container.buffer_store.pop_buffers(session_id) is None
            stored_files = list(test_settings.photo_storage_dir.glob("*"))
            assert len(stored_files) == 1
            assert stored_files[0].suffix == ".jpg"
            assert list(test_settings.photo_storage_dir.glob("*.mp4")) == []
            assert list(test_settings.photo_storage_dir.glob("*.webm")) == []

            # 6. Protected photo retrieval
            unauth_photo = client.get(f"/api/v1/sessions/{session_id}/photo")
            assert unauth_photo.status_code == 403

            wrong_token_photo = client.get(
                f"/api/v1/sessions/{session_id}/photo",
                headers={"X-Photo-Token": "wrong-token"},
            )
            assert wrong_token_photo.status_code == 403

            valid_photo = client.get(
                f"/api/v1/sessions/{session_id}/photo",
                headers={"X-Photo-Token": result["photo_access_token"]},
            )
            assert valid_photo.status_code == 200
            assert valid_photo.headers["content-type"] == "image/jpeg"
            assert valid_photo.content.startswith(b"\xff\xd8")

            # 7. Replay protection: attempting to complete or submit frames again returns 409
            replay_resp = client.post(
                f"/api/v1/sessions/{session_id}/complete",
                json={"session_nonce": nonce},
            )
            assert replay_resp.status_code == 409
            assert replay_resp.json()["error"]["code"] == "SESSION_INVALID_STATE"
    finally:
        set_container(None)


def test_readiness_endpoint_and_session_creation_fail_closed_when_model_unready(
    test_settings: Settings,
) -> None:
    detector = StubFaceDetector(ready=True)
    unready_pad = StubAntiSpoofingProvider(ready=False)
    container = build_service_container(
        settings=test_settings,
        face_detector=detector,
        anti_spoofing=unready_pad,
    )
    set_container(container)
    app = create_app(test_settings)

    try:
        with TestClient(app) as client:
            ready_resp = client.get("/api/v1/health/ready")
            assert ready_resp.status_code == 503
            assert ready_resp.json()["ready"] is False

            create_resp = client.post("/api/v1/sessions", json={})
            assert create_resp.status_code == 503
            assert create_resp.json()["error"]["code"] == "SERVICE_UNAVAILABLE"
    finally:
        set_container(None)


def test_payload_size_limit_and_rate_limiting(test_settings: Settings) -> None:
    strict_settings = test_settings.model_copy(
        update={
            "max_request_bytes": 1024,
            "rate_limit_requests_per_minute": 5,
        }
    )
    container = build_service_container(
        settings=strict_settings,
        face_detector=StubFaceDetector(ready=True),
        anti_spoofing=StubAntiSpoofingProvider(ready=True),
    )
    set_container(container)
    app = create_app(strict_settings)

    try:
        with TestClient(app) as client:
            # Oversized payload (> 1024 bytes) rejected with 413
            oversized = client.post(
                "/api/v1/sessions",
                content="x" * 2048,
                headers={"Content-Type": "application/json"},
            )
            assert oversized.status_code == 413
            assert oversized.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"

            # Exceed rate limit of 5 requests/min
            status_codes = [client.post("/api/v1/sessions", json={}).status_code for _ in range(7)]
            assert 429 in status_codes
    finally:
        set_container(None)
