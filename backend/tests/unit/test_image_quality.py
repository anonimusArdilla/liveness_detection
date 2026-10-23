"""Unit tests for image-quality checks and base64 frame decoding."""

from __future__ import annotations

import pytest
from app.core.config import Settings
from app.core.errors import AppError, ErrorCode
from app.schemas.liveness import QualityIssueCode
from app.services.image_quality import ImageQualityService, decode_base64_image
from app.vision.face_landmarks import BoundingBox, FaceAnalysis, HeadPose

from tests.conftest import make_encoded_test_frame


def _make_face(
    area_ratio: float = 0.18,
    center_offset: float = 0.05,
    yaw: float = 0.0,
    pitch: float = 0.0,
) -> FaceAnalysis:
    return FaceAnalysis(
        bounding_box=BoundingBox(x=200, y=120, width=220, height=220),
        face_area_ratio=area_ratio,
        center_offset_ratio=center_offset,
        head_pose=HeadPose(yaw=yaw, pitch=pitch, roll=0.0),
        landmarks_count=478,
    )


def test_decode_base64_image_valid_and_data_url() -> None:
    _, b64 = make_encoded_test_frame(640, 480)
    bgr1 = decode_base64_image(b64)
    assert bgr1.shape == (480, 640, 3)

    data_url = f"data:image/jpeg;base64,{b64}"
    bgr2 = decode_base64_image(data_url)
    assert bgr2.shape == (480, 640, 3)


@pytest.mark.parametrize(
    "bad_payload",
    [
        "",
        "   ",
        "not-valid-base64!!!",
        "ZHVtbXk=",  # valid base64 ("dummy"), but < 64 bytes and not an image
        "data:image/jpeg;base64," + ("A" * 200),  # valid base64 length, invalid image bytes
    ],
)
def test_decode_base64_image_rejects_malformed_inputs(bad_payload: str) -> None:
    with pytest.raises(AppError) as exc_info:
        decode_base64_image(bad_payload)
    assert exc_info.value.code == ErrorCode.IMAGE_DECODE_FAILED


def test_quality_service_passes_clean_centered_frame(test_settings: Settings) -> None:
    svc = ImageQualityService(test_settings)
    frame, _ = make_encoded_test_frame(640, 480, brightness=130, sharp=True)
    report = svc.evaluate_frame(frame, [_make_face()])
    assert report.passed is True
    assert report.issues == []


def test_quality_service_detects_no_face_and_multiple_faces(test_settings: Settings) -> None:
    svc = ImageQualityService(test_settings)
    frame, _ = make_encoded_test_frame(640, 480, brightness=130, sharp=True)

    no_face_report = svc.evaluate_frame(frame, [])
    assert no_face_report.passed is False
    assert [i.code for i in no_face_report.issues] == [QualityIssueCode.NO_FACE_DETECTED]

    multi_face_report = svc.evaluate_frame(frame, [_make_face(), _make_face()])
    assert multi_face_report.passed is False
    assert [i.code for i in multi_face_report.issues] == [QualityIssueCode.MULTIPLE_FACES_DETECTED]


def test_quality_service_detects_resolution_lighting_and_blur(test_settings: Settings) -> None:
    svc = ImageQualityService(test_settings)

    # Low resolution + too dark + blurry (flat image)
    frame_dark_blur, _ = make_encoded_test_frame(200, 150, brightness=15, sharp=False)
    report = svc.evaluate_frame(frame_dark_blur, [_make_face()])
    assert report.passed is False
    codes = {i.code for i in report.issues}
    assert QualityIssueCode.RESOLUTION_TOO_LOW in codes
    assert QualityIssueCode.LIGHTING_TOO_DARK in codes
    assert QualityIssueCode.IMAGE_TOO_BLURRY in codes

    # Too bright
    frame_bright, _ = make_encoded_test_frame(640, 480, brightness=245, sharp=True)
    report_bright = svc.evaluate_frame(frame_bright, [_make_face()])
    assert QualityIssueCode.LIGHTING_TOO_BRIGHT in {i.code for i in report_bright.issues}


def test_quality_service_detects_face_size_and_centering(test_settings: Settings) -> None:
    svc = ImageQualityService(test_settings)
    frame, _ = make_encoded_test_frame(640, 480, brightness=130, sharp=True)

    small_offcenter = _make_face(area_ratio=0.015, center_offset=0.42)
    report = svc.evaluate_frame(frame, [small_offcenter], require_centered=True)
    assert report.passed is False
    codes = {i.code for i in report.issues}
    assert QualityIssueCode.FACE_TOO_SMALL in codes
    assert QualityIssueCode.FACE_NOT_CENTERED in codes

    close_face = _make_face(area_ratio=0.88, center_offset=0.05)
    report_close = svc.evaluate_frame(frame, [close_face], require_centered=True)
    assert QualityIssueCode.FACE_TOO_CLOSE in {i.code for i in report_close.issues}
