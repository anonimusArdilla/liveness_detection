"""Shared pytest fixtures and deterministic synthetic helpers for backend tests."""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
import pytest
from app.core.config import Settings
from app.schemas.liveness import ModelStatusSchema
from app.storage.local_storage import LocalPrivatePhotoStorage
from app.vision.anti_spoofing import (
    AntiSpoofingProvider,
    BatchSpoofEvaluation,
    FrameSpoofScore,
)
from app.vision.face_detector import FaceDetectorProvider
from app.vision.face_landmarks import BoundingBox, FaceAnalysis, HeadPose


def make_encoded_test_frame(
    width: int = 640,
    height: int = 480,
    brightness: int = 128,
    sharp: bool = True,
    marker_value: int = 0,
) -> tuple[np.ndarray, str]:
    """Create an in-memory BGR image and its base64 JPEG encoding.

    Embeds `marker_value` in pixel (0, 0, 0) before encoding or allows tests to control
    sharpness (high-contrast checkerboard vs uniform flat image) and brightness.
    """
    img = np.full((height, width, 3), brightness, dtype=np.uint8)
    if sharp:
        # Draw high-frequency grid lines so Laplacian variance exceeds threshold
        for y in range(20, height - 20, 24):
            cv2.line(
                img,
                (20, y),
                (width - 20, y),
                (min(255, brightness + 55), max(0, brightness - 55), brightness),
                2,
            )
        for x in range(20, width - 20, 24):
            cv2.line(
                img,
                (x, 20),
                (x, height - 20),
                (max(0, brightness - 55), min(255, brightness + 55), brightness),
                2,
            )
    img[0, 0, 0] = marker_value % 256
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    assert ok
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return img, b64


class StubFaceDetector(FaceDetectorProvider):
    """Configurable test double for FaceDetectorProvider (used only in isolated unit/API tests)."""

    def __init__(
        self,
        ready: bool = True,
        faces_to_return: list[FaceAnalysis] | None = None,
    ) -> None:
        self.ready = ready
        self.faces_to_return = (
            faces_to_return
            if faces_to_return is not None
            else [
                FaceAnalysis(
                    bounding_box=BoundingBox(x=200, y=120, width=240, height=240),
                    face_area_ratio=0.1875,
                    center_offset_ratio=0.0,
                    head_pose=HeadPose(yaw=0.0, pitch=0.0, roll=0.0),
                    landmarks_count=478,
                )
            ]
        )

    def status(self) -> ModelStatusSchema:
        return ModelStatusSchema(
            name="StubFaceDetector",
            ready=self.ready,
            provider="TestExecutionProvider",
            path="stub_face_landmarker.task",
            checksum_verified=self.ready,
            reason=None if self.ready else "Stub detector disabled",
        )

    def detect_faces(self, bgr_image: np.ndarray) -> list[FaceAnalysis]:
        if not self.ready:
            raise RuntimeError("Stub detector is not ready")
        return list(self.faces_to_return)


class StubAntiSpoofingProvider(AntiSpoofingProvider):
    """Configurable test double for AntiSpoofingProvider (isolated to test suites)."""

    def __init__(
        self,
        ready: bool = True,
        real_score: float = 0.92,
        raise_on_infer: bool = False,
    ) -> None:
        self._ready = ready
        self.real_score = real_score
        self.raise_on_infer = raise_on_infer

    def is_ready(self) -> bool:
        return self._ready

    def model_statuses(self) -> list[ModelStatusSchema]:
        return [
            ModelStatusSchema(
                name="StubMiniFASNet",
                ready=self._ready,
                provider="TestExecutionProvider",
                path="stub_minifasnet.onnx",
                checksum_verified=self._ready,
                reason=None if self._ready else "Stub anti-spoofing model not loaded",
            )
        ]

    def predict_frame(self, bgr_image: np.ndarray, face_bbox: BoundingBox) -> FrameSpoofScore:
        if not self._ready or self.raise_on_infer:
            raise RuntimeError("Simulated ONNX inference failure")
        spoof = round(1.0 - self.real_score, 4)
        return FrameSpoofScore(
            real_probability=self.real_score,
            spoof_probability=spoof,
            class_probabilities=(spoof / 2.0, self.real_score, spoof / 2.0),
        )

    def evaluate_frames(
        self,
        frames_with_boxes: list[tuple[np.ndarray, BoundingBox]],
        real_score_threshold: float,
        min_frame_score_threshold: float,
    ) -> BatchSpoofEvaluation:
        if not self._ready or self.raise_on_infer:
            raise RuntimeError("Simulated ONNX batch inference failure")
        passed = self.real_score >= real_score_threshold and self.real_score >= min_frame_score_threshold
        return BatchSpoofEvaluation(
            passed=passed,
            aggregate_real_score=self.real_score,
            min_frame_real_score=self.real_score,
            frames_evaluated=len(frames_with_boxes),
            threshold_applied=real_score_threshold,
            model_name="StubMiniFASNet",
        )


@pytest.fixture()
def test_settings(tmp_path: Path) -> Settings:
    """Provide isolated test settings backed by a temporary SQLite DB and storage directory."""
    db_file = tmp_path / "test_liveness.db"
    photos_dir = tmp_path / "accepted_photos"
    return Settings(
        database_url=f"sqlite:///{db_file.as_posix()}",
        photo_storage_dir=photos_dir,
        session_ttl_seconds=120,
        challenge_count=3,
        min_frames_per_batch=2,
        max_frames_per_batch=6,
        min_challenge_passing_frames=2,
        rate_limit_requests_per_minute=200,
    )


@pytest.fixture()
def photo_storage(test_settings: Settings) -> LocalPrivatePhotoStorage:
    return LocalPrivatePhotoStorage(base_dir=test_settings.photo_storage_dir)
