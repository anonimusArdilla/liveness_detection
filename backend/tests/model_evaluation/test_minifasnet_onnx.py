"""Integration tests exercising actual ONNX Runtime MiniFASNet and MediaPipe models."""

from __future__ import annotations

import numpy as np
import pytest
from app.core.config import get_settings
from app.vision.face_detector import MediaPipeFaceDetector
from app.vision.face_landmarks import BoundingBox
from app.vision.providers.minifasnet_onnx import (
    MiniFASNetOnnxProvider,
    crop_face_minivision,
    preprocess_minifasnet_patch,
)

settings = get_settings()
MODELS_AVAILABLE = (
    settings.primary_model_path.is_file()
    and (settings.secondary_model_path is None or settings.secondary_model_path.is_file())
    and settings.face_landmarker_model_path.is_file()
)


@pytest.mark.requires_models
@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Real ONNX and MediaPipe model files not installed")
def test_real_minifasnet_onnx_loading_checksum_and_inference() -> None:
    provider = MiniFASNetOnnxProvider(
        primary_model_path=settings.primary_model_path,
        primary_scale=settings.primary_model_scale,
        primary_sha256=settings.primary_model_sha256,
        secondary_model_path=settings.secondary_model_path,
        secondary_scale=settings.secondary_model_scale,
        secondary_sha256=settings.secondary_model_sha256,
        verify_checksum=True,
    )
    assert provider.is_ready() is True
    for st in provider.model_statuses():
        assert st.ready is True
        assert st.checksum_verified is True

    # Test single-frame and multi-frame inference on synthetic BGR input
    rng = np.random.default_rng(7)
    synthetic_bgr = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
    bbox = BoundingBox(x=180, y=100, width=220, height=260)

    score = provider.predict_frame(synthetic_bgr, bbox)
    assert 0.0 <= score.real_probability <= 1.0
    assert 0.0 <= score.spoof_probability <= 1.0
    assert abs((score.real_probability + score.spoof_probability) - 1.0) < 1e-2

    batch_eval = provider.evaluate_frames(
        frames_with_boxes=[(synthetic_bgr, bbox), (synthetic_bgr, bbox)],
        real_score_threshold=0.65,
        min_frame_score_threshold=0.45,
    )
    assert batch_eval.frames_evaluated == 2
    assert batch_eval.model_name == "MiniFASNetV2+MiniFASNetV1SE"


@pytest.mark.requires_models
@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Real ONNX and MediaPipe model files not installed")
def test_real_mediapipe_face_landmarker_loading_and_execution() -> None:
    detector = MediaPipeFaceDetector(
        model_path=settings.face_landmarker_model_path,
        expected_sha256=settings.face_landmarker_sha256,
        verify_checksum=True,
    )
    try:
        st = detector.status()
        assert st.ready is True
        assert st.checksum_verified is True

        blank_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = detector.detect_faces(blank_bgr)
        assert faces == []
    finally:
        detector.close()


def test_minivision_crop_and_preprocess_tensor_shape() -> None:
    bgr = np.full((480, 640, 3), 120, dtype=np.uint8)
    bbox = BoundingBox(x=10, y=10, width=150, height=180)
    patch = crop_face_minivision(bgr, bbox, scale=2.7, out_w=80, out_h=80)
    assert patch.shape == (80, 80, 3)

    tensor = preprocess_minifasnet_patch(patch)
    assert tensor.shape == (1, 3, 80, 80)
    assert tensor.dtype == np.float32
    assert float(tensor.max()) == 120.0
