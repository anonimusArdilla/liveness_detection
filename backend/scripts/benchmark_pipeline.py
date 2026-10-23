"""Benchmark script measuring real CPU latency and memory footprint of the liveness vision pipeline."""

from __future__ import annotations

import json
import statistics
import time
import tracemalloc

import cv2
import numpy as np
from app.core.config import get_settings
from app.services.image_quality import ImageQualityService
from app.vision.face_detector import MediaPipeFaceDetector
from app.vision.face_landmarks import BoundingBox
from app.vision.providers.minifasnet_onnx import MiniFASNetOnnxProvider


def generate_benchmark_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a deterministic textured BGR test frame for CPU latency benchmarking."""
    rng = np.random.default_rng(42)
    base = rng.integers(80, 170, size=(height, width, 3), dtype=np.uint8)
    # Add high-frequency geometric features so Laplacian variance and ResNet/MiniFASNet convolutions do full work
    cv2.circle(base, (width // 2, height // 2), 95, (190, 200, 215), -1)
    cv2.circle(base, (width // 2 - 32, height // 2 - 20), 12, (40, 40, 40), -1)
    cv2.circle(base, (width // 2 + 32, height // 2 - 20), 12, (40, 40, 40), -1)
    cv2.ellipse(base, (width // 2, height // 2 + 30), (28, 12), 0, 0, 180, (60, 70, 140), 3)
    return base


def summarize_latencies(samples_ms: list[float]) -> dict[str, float]:
    sorted_s = sorted(samples_ms)
    p95_idx = min(len(sorted_s) - 1, int(round(0.95 * (len(sorted_s) - 1))))
    return {
        "mean_ms": round(statistics.mean(sorted_s), 2),
        "median_ms": round(statistics.median(sorted_s), 2),
        "p95_ms": round(sorted_s[p95_idx], 2),
        "min_ms": round(sorted_s[0], 2),
        "max_ms": round(sorted_s[-1], 2),
    }


def run_benchmark(iterations: int = 25) -> dict[str, object]:
    settings = get_settings()
    tracemalloc.start()

    t0 = time.perf_counter()
    detector = MediaPipeFaceDetector(
        model_path=settings.face_landmarker_model_path,
        expected_sha256=settings.face_landmarker_sha256,
        verify_checksum=settings.verify_model_checksum,
    )
    pad_single = MiniFASNetOnnxProvider(
        primary_model_path=settings.primary_model_path,
        primary_scale=settings.primary_model_scale,
        primary_sha256=settings.primary_model_sha256,
        secondary_model_path=None,
        verify_checksum=settings.verify_model_checksum,
    )
    pad_ensemble = MiniFASNetOnnxProvider(
        primary_model_path=settings.primary_model_path,
        primary_scale=settings.primary_model_scale,
        primary_sha256=settings.primary_model_sha256,
        secondary_model_path=settings.secondary_model_path,
        secondary_scale=settings.secondary_model_scale,
        secondary_sha256=settings.secondary_model_sha256,
        verify_checksum=settings.verify_model_checksum,
    )
    quality_svc = ImageQualityService(settings)
    init_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    frame = generate_benchmark_frame(640, 480)
    bbox = BoundingBox(x=220, y=140, width=200, height=220)

    quality_times: list[float] = []
    detector_times: list[float] = []
    single_onnx_times: list[float] = []
    ensemble_onnx_times: list[float] = []

    for _ in range(iterations):
        s = time.perf_counter()
        faces = detector.detect_faces(frame)
        detector_times.append((time.perf_counter() - s) * 1000.0)

        s = time.perf_counter()
        quality_svc.evaluate_frame(frame, faces)
        quality_times.append((time.perf_counter() - s) * 1000.0)

        s = time.perf_counter()
        pad_single.predict_frame(frame, bbox)
        single_onnx_times.append((time.perf_counter() - s) * 1000.0)

        s = time.perf_counter()
        pad_ensemble.predict_frame(frame, bbox)
        ensemble_onnx_times.append((time.perf_counter() - s) * 1000.0)

    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    detector.close()

    report = {
        "iterations": iterations,
        "frame_resolution": "640x480",
        "execution_provider": settings.onnx_execution_provider,
        "intra_op_threads": settings.onnx_intra_op_threads,
        "model_initialization_ms": init_time_ms,
        "python_heap_current_mb": round(current_bytes / (1024 * 1024), 2),
        "python_heap_peak_mb": round(peak_bytes / (1024 * 1024), 2),
        "stages": {
            "image_quality_opencv": summarize_latencies(quality_times),
            "mediapipe_face_landmarker": summarize_latencies(detector_times),
            "minifasnet_v2_single_onnx": summarize_latencies(single_onnx_times),
            "minifasnet_v2_plus_v1se_ensemble_onnx": summarize_latencies(ensemble_onnx_times),
        },
    }
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run_benchmark()
