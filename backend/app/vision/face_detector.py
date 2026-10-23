"""Face detection and landmark extraction service backed by MediaPipe FaceLandmarker."""

from __future__ import annotations

import hashlib
import threading
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from app.core.logging import get_logger
from app.schemas.liveness import ModelStatusSchema
from app.vision.face_landmarks import FaceAnalysis, Point3D, analyze_face_landmarks

logger = get_logger(__name__)


class FaceDetectorProvider(ABC):
    """Interface for detecting faces and extracting 3D facial landmarks from BGR images."""

    @abstractmethod
    def status(self) -> ModelStatusSchema:
        """Report model readiness and integrity status."""

    @abstractmethod
    def detect_faces(self, bgr_image: np.ndarray) -> list[FaceAnalysis]:
        """Detect all faces in a BGR frame and return their geometric/landmark analyses."""


class MediaPipeFaceDetector(FaceDetectorProvider):
    """MediaPipe Tasks FaceLandmarker implementation with SHA-256 checksum verification."""

    def __init__(
        self,
        model_path: Path,
        expected_sha256: str | None = None,
        verify_checksum: bool = True,
        max_faces: int = 3,
        min_confidence: float = 0.5,
    ) -> None:
        self.model_path = model_path
        self.expected_sha256 = expected_sha256
        self.verify_checksum = verify_checksum
        self.max_faces = max_faces
        self.min_confidence = min_confidence

        self._lock = threading.Lock()
        self._landmarker: mp_vision.FaceLandmarker | None = None
        self._ready = False
        self._checksum_verified = False
        self._failure_reason: str | None = None

        self._initialize()

    def _initialize(self) -> None:
        if not self.model_path.is_file():
            self._ready = False
            self._failure_reason = f"Model file not found at {self.model_path.name}"
            logger.warning("MediaPipe FaceLandmarker unavailable: %s", self._failure_reason)
            return

        try:
            model_bytes = self.model_path.read_bytes()
            if self.verify_checksum and self.expected_sha256:
                digest = hashlib.sha256(model_bytes).hexdigest()
                if digest.lower() != self.expected_sha256.lower():
                    self._ready = False
                    self._failure_reason = "SHA-256 checksum mismatch for face_landmarker.task"
                    logger.error(
                        "MediaPipe FaceLandmarker checksum mismatch (expected %s, got %s)",
                        self.expected_sha256,
                        digest,
                    )
                    return
                self._checksum_verified = True

            # Pass model_asset_buffer directly to avoid Windows non-ASCII path issues in C++ TFLite loader
            base_options = mp_python.BaseOptions(model_asset_buffer=model_bytes)
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=True,
                num_faces=self.max_faces,
                min_face_detection_confidence=self.min_confidence,
                min_face_presence_confidence=self.min_confidence,
            )
            self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            self._ready = True
            self._failure_reason = None
            logger.info("Loaded MediaPipe FaceLandmarker (%s)", self.model_path.name)
        except Exception as exc:
            self._ready = False
            self._failure_reason = f"Failed to initialize MediaPipe FaceLandmarker: {exc}"
            logger.error("MediaPipe initialization error: %s", exc)

    def status(self) -> ModelStatusSchema:
        return ModelStatusSchema(
            name="MediaPipe-FaceLandmarker",
            ready=self._ready,
            provider="CPUExecutionProvider (TFLite XNNPACK)",
            path=self.model_path.name,
            checksum_verified=self._checksum_verified,
            reason=self._failure_reason,
        )

    def detect_faces(self, bgr_image: np.ndarray) -> list[FaceAnalysis]:
        if not self._ready or self._landmarker is None:
            raise RuntimeError(self._failure_reason or "FaceLandmarker is not ready.")

        height, width = bgr_image.shape[:2]
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        try:
            with self._lock:
                result = self._landmarker.detect(mp_image)

            analyses: list[FaceAnalysis] = []
            matrices = result.facial_transformation_matrixes or []

            for idx, face_lms in enumerate(result.face_landmarks):
                points = [Point3D(x=float(lm.x), y=float(lm.y), z=float(lm.z)) for lm in face_lms]
                matrix = np.asarray(matrices[idx], dtype=np.float64) if idx < len(matrices) else None
                analyses.append(
                    analyze_face_landmarks(
                        landmarks=points,
                        image_width=width,
                        image_height=height,
                        transformation_matrix=matrix,
                    )
                )
            return analyses
        finally:
            del mp_image
            del rgb_image

    def close(self) -> None:
        with self._lock:
            if self._landmarker is not None:
                self._landmarker.close()
                self._landmarker = None
                self._ready = False
