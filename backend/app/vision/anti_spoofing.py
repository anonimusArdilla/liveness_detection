"""Abstract interface and result structures for passive facial anti-spoofing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from app.schemas.liveness import ModelStatusSchema
from app.vision.face_landmarks import BoundingBox


@dataclass(frozen=True)
class FrameSpoofScore:
    """Single-frame passive anti-spoofing prediction."""

    real_probability: float
    spoof_probability: float
    class_probabilities: tuple[float, float, float]


@dataclass(frozen=True)
class BatchSpoofEvaluation:
    """Aggregated multi-frame passive anti-spoofing evaluation result."""

    passed: bool
    aggregate_real_score: float
    min_frame_real_score: float
    frames_evaluated: int
    threshold_applied: float
    model_name: str


class AntiSpoofingProvider(ABC):
    """Pluggable interface for passive facial presentation attack detection (PAD) models."""

    @abstractmethod
    def is_ready(self) -> bool:
        """Return True only when all required model weights are verified and loaded."""

    @abstractmethod
    def model_statuses(self) -> list[ModelStatusSchema]:
        """Return detailed readiness and integrity status for underlying model files."""

    @abstractmethod
    def predict_frame(self, bgr_image: np.ndarray, face_bbox: BoundingBox) -> FrameSpoofScore:
        """Evaluate a single BGR frame and facial bounding box."""

    @abstractmethod
    def evaluate_frames(
        self,
        frames_with_boxes: list[tuple[np.ndarray, BoundingBox]],
        real_score_threshold: float,
        min_frame_score_threshold: float,
    ) -> BatchSpoofEvaluation:
        """Evaluate a collection of frames and apply multi-frame aggregation policy."""
