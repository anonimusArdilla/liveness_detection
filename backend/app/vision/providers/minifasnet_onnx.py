"""ONNX Runtime MiniFASNet passive anti-spoofing provider (V2 + V1SE multi-scale fusion)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from app.core.logging import get_logger
from app.schemas.liveness import ModelStatusSchema
from app.vision.anti_spoofing import (
    AntiSpoofingProvider,
    BatchSpoofEvaluation,
    FrameSpoofScore,
)
from app.vision.face_landmarks import BoundingBox

logger = get_logger(__name__)


@dataclass
class LoadedScaleModel:
    """Internal holder for a single scale-specific MiniFASNet ONNX session."""

    name: str
    path: Path
    scale: float
    session: ort.InferenceSession
    input_name: str
    output_name: str
    input_height: int
    input_width: int
    checksum_verified: bool


def crop_face_minivision(
    bgr_image: np.ndarray,
    bbox: BoundingBox,
    scale: float,
    out_w: int = 80,
    out_h: int = 80,
) -> np.ndarray:
    """Crop and resize face patch matching upstream minivision-ai/Silent-Face-Anti-Spoofing.

    Faithfully implements `CropImage._get_new_box` from upstream `src/generate_patches.py`:
    expands the bounding box around its center by `min((src_h-1)/box_h, (src_w-1)/box_w, scale)`,
    shifts box edges inward when clamping against image borders so aspect ratio is preserved,
    and resizes to `(out_w, out_h)` (80x80).
    """
    src_h, src_w = bgr_image.shape[:2]
    x = float(bbox.x)
    y = float(bbox.y)
    box_w = float(max(1, bbox.width))
    box_h = float(max(1, bbox.height))

    effective_scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))
    new_width = box_w * effective_scale
    new_height = box_h * effective_scale
    center_x = box_w / 2.0 + x
    center_y = box_h / 2.0 + y

    left_top_x = center_x - new_width / 2.0
    left_top_y = center_y - new_height / 2.0
    right_bottom_x = center_x + new_width / 2.0
    right_bottom_y = center_y + new_height / 2.0

    if left_top_x < 0:
        right_bottom_x -= left_top_x
        left_top_x = 0.0

    if left_top_y < 0:
        right_bottom_y -= left_top_y
        left_top_y = 0.0

    if right_bottom_x > src_w - 1:
        left_top_x -= right_bottom_x - src_w + 1
        right_bottom_x = float(src_w - 1)

    if right_bottom_y > src_h - 1:
        left_top_y -= right_bottom_y - src_h + 1
        right_bottom_y = float(src_h - 1)

    x1 = max(0, int(left_top_x))
    y1 = max(0, int(left_top_y))
    x2 = min(src_w - 1, int(right_bottom_x))
    y2 = min(src_h - 1, int(right_bottom_y))

    patch = bgr_image[y1 : y2 + 1, x1 : x2 + 1]
    if patch.size == 0:
        return cv2.resize(bgr_image, (out_w, out_h))
    return cv2.resize(patch, (out_w, out_h))


def preprocess_minifasnet_patch(patch_bgr: np.ndarray) -> np.ndarray:
    """Convert 80x80 BGR uint8 patch to (1, 3, 80, 80) float32 tensor in [0.0, 255.0].

    Upstream `Silent-Face-Anti-Spoofing/src/data_io/functional.py` (`to_tensor`) converts
    numpy HWC uint8 arrays to CHW float32 without dividing by 255.
    """
    tensor = patch_bgr.astype(np.float32)
    tensor = np.transpose(tensor, (2, 0, 1))
    return np.expand_dims(tensor, axis=0)


def softmax_logits(logits: np.ndarray) -> np.ndarray:
    """Compute numerically stable softmax over 1D or 2D logits array."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    return np.asarray(exp_vals / np.sum(exp_vals, axis=-1, keepdims=True), dtype=np.float64)


class MiniFASNetOnnxProvider(AntiSpoofingProvider):
    """Production ONNX Runtime provider for MiniFASNetV2 and optional MiniFASNetV1SE fusion."""

    def __init__(
        self,
        primary_model_path: Path,
        primary_scale: float = 2.7,
        primary_sha256: str | None = None,
        secondary_model_path: Path | None = None,
        secondary_scale: float = 4.0,
        secondary_sha256: str | None = None,
        verify_checksum: bool = True,
        execution_provider: str = "CPUExecutionProvider",
        intra_op_threads: int = 2,
        inter_op_threads: int = 1,
    ) -> None:
        self.primary_model_path = primary_model_path
        self.primary_scale = primary_scale
        self.primary_sha256 = primary_sha256
        self.secondary_model_path = secondary_model_path
        self.secondary_scale = secondary_scale
        self.secondary_sha256 = secondary_sha256
        self.verify_checksum = verify_checksum
        self.execution_provider = execution_provider
        self.intra_op_threads = intra_op_threads
        self.inter_op_threads = inter_op_threads

        self._models: list[LoadedScaleModel] = []
        self._statuses: list[ModelStatusSchema] = []
        self._ready: bool = False

        self._load_all_models()

    def _load_single_model(
        self,
        name: str,
        path: Path,
        scale: float,
        expected_sha256: str | None,
    ) -> tuple[LoadedScaleModel | None, ModelStatusSchema]:
        if not path.is_file():
            reason = f"Model weight file not found: {path.name}"
            logger.warning("Anti-spoofing model %s unavailable: %s", name, reason)
            return None, ModelStatusSchema(
                name=name,
                ready=False,
                provider=self.execution_provider,
                path=path.name,
                checksum_verified=False,
                reason=reason,
            )

        try:
            raw_bytes = path.read_bytes()
            checksum_ok = False
            if self.verify_checksum and expected_sha256:
                digest = hashlib.sha256(raw_bytes).hexdigest()
                if digest.lower() != expected_sha256.lower():
                    reason = f"SHA-256 checksum mismatch for {path.name}"
                    logger.error(
                        "Checksum mismatch for %s (expected %s, got %s)",
                        path.name,
                        expected_sha256,
                        digest,
                    )
                    return None, ModelStatusSchema(
                        name=name,
                        ready=False,
                        provider=self.execution_provider,
                        path=path.name,
                        checksum_verified=False,
                        reason=reason,
                    )
                checksum_ok = True

            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = self.intra_op_threads
            sess_options.inter_op_num_threads = self.inter_op_threads
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            available_providers = ort.get_available_providers()
            providers = (
                [self.execution_provider, "CPUExecutionProvider"]
                if self.execution_provider in available_providers
                and self.execution_provider != "CPUExecutionProvider"
                else ["CPUExecutionProvider"]
            )

            session = ort.InferenceSession(
                raw_bytes,
                sess_options=sess_options,
                providers=providers,
            )
            input_meta = session.get_inputs()[0]
            output_meta = session.get_outputs()[0]
            shape = input_meta.shape
            if len(shape) != 4 or int(shape[1]) != 3:
                raise ValueError(f"Unexpected input tensor shape: {shape}")

            in_h = int(shape[2]) if isinstance(shape[2], int) else 80
            in_w = int(shape[3]) if isinstance(shape[3], int) else 80

            # Perform dry-run warmup to verify ONNX graph execution
            dummy = np.zeros((1, 3, in_h, in_w), dtype=np.float32)
            out = session.run([output_meta.name], {input_meta.name: dummy})[0]
            if out.ndim != 2 or out.shape[1] != 3:
                raise ValueError(f"Unexpected output tensor shape: {out.shape}")

            loaded = LoadedScaleModel(
                name=name,
                path=path,
                scale=scale,
                session=session,
                input_name=input_meta.name,
                output_name=output_meta.name,
                input_height=in_h,
                input_width=in_w,
                checksum_verified=checksum_ok,
            )
            status = ModelStatusSchema(
                name=name,
                ready=True,
                provider=providers[0],
                path=path.name,
                checksum_verified=checksum_ok,
                reason=None,
            )
            logger.info("Loaded ONNX anti-spoofing model %s (scale=%.1f)", name, scale)
            return loaded, status
        except Exception as exc:
            reason = f"Failed to load ONNX model {path.name}: {exc}"
            logger.error("ONNX model initialization failure (%s): %s", name, exc)
            return None, ModelStatusSchema(
                name=name,
                ready=False,
                provider=self.execution_provider,
                path=path.name,
                checksum_verified=False,
                reason=reason,
            )

    def _load_all_models(self) -> None:
        self._models.clear()
        self._statuses.clear()

        primary_loaded, primary_status = self._load_single_model(
            name="MiniFASNetV2 (scale 2.7)",
            path=self.primary_model_path,
            scale=self.primary_scale,
            expected_sha256=self.primary_sha256,
        )
        self._statuses.append(primary_status)
        if primary_loaded is not None:
            self._models.append(primary_loaded)

        if self.secondary_model_path is not None:
            secondary_loaded, secondary_status = self._load_single_model(
                name="MiniFASNetV1SE (scale 4.0)",
                path=self.secondary_model_path,
                scale=self.secondary_scale,
                expected_sha256=self.secondary_sha256,
            )
            self._statuses.append(secondary_status)
            if secondary_loaded is not None:
                self._models.append(secondary_loaded)
            else:
                # Fail closed if a configured secondary model fails to load
                self._ready = False
                return

        self._ready = primary_loaded is not None and len(self._models) > 0

    def is_ready(self) -> bool:
        return self._ready

    def model_statuses(self) -> list[ModelStatusSchema]:
        return list(self._statuses)

    @property
    def active_model_label(self) -> str:
        if not self._models:
            return "unloaded"
        return "+".join(m.path.stem for m in self._models)

    def predict_frame(self, bgr_image: np.ndarray, face_bbox: BoundingBox) -> FrameSpoofScore:
        if not self._ready or not self._models:
            raise RuntimeError("Passive anti-spoofing ONNX models are not loaded.")

        prob_accumulator = np.zeros((1, 3), dtype=np.float64)
        for model in self._models:
            patch = crop_face_minivision(
                bgr_image=bgr_image,
                bbox=face_bbox,
                scale=model.scale,
                out_w=model.input_width,
                out_h=model.input_height,
            )
            input_tensor = preprocess_minifasnet_patch(patch)
            try:
                logits = model.session.run(
                    [model.output_name],
                    {model.input_name: input_tensor},
                )[0]
                probs = softmax_logits(logits)
                prob_accumulator += probs
            finally:
                del input_tensor
                del patch

        mean_probs = (prob_accumulator / len(self._models))[0]
        # Upstream Silent-Face-Anti-Spoofing 3-class taxonomy:
        # Index 1 = Real (Genuine live face)
        # Index 0 & Index 2 = Spoof / Presentation Attack (2D print, screen replay, etc.)
        real_prob = float(np.clip(mean_probs[1], 0.0, 1.0))
        spoof_prob = float(np.clip(mean_probs[0] + mean_probs[2], 0.0, 1.0))
        return FrameSpoofScore(
            real_probability=round(real_prob, 4),
            spoof_probability=round(spoof_prob, 4),
            class_probabilities=(
                round(float(mean_probs[0]), 4),
                round(float(mean_probs[1]), 4),
                round(float(mean_probs[2]), 4),
            ),
        )

    def evaluate_frames(
        self,
        frames_with_boxes: list[tuple[np.ndarray, BoundingBox]],
        real_score_threshold: float,
        min_frame_score_threshold: float,
    ) -> BatchSpoofEvaluation:
        if not self._ready or not self._models:
            raise RuntimeError("Passive anti-spoofing ONNX models are not ready.")
        if not frames_with_boxes:
            raise ValueError("At least one frame is required for passive anti-spoofing evaluation.")

        frame_scores: list[float] = []
        for bgr_frame, bbox in frames_with_boxes:
            pred = self.predict_frame(bgr_frame, bbox)
            frame_scores.append(pred.real_probability)

        aggregate_real = round(float(np.mean(frame_scores)), 4)
        min_real = round(float(np.min(frame_scores)), 4)
        passed = aggregate_real >= real_score_threshold and min_real >= min_frame_score_threshold

        return BatchSpoofEvaluation(
            passed=passed,
            aggregate_real_score=aggregate_real,
            min_frame_real_score=min_real,
            frames_evaluated=len(frame_scores),
            threshold_applied=real_score_threshold,
            model_name=self.active_model_label,
        )
