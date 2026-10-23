"""Image-quality assessment and safe in-memory base64 frame decoding."""

from __future__ import annotations

import base64
import binascii

import cv2
import numpy as np

from app.core.config import Settings
from app.core.errors import AppError, ErrorCode
from app.schemas.liveness import (
    BoundingBoxSchema,
    HeadPoseSchema,
    QualityIssue,
    QualityIssueCode,
    QualityReportSchema,
)
from app.vision.face_landmarks import FaceAnalysis


def decode_base64_image(image_base64: str, max_bytes: int = 2_500_000) -> np.ndarray:
    """Decode a base64 or data-URL encoded JPEG/PNG image strictly in memory."""
    payload = image_base64.strip()
    if "," in payload and payload.lower().startswith("data:image/"):
        payload = payload.split(",", 1)[1].strip()

    if not payload:
        raise AppError(
            code=ErrorCode.IMAGE_DECODE_FAILED,
            message="Empty image payload received.",
            status_code=400,
        )

    try:
        raw_bytes = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise AppError(
            code=ErrorCode.IMAGE_DECODE_FAILED,
            message="Malformed base64 image encoding.",
            status_code=400,
        ) from exc

    if len(raw_bytes) < 64 or len(raw_bytes) > max_bytes:
        raise AppError(
            code=ErrorCode.IMAGE_DECODE_FAILED,
            message="Encoded image size is outside allowed bounds.",
            status_code=400,
        )

    buf = np.frombuffer(raw_bytes, dtype=np.uint8)
    try:
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    finally:
        del buf
        del raw_bytes

    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise AppError(
            code=ErrorCode.IMAGE_DECODE_FAILED,
            message="Unable to decode frame as a valid 3-channel color image.",
            status_code=400,
        )

    return bgr


class ImageQualityService:
    """Evaluates frame resolution, lighting, sharpness, face count, size, and positioning."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def evaluate_frame(
        self,
        bgr_image: np.ndarray,
        faces: list[FaceAnalysis],
        require_centered: bool = True,
    ) -> QualityReportSchema:
        """Run all configured image-quality and face-positioning checks on a frame."""
        height, width = bgr_image.shape[:2]
        issues: list[QualityIssue] = []

        # 1. Resolution check
        if width < self.settings.min_image_width or height < self.settings.min_image_height:
            issues.append(
                QualityIssue(
                    code=QualityIssueCode.RESOLUTION_TOO_LOW,
                    message=(
                        f"Camera resolution ({width}x{height}) is below the minimum required "
                        f"({self.settings.min_image_width}x{self.settings.min_image_height})."
                    ),
                )
            )
        elif width > self.settings.max_image_width or height > self.settings.max_image_height:
            issues.append(
                QualityIssue(
                    code=QualityIssueCode.RESOLUTION_TOO_HIGH,
                    message=(
                        f"Image resolution ({width}x{height}) exceeds the maximum allowed "
                        f"({self.settings.max_image_width}x{self.settings.max_image_height})."
                    ),
                )
            )

        # 2. Brightness & blur checks on grayscale conversion
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        try:
            brightness = round(float(np.mean(gray)), 2)
            blur_score = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)
        finally:
            del gray

        if brightness < self.settings.min_brightness:
            issues.append(
                QualityIssue(
                    code=QualityIssueCode.LIGHTING_TOO_DARK,
                    message="Lighting is too dim. Please move to a brighter area or face a light source.",
                )
            )
        elif brightness > self.settings.max_brightness:
            issues.append(
                QualityIssue(
                    code=QualityIssueCode.LIGHTING_TOO_BRIGHT,
                    message="Lighting is too bright or overexposed. Avoid strong backlighting or glare.",
                )
            )

        if blur_score < self.settings.min_blur_laplacian_var:
            issues.append(
                QualityIssue(
                    code=QualityIssueCode.IMAGE_TOO_BLURRY,
                    message="Image appears blurry. Hold your device steady and ensure the camera lens is clean.",
                )
            )

        # 3. Face presence & count checks
        face_count = len(faces)
        if face_count == 0:
            issues.append(
                QualityIssue(
                    code=QualityIssueCode.NO_FACE_DETECTED,
                    message="No face detected. Position your face clearly in front of the camera.",
                )
            )
            return QualityReportSchema(
                passed=False,
                width=width,
                height=height,
                brightness=brightness,
                blur_score=blur_score,
                face_count=0,
                issues=issues,
            )

        if face_count > 1:
            issues.append(
                QualityIssue(
                    code=QualityIssueCode.MULTIPLE_FACES_DETECTED,
                    message="Multiple faces detected. Ensure only one person is visible in the frame.",
                )
            )
            return QualityReportSchema(
                passed=False,
                width=width,
                height=height,
                brightness=brightness,
                blur_score=blur_score,
                face_count=face_count,
                issues=issues,
            )

        # 4. Single-face size & positioning checks
        face = faces[0]
        if face.face_area_ratio < self.settings.min_face_area_ratio:
            issues.append(
                QualityIssue(
                    code=QualityIssueCode.FACE_TOO_SMALL,
                    message="Your face is too far from the camera. Move closer so your face fills the guide.",
                )
            )
        elif face.face_area_ratio > self.settings.max_face_area_ratio:
            issues.append(
                QualityIssue(
                    code=QualityIssueCode.FACE_TOO_CLOSE,
                    message="Your face is too close to the camera. Move slightly back so your full face is visible.",
                )
            )

        if require_centered and face.center_offset_ratio > self.settings.max_center_offset_ratio:
            issues.append(
                QualityIssue(
                    code=QualityIssueCode.FACE_NOT_CENTERED,
                    message="Your face is outside the center guide area. Center your face in the frame.",
                )
            )

        return QualityReportSchema(
            passed=len(issues) == 0,
            width=width,
            height=height,
            brightness=brightness,
            blur_score=blur_score,
            face_count=1,
            face_area_ratio=face.face_area_ratio,
            center_offset_ratio=face.center_offset_ratio,
            bounding_box=BoundingBoxSchema(
                x=face.bounding_box.x,
                y=face.bounding_box.y,
                width=face.bounding_box.width,
                height=face.bounding_box.height,
            ),
            head_pose=HeadPoseSchema(
                yaw=face.head_pose.yaw,
                pitch=face.head_pose.pitch,
                roll=face.head_pose.roll,
            ),
            issues=issues,
        )
