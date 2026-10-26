"""Facial landmark geometry, 3D head-pose and expression estimation for active liveness challenges."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

# Key MediaPipe 478-landmark indices:
# 1: Nose tip
# 10: Forehead top center
# 152: Chin bottom center
# 234: Subject's right temple/cheekbone (left side of unmirrored image)
# 454: Subject's left temple/cheekbone (right side of unmirrored image)
# 33: Subject's right eye outer corner
# 263: Subject's left eye outer corner
NOSE_TIP_IDX = 1
FOREHEAD_IDX = 10
CHIN_IDX = 152
RIGHT_CHEEK_IDX = 234
LEFT_CHEEK_IDX = 454
RIGHT_EYE_OUTER_IDX = 33
LEFT_EYE_OUTER_IDX = 263

# MediaPipe FaceLandmarker blendshape category names (ARKit-compatible, 52 categories)
# used to derive the expression scores evaluated by active expression challenges.
BLENDSHAPE_SMILE = ("mouthSmileLeft", "mouthSmileRight")
BLENDSHAPE_EYE_BLINK = ("eyeBlinkLeft", "eyeBlinkRight")
BLENDSHAPE_BROW_INNER_UP = "browInnerUp"
BLENDSHAPE_BROW_OUTER_UP = ("browOuterUpLeft", "browOuterUpRight")


@dataclass(frozen=True)
class Point3D:
    """Normalized 3D facial landmark coordinate."""

    x: float
    y: float
    z: float


@dataclass(frozen=True)
class BoundingBox:
    """Pixel-space facial bounding box [x, y, width, height]."""

    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2.0


@dataclass(frozen=True)
class HeadPose:
    """Estimated 3D head orientation in degrees from the subject's perspective.

    Sign conventions (subject perspective in unmirrored camera frame):
    - yaw: Negative when user turns head to THEIR left; positive when turning to THEIR right.
    - pitch: Negative when user tilts head down; positive when tilting head up.
    - roll: In-plane tilt angle in degrees.
    """

    yaw: float
    pitch: float
    roll: float


@dataclass(frozen=True)
class FacialExpression:
    """Facial expression activation scores in [0, 1] derived from MediaPipe blendshapes.

    - smile: Mean of left/right `mouthSmile` activations.
    - eyes_closed: MIN of left/right `eyeBlink` activations (both eyes must close; a wink
      or a single occluded eye does not count as a blink).
    - eyes_open_max: MAX of left/right `eyeBlink` activations (both eyes must be open for
      this value to be low).
    - brow_raise: Mean of `browInnerUp` and the averaged left/right `browOuterUp` activations.
    """

    smile: float
    eyes_closed: float
    eyes_open_max: float
    brow_raise: float


@dataclass(frozen=True)
class FaceAnalysis:
    """Complete geometric and landmark analysis for a single detected face.

    `expression` is None when the detector does not provide blendshape output; expression
    challenges must fail closed in that case.
    """

    bounding_box: BoundingBox
    face_area_ratio: float
    center_offset_ratio: float
    head_pose: HeadPose
    landmarks_count: int
    expression: FacialExpression | None = None


def _clamp_unit(value: float) -> float:
    """Clamp a score to [0, 1], mapping NaN/inf to 0.0 (fail-safe for downstream thresholds)."""
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))


def expression_from_blendshapes(scores: Mapping[str, float]) -> FacialExpression | None:
    """Derive smile/blink/eyebrow-raise scores from MediaPipe blendshape category scores.

    Returns None when any required category is missing so callers fail closed rather than
    silently treating a missing activation as 0.0.
    """
    required = (
        *BLENDSHAPE_SMILE,
        *BLENDSHAPE_EYE_BLINK,
        BLENDSHAPE_BROW_INNER_UP,
        *BLENDSHAPE_BROW_OUTER_UP,
    )
    if any(name not in scores for name in required):
        return None

    s = {name: _clamp_unit(float(scores[name])) for name in required}
    smile = (s[BLENDSHAPE_SMILE[0]] + s[BLENDSHAPE_SMILE[1]]) / 2.0
    blink_left, blink_right = s[BLENDSHAPE_EYE_BLINK[0]], s[BLENDSHAPE_EYE_BLINK[1]]
    outer_up = (s[BLENDSHAPE_BROW_OUTER_UP[0]] + s[BLENDSHAPE_BROW_OUTER_UP[1]]) / 2.0
    brow_raise = (s[BLENDSHAPE_BROW_INNER_UP] + outer_up) / 2.0

    return FacialExpression(
        smile=round(smile, 4),
        eyes_closed=round(min(blink_left, blink_right), 4),
        eyes_open_max=round(max(blink_left, blink_right), 4),
        brow_raise=round(brow_raise, 4),
    )


def compute_bounding_box(landmarks: list[Point3D], image_width: int, image_height: int) -> BoundingBox:
    """Compute clamped pixel-space bounding box from normalized facial landmarks."""
    xs = [lm.x * image_width for lm in landmarks]
    ys = [lm.y * image_height for lm in landmarks]

    x_min = max(0, int(math.floor(min(xs))))
    y_min = max(0, int(math.floor(min(ys))))
    x_max = min(image_width - 1, int(math.ceil(max(xs))))
    y_max = min(image_height - 1, int(math.ceil(max(ys))))

    width = max(1, x_max - x_min)
    height = max(1, y_max - y_min)
    return BoundingBox(x=x_min, y=y_min, width=width, height=height)


def estimate_head_pose(
    landmarks: list[Point3D],
    transformation_matrix: np.ndarray | None = None,
) -> HeadPose:
    """Estimate 3D Euler angles (yaw, pitch, roll) in degrees from 478 MediaPipe landmarks.

    Combines 3D landmark depth/asymmetry between the nose tip (idx 1) and lateral
    cheekbones (idx 234, 454) with optional 4x4 facial transformation matrix data.
    """
    if len(landmarks) <= LEFT_CHEEK_IDX:
        return HeadPose(yaw=0.0, pitch=0.0, roll=0.0)

    nose = landmarks[NOSE_TIP_IDX]
    forehead = landmarks[FOREHEAD_IDX]
    chin = landmarks[CHIN_IDX]
    right_cheek = landmarks[RIGHT_CHEEK_IDX]
    left_cheek = landmarks[LEFT_CHEEK_IDX]
    right_eye = landmarks[RIGHT_EYE_OUTER_IDX]
    left_eye = landmarks[LEFT_EYE_OUTER_IDX]

    # Horizontal asymmetry ratio in unmirrored camera frame:
    # right_cheek (234) is on the left side of image; left_cheek (454) is on the right side.
    # When subject turns head to THEIR left, nose moves toward left_cheek (454),
    # making dist_to_left_cheek < dist_to_right_cheek -> negative yaw.
    dist_to_right_cheek = nose.x - right_cheek.x
    dist_to_left_cheek = left_cheek.x - nose.x
    cheek_span = max(1e-5, left_cheek.x - right_cheek.x)
    asym_x = (dist_to_left_cheek - dist_to_right_cheek) / cheek_span

    # 3D depth differential between left and right cheeks (normalized by cheek span)
    z_diff = (right_cheek.z - left_cheek.z) / cheek_span
    geom_yaw = float(np.clip((asym_x * 48.0) + (z_diff * 15.0), -75.0, 75.0))

    # Vertical pitch ratio: nose position relative to forehead-chin midpoint
    face_height = max(1e-5, chin.y - forehead.y)
    mid_y = (forehead.y + chin.y) / 2.0
    rel_nose_y = (mid_y - nose.y) / (face_height / 2.0)
    # Adjust for natural anatomical nose position slightly below geometric midpoint
    geom_pitch = float(np.clip((rel_nose_y + 0.06) * 45.0, -65.0, 65.0))

    # Roll from inter-ocular line
    dy = left_eye.y - right_eye.y
    dx = max(1e-5, left_eye.x - right_eye.x)
    geom_roll = float(np.clip(math.degrees(math.atan2(dy, dx)), -65.0, 65.0))

    if transformation_matrix is not None and transformation_matrix.shape == (4, 4):
        rot = transformation_matrix[:3, :3]
        sy = math.sqrt(float(rot[0, 0] ** 2 + rot[1, 0] ** 2))
        if sy > 1e-6:
            mat_pitch = math.degrees(math.atan2(float(rot[2, 1]), float(rot[2, 2])))
            mat_yaw = math.degrees(math.atan2(float(-rot[2, 0]), sy))
        else:
            mat_pitch = math.degrees(math.atan2(float(-rot[1, 2]), float(rot[1, 1])))
            mat_yaw = math.degrees(math.atan2(float(-rot[2, 0]), sy))
        # Blend geometric landmark pose (sign-authoritative) with matrix magnitude
        if (mat_yaw * geom_yaw) > 0:
            geom_yaw = 0.65 * geom_yaw + 0.35 * mat_yaw
        if (mat_pitch * geom_pitch) > 0:
            geom_pitch = 0.65 * geom_pitch + 0.35 * mat_pitch

    return HeadPose(
        yaw=round(geom_yaw, 2),
        pitch=round(geom_pitch, 2),
        roll=round(geom_roll, 2),
    )


def analyze_face_landmarks(
    landmarks: list[Point3D],
    image_width: int,
    image_height: int,
    transformation_matrix: np.ndarray | None = None,
    blendshapes: Mapping[str, float] | None = None,
) -> FaceAnalysis:
    """Compute bounding box, area ratio, centering offset, 3D head pose and expression for a face."""
    bbox = compute_bounding_box(landmarks, image_width, image_height)
    frame_area = float(max(1, image_width * image_height))
    face_area_ratio = round(bbox.area / frame_area, 4)

    norm_cx = bbox.center_x / max(1.0, float(image_width))
    norm_cy = bbox.center_y / max(1.0, float(image_height))
    center_offset_ratio = round(
        math.hypot(norm_cx - 0.5, norm_cy - 0.5),
        4,
    )

    head_pose = estimate_head_pose(landmarks, transformation_matrix)
    return FaceAnalysis(
        bounding_box=bbox,
        face_area_ratio=face_area_ratio,
        center_offset_ratio=center_offset_ratio,
        head_pose=head_pose,
        landmarks_count=len(landmarks),
        expression=expression_from_blendshapes(blendshapes) if blendshapes is not None else None,
    )
