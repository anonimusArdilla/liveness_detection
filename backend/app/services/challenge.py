"""Server-side active liveness challenge generation and movement verification."""

from __future__ import annotations

import secrets
from dataclasses import dataclass

from app.core.config import Settings
from app.core.errors import AppError, ErrorCode
from app.schemas.liveness import ChallengeType
from app.vision.face_landmarks import FaceAnalysis, HeadPose

CHALLENGE_INSTRUCTIONS: dict[ChallengeType, str] = {
    ChallengeType.CENTER_FACE: "Keep your face centered and look straight at the camera.",
    ChallengeType.TURN_LEFT: "Slowly turn your head to your left.",
    ChallengeType.TURN_RIGHT: "Slowly turn your head to your right.",
}


@dataclass(frozen=True)
class ChallengeEvaluationResult:
    """Result of verifying a batch of frames against a specific active challenge."""

    passed: bool
    passing_frame_count: int
    required_passing_frames: int
    feedback_message: str
    best_frame_index: int | None


class ActiveChallengeService:
    """Generates randomized active challenges and verifies head-pose response server-side."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._rng = secrets.SystemRandom()

    def generate_challenge_sequence(self) -> list[tuple[ChallengeType, str]]:
        """Generate a randomized challenge sequence controlled by the server.

        Always begins with `CENTER_FACE` to establish a frontal reference frame and
        high-quality portrait capture, followed by a cryptographically randomized ordering
        of `TURN_LEFT` and `TURN_RIGHT` (plus additional alternating turns if configured).
        """
        count = self.settings.challenge_count
        directional = [ChallengeType.TURN_LEFT, ChallengeType.TURN_RIGHT]
        self._rng.shuffle(directional)

        sequence: list[ChallengeType] = [ChallengeType.CENTER_FACE]
        for idx in range(count - 1):
            sequence.append(directional[idx % len(directional)])

        return [(ctype, CHALLENGE_INSTRUCTIONS[ctype]) for ctype in sequence]

    def validate_frame_timestamps(
        self,
        timestamps_ms: list[int],
        last_session_timestamp_ms: int,
    ) -> int:
        """Verify that frame timestamps are strictly increasing and fresh relative to session state."""
        if not timestamps_ms:
            raise AppError(
                code=ErrorCode.FRAME_ORDER_INVALID,
                message="Frame batch cannot be empty.",
                status_code=400,
            )

        prev = last_session_timestamp_ms
        for ts in timestamps_ms:
            if ts <= prev:
                raise AppError(
                    code=ErrorCode.FRAME_ORDER_INVALID,
                    message=(
                        "Out-of-order or replayed frame timestamp detected. "
                        "Timestamps must be strictly increasing."
                    ),
                    status_code=400,
                )
            prev = ts

        return prev

    def is_pose_matching_challenge(
        self,
        challenge_type: ChallengeType,
        pose: HeadPose,
    ) -> bool:
        """Check whether a single frame's 3D head pose satisfies the target challenge."""
        if challenge_type == ChallengeType.CENTER_FACE:
            return (
                abs(pose.yaw) <= self.settings.center_yaw_max_deg
                and abs(pose.pitch) <= self.settings.center_pitch_max_deg
            )
        if challenge_type == ChallengeType.TURN_LEFT:
            return pose.yaw <= self.settings.turn_left_min_yaw_deg
        if challenge_type == ChallengeType.TURN_RIGHT:
            return pose.yaw >= self.settings.turn_right_min_yaw_deg
        return False

    def evaluate_challenge_batch(
        self,
        challenge_type: ChallengeType,
        valid_face_analyses: list[tuple[int, FaceAnalysis]],
    ) -> ChallengeEvaluationResult:
        """Evaluate quality-passing frames in a batch against the current challenge."""
        required = min(self.settings.min_challenge_passing_frames, self.settings.min_frames_per_batch)
        matching_indices: list[int] = []

        for frame_idx, analysis in valid_face_analyses:
            if self.is_pose_matching_challenge(challenge_type, analysis.head_pose):
                matching_indices.append(frame_idx)

        passed = len(matching_indices) >= required
        best_idx: int | None = None
        if matching_indices:
            if challenge_type == ChallengeType.CENTER_FACE:
                # Pick the frame closest to dead-center yaw/pitch
                best_idx = min(
                    matching_indices,
                    key=lambda idx: (
                        abs(next(a.head_pose.yaw for i, a in valid_face_analyses if i == idx))
                        + abs(next(a.head_pose.pitch for i, a in valid_face_analyses if i == idx))
                    ),
                )
            else:
                best_idx = matching_indices[-1]

        if passed:
            return ChallengeEvaluationResult(
                passed=True,
                passing_frame_count=len(matching_indices),
                required_passing_frames=required,
                feedback_message=f"Challenge '{challenge_type.value}' verified.",
                best_frame_index=best_idx,
            )

        # Build actionable guidance when the pose threshold was not reached
        if challenge_type == ChallengeType.CENTER_FACE:
            hint = "Look straight ahead at the camera and keep your head level."
        elif challenge_type == ChallengeType.TURN_LEFT:
            hint = "Turn your head further to your left while keeping your face visible."
        else:
            hint = "Turn your head further to your right while keeping your face visible."

        return ChallengeEvaluationResult(
            passed=False,
            passing_frame_count=len(matching_indices),
            required_passing_frames=required,
            feedback_message=hint,
            best_frame_index=None,
        )
