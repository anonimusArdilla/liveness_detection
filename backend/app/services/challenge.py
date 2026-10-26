"""Server-side active liveness challenge generation and movement/expression verification."""

from __future__ import annotations

import secrets
from dataclasses import dataclass

from app.core.config import Settings
from app.core.errors import AppError, ErrorCode
from app.schemas.liveness import ChallengeType
from app.vision.face_landmarks import FaceAnalysis, FacialExpression, HeadPose

CHALLENGE_INSTRUCTIONS: dict[ChallengeType, str] = {
    ChallengeType.CENTER_FACE: (
        "Keep your face centered, look straight at the camera, and keep a neutral expression."
    ),
    ChallengeType.TURN_LEFT: "Slowly turn your head to your left.",
    ChallengeType.TURN_RIGHT: "Slowly turn your head to your right.",
    ChallengeType.SMILE: "Look at the camera and smile broadly.",
    ChallengeType.BLINK: "Look at the camera and blink slowly: close both eyes briefly, then open them.",
    ChallengeType.RAISE_EYEBROWS: "Look at the camera and raise your eyebrows high.",
}

# Mandatory first step establishing the frontal, neutral reference frame and portrait capture.
INITIAL_CHALLENGE: ChallengeType = ChallengeType.CENTER_FACE

# Pool from which exactly THREE follow-up challenges are drawn per session (cryptographically random).
RANDOM_CHALLENGE_POOL: tuple[ChallengeType, ...] = (
    ChallengeType.TURN_LEFT,
    ChallengeType.TURN_RIGHT,
    ChallengeType.SMILE,
    ChallengeType.BLINK,
    ChallengeType.RAISE_EYEBROWS,
)

# Total steps per session: INITIAL_CHALLENGE + exactly three random challenges.
TOTAL_CHALLENGE_STEPS = 4

POSE_CHALLENGES: frozenset[ChallengeType] = frozenset(
    {ChallengeType.CENTER_FACE, ChallengeType.TURN_LEFT, ChallengeType.TURN_RIGHT}
)
EXPRESSION_CHALLENGES: frozenset[ChallengeType] = frozenset(
    {ChallengeType.SMILE, ChallengeType.BLINK, ChallengeType.RAISE_EYEBROWS}
)

# Actionable guidance returned when a challenge threshold was not reached.
CHALLENGE_FAILURE_HINTS: dict[ChallengeType, str] = {
    ChallengeType.CENTER_FACE: (
        "Look straight ahead at the camera, keep your head level, and relax your face "
        "to a neutral expression."
    ),
    ChallengeType.TURN_LEFT: "Turn your head further to your left while keeping your face visible.",
    ChallengeType.TURN_RIGHT: "Turn your head further to your right while keeping your face visible.",
    ChallengeType.SMILE: "Smile more broadly while keeping your face centered and visible.",
    ChallengeType.BLINK: (
        "Blink slowly right after starting the capture: close both eyes fully for a moment, "
        "then open them again."
    ),
    ChallengeType.RAISE_EYEBROWS: "Raise your eyebrows higher while keeping your face visible.",
}

EXPRESSION_UNAVAILABLE_HINT = (
    "Facial expression analysis is unavailable for this frame batch. Please try again."
)


@dataclass(frozen=True)
class ChallengeEvaluationResult:
    """Result of verifying a batch of frames against a specific active challenge."""

    passed: bool
    passing_frame_count: int
    required_passing_frames: int
    feedback_message: str
    best_frame_index: int | None


class ActiveChallengeService:
    """Generates randomized active challenges and verifies head-pose/expression responses server-side."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._rng = secrets.SystemRandom()

    def generate_challenge_sequence(self) -> list[tuple[ChallengeType, str]]:
        """Generate a randomized challenge sequence controlled by the server.

        Always begins with `CENTER_FACE` to establish a frontal, neutral reference frame and
        high-quality portrait capture, followed by EXACTLY THREE unique challenges drawn with 
        a CSPRNG from `RANDOM_CHALLENGE_POOL` (head turns, smile, blink, or eyebrow raise). 
        The follow-up challenges are only disclosed to the client sequentially after the 
        previous challenge has been verified.
        """
        follow_ups = self._rng.sample(RANDOM_CHALLENGE_POOL, 3)
        sequence: list[ChallengeType] = [INITIAL_CHALLENGE, *follow_ups]
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
        """Check whether a single frame's 3D head pose satisfies the target challenge's pose rule.

        Expression challenges have no pose requirement of their own (they are evaluated by
        `is_frame_matching_challenge`) and therefore return False here.
        """
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

    def is_frame_matching_challenge(
        self,
        challenge_type: ChallengeType,
        analysis: FaceAnalysis,
    ) -> bool:
        """Check whether a single frame satisfies a sustained (non-temporal) challenge.

        - CENTER_FACE: frontal head pose AND, when expression data is available, a neutral
          expression (no smile / raised eyebrows). The neutral gate guarantees that a single
          static image cannot satisfy both CENTER_FACE and a later SMILE / RAISE_EYEBROWS step.
          It is applied uniformly regardless of the follow-up challenge so the CENTER_FACE
          behaviour never leaks which challenge comes next.
        - TURN_LEFT / TURN_RIGHT: yaw thresholds.
        - SMILE / RAISE_EYEBROWS: blendshape activation thresholds; fail closed (False) when
          expression data is unavailable.
        - BLINK is temporal and evaluated batch-wide in `evaluate_challenge_batch`.
        """
        if challenge_type in POSE_CHALLENGES:
            if not self.is_pose_matching_challenge(challenge_type, analysis.head_pose):
                return False
            if challenge_type == ChallengeType.CENTER_FACE and analysis.expression is not None:
                neutral_max = self.settings.expression_neutral_max_score
                return (
                    analysis.expression.smile <= neutral_max
                    and analysis.expression.brow_raise <= neutral_max
                )
            return True

        expression = analysis.expression
        if expression is None:
            return False
        if challenge_type == ChallengeType.SMILE:
            return expression.smile >= self.settings.smile_min_score
        if challenge_type == ChallengeType.RAISE_EYEBROWS:
            return expression.brow_raise >= self.settings.eyebrow_raise_min_score
        return False

    def evaluate_challenge_batch(
        self,
        challenge_type: ChallengeType,
        valid_face_analyses: list[tuple[int, FaceAnalysis]],
    ) -> ChallengeEvaluationResult:
        """Evaluate quality-passing frames in a batch against the current challenge."""
        if challenge_type == ChallengeType.BLINK:
            return self._evaluate_blink_batch(valid_face_analyses)

        required = min(self.settings.min_challenge_passing_frames, self.settings.min_frames_per_batch)
        analyses_by_idx = dict(valid_face_analyses)
        matching_indices: list[int] = [
            frame_idx
            for frame_idx, analysis in valid_face_analyses
            if self.is_frame_matching_challenge(challenge_type, analysis)
        ]

        passed = len(matching_indices) >= required
        best_idx: int | None = None
        if matching_indices:
            if challenge_type == ChallengeType.CENTER_FACE:
                # Pick the frame closest to dead-center yaw/pitch
                best_idx = min(
                    matching_indices,
                    key=lambda idx: (
                        abs(analyses_by_idx[idx].head_pose.yaw)
                        + abs(analyses_by_idx[idx].head_pose.pitch)
                    ),
                )
            elif challenge_type == ChallengeType.SMILE:
                best_idx = max(matching_indices, key=lambda idx: _expr(analyses_by_idx[idx]).smile)
            elif challenge_type == ChallengeType.RAISE_EYEBROWS:
                best_idx = max(
                    matching_indices, key=lambda idx: _expr(analyses_by_idx[idx]).brow_raise
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

        return ChallengeEvaluationResult(
            passed=False,
            passing_frame_count=len(matching_indices),
            required_passing_frames=required,
            feedback_message=self._failure_hint(challenge_type, valid_face_analyses),
            best_frame_index=None,
        )

    def _evaluate_blink_batch(
        self,
        valid_face_analyses: list[tuple[int, FaceAnalysis]],
    ) -> ChallengeEvaluationResult:
        """Verify a blink as an open<->closed eye-state TRANSITION within one frame batch.

        A blink is only accepted when the same batch contains at least one frame with BOTH
        eyes closed and at least one frame with BOTH eyes open. A static image (eyes open or
        closed) can therefore never pass this challenge on its own. Frames must also keep the
        head roughly frontal so a turned head cannot fake eye-closure activations.
        """
        closed_indices: list[int] = []
        open_indices: list[int] = []
        expression_available = False

        for frame_idx, analysis in valid_face_analyses:
            expression = analysis.expression
            if expression is None:
                continue
            expression_available = True
            if not self.is_pose_matching_challenge(ChallengeType.CENTER_FACE, analysis.head_pose):
                continue
            if expression.eyes_closed >= self.settings.blink_closed_min_score:
                closed_indices.append(frame_idx)
            elif expression.eyes_open_max <= self.settings.blink_open_max_score:
                open_indices.append(frame_idx)

        # One closed frame + one open frame are required to prove an eye-state transition.
        required = 1
        passed = len(closed_indices) >= required and len(open_indices) >= 1

        if passed:
            analyses_by_idx = dict(valid_face_analyses)
            # Prefer the most clearly open-eyed frame for passive anti-spoofing sampling.
            best_idx = min(open_indices, key=lambda idx: _expr(analyses_by_idx[idx]).eyes_open_max)
            return ChallengeEvaluationResult(
                passed=True,
                passing_frame_count=len(closed_indices),
                required_passing_frames=required,
                feedback_message=f"Challenge '{ChallengeType.BLINK.value}' verified.",
                best_frame_index=best_idx,
            )

        return ChallengeEvaluationResult(
            passed=False,
            passing_frame_count=len(closed_indices),
            required_passing_frames=required,
            feedback_message=(
                CHALLENGE_FAILURE_HINTS[ChallengeType.BLINK]
                if expression_available
                else EXPRESSION_UNAVAILABLE_HINT
            ),
            best_frame_index=None,
        )

    @staticmethod
    def _failure_hint(
        challenge_type: ChallengeType,
        valid_face_analyses: list[tuple[int, FaceAnalysis]],
    ) -> str:
        if challenge_type in EXPRESSION_CHALLENGES and all(
            analysis.expression is None for _, analysis in valid_face_analyses
        ):
            return EXPRESSION_UNAVAILABLE_HINT
        return CHALLENGE_FAILURE_HINTS.get(
            challenge_type, "Follow the on-screen instruction and try again."
        )


def _expr(analysis: FaceAnalysis) -> FacialExpression:
    """Return the (non-None) expression of an analysis already verified to carry one."""
    if analysis.expression is None:  # pragma: no cover - guarded by callers
        raise ValueError("FaceAnalysis has no expression data.")
    return analysis.expression
