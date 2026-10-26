"""Pydantic schemas for the facial liveness detection REST API (/api/v1)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LivenessOutcome(StrEnum):
    """Explicit final or intermediate outcomes for a capture session."""

    PENDING = "PENDING"
    QUALITY_FAILED = "QUALITY_FAILED"
    CHALLENGE_FAILED = "CHALLENGE_FAILED"
    SPOOF_SUSPECTED = "SPOOF_SUSPECTED"
    VALIDATION_UNAVAILABLE = "VALIDATION_UNAVAILABLE"
    VALIDATED = "VALIDATED"
    EXPIRED = "EXPIRED"


class ChallengeType(StrEnum):
    """Supported server-controlled active liveness challenge actions.

    `CENTER_FACE` is always the first step; exactly one of the remaining actions is
    drawn at random by the server as the second (final) step.
    """

    CENTER_FACE = "CENTER_FACE"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    SMILE = "SMILE"
    BLINK = "BLINK"
    RAISE_EYEBROWS = "RAISE_EYEBROWS"


class QualityIssueCode(StrEnum):
    """Actionable image-quality and face-positioning failure codes."""

    RESOLUTION_TOO_LOW = "RESOLUTION_TOO_LOW"
    RESOLUTION_TOO_HIGH = "RESOLUTION_TOO_HIGH"
    NO_FACE_DETECTED = "NO_FACE_DETECTED"
    MULTIPLE_FACES_DETECTED = "MULTIPLE_FACES_DETECTED"
    FACE_TOO_SMALL = "FACE_TOO_SMALL"
    FACE_TOO_CLOSE = "FACE_TOO_CLOSE"
    FACE_NOT_CENTERED = "FACE_NOT_CENTERED"
    LIGHTING_TOO_DARK = "LIGHTING_TOO_DARK"
    LIGHTING_TOO_BRIGHT = "LIGHTING_TOO_BRIGHT"
    IMAGE_TOO_BLURRY = "IMAGE_TOO_BLURRY"


class QualityIssue(BaseModel):
    """Single actionable quality issue reported to guide the user."""

    code: QualityIssueCode
    message: str


class BoundingBoxSchema(BaseModel):
    """Facial bounding box in pixel coordinates."""

    x: int
    y: int
    width: int
    height: int


class HeadPoseSchema(BaseModel):
    """Estimated 3D head pose Euler angles in degrees."""

    yaw: float = Field(description="Horizontal head rotation in degrees (-left, +right)")
    pitch: float = Field(description="Vertical head tilt in degrees (-down, +up)")
    roll: float = Field(description="In-plane head tilt in degrees")


class QualityReportSchema(BaseModel):
    """Comprehensive image-quality assessment for a frame."""

    passed: bool
    width: int
    height: int
    brightness: float
    blur_score: float
    face_count: int
    face_area_ratio: float | None = None
    center_offset_ratio: float | None = None
    bounding_box: BoundingBoxSchema | None = None
    head_pose: HeadPoseSchema | None = None
    issues: list[QualityIssue] = Field(default_factory=list)


class ModelStatusSchema(BaseModel):
    """Readiness details for a single computer-vision model component."""

    name: str
    ready: bool
    provider: str
    path: str | None = None
    checksum_verified: bool = False
    reason: str | None = None


class HealthResponse(BaseModel):
    """Response schema for liveness and readiness health endpoints."""

    status: str = Field(description="'ok' when alive, 'ready' or 'not_ready' for readiness")
    ready: bool = Field(description="True only when all required models and storage are ready")
    version: str
    models: list[ModelStatusSchema]
    database_ready: bool
    storage_ready: bool


class CreateSessionRequest(BaseModel):
    """Request payload for creating a new capture session."""

    model_config = ConfigDict(extra="forbid")

    registration_ref: str | None = Field(
        default=None,
        max_length=128,
        pattern=r"^[a-zA-Z0-9_.:-]*$",
        description="Optional external registration or user flow correlation identifier",
    )


class ChallengeInstructionSchema(BaseModel):
    """Progressive instruction for the current active challenge step."""

    challenge_id: str
    step_index: int
    total_steps: int
    challenge_type: ChallengeType
    instruction: str
    completed: bool


class CreateSessionResponse(BaseModel):
    """Response returned when a new capture session is created."""

    session_id: str
    session_nonce: str = Field(
        description="One-time cryptographic nonce required for frame submission and finalization"
    )
    outcome: LivenessOutcome
    expires_at: datetime
    total_challenges: int
    current_challenge: ChallengeInstructionSchema


class CurrentChallengeResponse(BaseModel):
    """Response describing the current challenge step without leaking future challenges."""

    session_id: str
    outcome: LivenessOutcome
    expires_at: datetime
    all_challenges_completed: bool
    current_challenge: ChallengeInstructionSchema | None = None


class FrameInputSchema(BaseModel):
    """Single camera frame submitted within a bounded challenge batch."""

    model_config = ConfigDict(extra="forbid")

    timestamp_ms: int = Field(
        gt=0,
        description="Monotonically increasing capture timestamp in milliseconds",
    )
    image_base64: str = Field(
        min_length=32,
        max_length=2_500_000,
        description="JPEG or PNG encoded frame as base64 or data URL",
    )


class SubmitChallengeFramesRequest(BaseModel):
    """Request payload for submitting a batch of frames for the current active challenge."""

    model_config = ConfigDict(extra="forbid")

    session_nonce: str = Field(min_length=16, max_length=128)
    challenge_id: str = Field(min_length=8, max_length=64)
    frames: list[FrameInputSchema] = Field(min_length=1, max_length=30)


class SubmitChallengeFramesResponse(BaseModel):
    """Response after evaluating a batch of frames against quality and active challenge rules."""

    session_id: str
    outcome: LivenessOutcome
    challenge_passed: bool
    step_index: int
    total_steps: int
    feedback_message: str
    quality_report: QualityReportSchema
    all_challenges_completed: bool
    next_challenge: ChallengeInstructionSchema | None = None


class CompleteSessionRequest(BaseModel):
    """Request payload to finalize the capture session and run passive anti-spoofing."""

    model_config = ConfigDict(extra="forbid")

    session_nonce: str = Field(min_length=16, max_length=128)


class PassiveSpoofReportSchema(BaseModel):
    """Summary of passive ONNX MiniFASNet anti-spoofing evaluation."""

    passed: bool
    aggregate_real_score: float
    min_frame_real_score: float
    frames_evaluated: int
    threshold_applied: float
    model_name: str


class SessionResultResponse(BaseModel):
    """Final or current validation outcome for a capture session."""

    session_id: str
    registration_ref: str | None = None
    outcome: LivenessOutcome
    validated: bool
    created_at: datetime
    expires_at: datetime
    completed_at: datetime | None = None
    reason: str
    quality_passed: bool
    challenges_passed: bool
    passive_spoof_passed: bool
    passive_report: PassiveSpoofReportSchema | None = None
    quality_issues: list[QualityIssue] = Field(default_factory=list)
    photo_available: bool = False
    photo_access_token: str | None = Field(
        default=None,
        description="Protected one-session token returned upon finalization to retrieve the accepted photo",
    )
    verification_signature: str | None = Field(
        default=None,
        description="SHA-256 integrity digest binding session_id, registration_ref, outcome, and timestamp",
    )


class ErrorResponse(BaseModel):
    """Standardized error envelope returned by all endpoints."""

    error: dict[str, Any]
