"""Application configuration and operational thresholds loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    """Centralized, validated configuration for the facial liveness detection backend."""

    model_config = SettingsConfigDict(
        env_prefix="LIVENESS_",
        env_file=str(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server & Runtime
    host: str = Field(default="127.0.0.1", description="Bind host address")
    port: int = Field(default=8000, ge=1, le=65535, description="Bind port")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    environment: Literal["development", "testing", "production"] = "development"

    # Security & CORS
    cors_origins: list[str] = Field(
        default=["http://localhost:5173", "http://127.0.0.1:5173"],
        description="Explicit list of allowed frontend origins",
    )

    # Database & Storage
    database_url: str = Field(
        default=f"sqlite:///{(BACKEND_ROOT / 'data' / 'liveness.db').as_posix()}",
        description="SQLAlchemy database connection string (SQLite or PostgreSQL)",
    )
    photo_storage_dir: Path = Field(
        default=BACKEND_ROOT / "storage_data" / "accepted_photos",
        description="Private server directory outside frontend public root for accepted photos",
    )

    # Face Landmarker & Anti-Spoofing ONNX Model Configuration
    face_landmarker_model_path: Path = Field(
        default=REPO_ROOT / "models" / "face_landmarker.task",
        description="Path to MediaPipe FaceLandmarker .task model bundle",
    )
    face_landmarker_sha256: str | None = Field(
        default="64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
        description="Expected SHA-256 digest for MediaPipe FaceLandmarker model bundle",
    )
    primary_model_path: Path = Field(
        default=REPO_ROOT / "models" / "MiniFASNetV2.onnx",
        description="Path to primary MiniFASNetV2 ONNX model",
    )
    primary_model_scale: float = Field(
        default=2.7,
        gt=0.5,
        le=6.0,
        description="Face bounding box crop scale factor for primary MiniFASNetV2 model",
    )
    primary_model_sha256: str | None = Field(
        default="b32929adc2d9c34b9486f8c4c7bc97c1b69bc0ea9befefc380e4faae4e463907",
        description="Expected SHA-256 digest for primary model file integrity check",
    )
    secondary_model_path: Path | None = Field(
        default=REPO_ROOT / "models" / "MiniFASNetV1SE.onnx",
        description="Optional path to secondary MiniFASNetV1SE ONNX model for multi-scale fusion",
    )
    secondary_model_scale: float = Field(
        default=4.0,
        gt=0.5,
        le=6.0,
        description="Face bounding box crop scale factor for secondary MiniFASNetV1SE model",
    )
    secondary_model_sha256: str | None = Field(
        default="ebab7f90c7833fbccd46d3a555410e78d969db5438e169b6524be444862b3676",
        description="Expected SHA-256 digest for secondary model file integrity check",
    )
    verify_model_checksum: bool = Field(
        default=True,
        description="Verify SHA-256 checksum of model files when a digest is configured",
    )
    onnx_execution_provider: str = Field(
        default="CPUExecutionProvider",
        description="ONNX Runtime execution provider (CPUExecutionProvider default; CUDAExecutionProvider optional)",
    )
    onnx_intra_op_threads: int = Field(default=2, ge=1, le=32)
    onnx_inter_op_threads: int = Field(default=1, ge=1, le=16)

    # Image Quality Thresholds
    min_image_width: int = Field(default=320, ge=160, le=1920)
    min_image_height: int = Field(default=240, ge=120, le=1080)
    max_image_width: int = Field(default=4096, ge=640, le=8192)
    max_image_height: int = Field(default=4096, ge=480, le=8192)
    min_brightness: float = Field(default=45.0, ge=5.0, le=200.0)
    max_brightness: float = Field(default=225.0, ge=100.0, le=254.0)
    min_blur_laplacian_var: float = Field(
        default=35.0,
        ge=1.0,
        le=1000.0,
        description="Minimum variance of the Laplacian filter before flagging image as blurry",
    )
    min_face_area_ratio: float = Field(
        default=0.04,
        ge=0.01,
        le=0.5,
        description="Minimum ratio of face bounding-box area to frame area",
    )
    max_face_area_ratio: float = Field(
        default=0.75,
        ge=0.2,
        le=0.98,
        description="Maximum ratio of face bounding-box area to frame area",
    )
    max_center_offset_ratio: float = Field(
        default=0.28,
        ge=0.05,
        le=0.5,
        description="Maximum normalized Euclidean offset of face center from frame center",
    )

    # Active Challenge & Session Lifecycle Thresholds
    # Note: every session has exactly two steps (CENTER_FACE + one random challenge); the
    # sequence length is fixed in ActiveChallengeService and intentionally not configurable.
    session_ttl_seconds: int = Field(default=180, ge=30, le=900)
    center_yaw_max_deg: float = Field(default=12.0, ge=3.0, le=30.0)
    center_pitch_max_deg: float = Field(default=16.0, ge=3.0, le=35.0)
    turn_left_min_yaw_deg: float = Field(
        default=-13.0,
        le=-5.0,
        ge=-60.0,
        description="Maximum (negative) yaw angle threshold required to pass TURN_LEFT",
    )
    turn_right_min_yaw_deg: float = Field(
        default=13.0,
        ge=5.0,
        le=60.0,
        description="Minimum (positive) yaw angle threshold required to pass TURN_RIGHT",
    )
    min_challenge_passing_frames: int = Field(default=2, ge=1, le=10)

    # Expression Challenge Thresholds (MediaPipe blendshape activations in [0, 1])
    expression_neutral_max_score: float = Field(
        default=0.35,
        ge=0.1,
        le=0.8,
        description=(
            "Maximum smile / eyebrow-raise activation allowed during CENTER_FACE so the "
            "frontal reference frame is neutral and a single static photo cannot pass an "
            "expression challenge as well"
        ),
    )
    smile_min_score: float = Field(
        default=0.55,
        ge=0.2,
        le=0.95,
        description="Minimum mean mouthSmileLeft/Right activation required to pass SMILE",
    )
    eyebrow_raise_min_score: float = Field(
        default=0.5,
        ge=0.2,
        le=0.95,
        description="Minimum combined browInnerUp/browOuterUp activation required to pass RAISE_EYEBROWS",
    )
    blink_closed_min_score: float = Field(
        default=0.5,
        ge=0.25,
        le=0.95,
        description="Minimum eyeBlink activation of BOTH eyes for a frame to count as eyes-closed",
    )
    blink_open_max_score: float = Field(
        default=0.3,
        ge=0.05,
        le=0.6,
        description="Maximum eyeBlink activation of EITHER eye for a frame to count as eyes-open",
    )

    # Passive Anti-Spoofing Decision Thresholds
    spoof_real_score_threshold: float = Field(
        default=0.65,
        ge=0.1,
        le=0.99,
        description="Minimum aggregated probability of 'Real' class across sampled frames",
    )
    spoof_min_frame_score: float = Field(
        default=0.45,
        ge=0.05,
        le=0.99,
        description="Minimum single-frame probability of 'Real' class allowed in validated batch",
    )
    passive_sample_frames_max: int = Field(default=5, ge=1, le=15)

    # Request & Rate Limits
    max_request_bytes: int = Field(default=6 * 1024 * 1024, ge=262144, le=25 * 1024 * 1024)
    max_frames_per_batch: int = Field(default=10, ge=2, le=30)
    min_frames_per_batch: int = Field(default=2, ge=1, le=10)
    rate_limit_requests_per_minute: int = Field(default=60, ge=5, le=1000)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("["):
                import json

                parsed = json.loads(stripped)
                return [str(item).strip() for item in parsed if str(item).strip()]
            return [item.strip() for item in stripped.split(",") if item.strip()]
        return value

    @field_validator("face_landmarker_model_path", "primary_model_path", "photo_storage_dir", mode="before")
    @classmethod
    def resolve_relative_paths(cls, value: str | Path) -> Path:
        p = Path(value)
        if not p.is_absolute():
            return (BACKEND_ROOT / p).resolve()
        return p

    @field_validator("secondary_model_path", mode="before")
    @classmethod
    def resolve_optional_secondary_path(cls, value: str | Path | None) -> Path | None:
        if value is None or str(value).strip() == "":
            return None
        p = Path(value)
        if not p.is_absolute():
            return (BACKEND_ROOT / p).resolve()
        return p

    @model_validator(mode="after")
    def validate_expression_thresholds(self) -> Settings:
        """Ensure expression thresholds leave a strict margin between neutral and active states."""
        if self.blink_open_max_score >= self.blink_closed_min_score:
            raise ValueError("blink_open_max_score must be strictly lower than blink_closed_min_score")
        if self.expression_neutral_max_score >= min(self.smile_min_score, self.eyebrow_raise_min_score):
            raise ValueError(
                "expression_neutral_max_score must be strictly lower than smile_min_score "
                "and eyebrow_raise_min_score"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()
