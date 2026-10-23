"""FastAPI dependency injection, service container, and rate-limiting middleware."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from collections.abc import Generator
from dataclasses import dataclass

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.errors import AppError, ErrorCode
from app.repositories.database import DatabaseManager, get_db_manager
from app.repositories.session_repository import SessionRepository
from app.services.capture_session import CaptureSessionService, TransientSessionBufferStore
from app.services.challenge import ActiveChallengeService
from app.services.image_quality import ImageQualityService
from app.services.liveness_pipeline import LivenessPipelineService
from app.storage.base import PhotoStorageProvider
from app.storage.local_storage import LocalPrivatePhotoStorage
from app.vision.anti_spoofing import AntiSpoofingProvider
from app.vision.face_detector import FaceDetectorProvider, MediaPipeFaceDetector
from app.vision.providers.minifasnet_onnx import MiniFASNetOnnxProvider


@dataclass
class ServiceContainer:
    """Holds singleton service instances for the application lifecycle."""

    settings: Settings
    db_manager: DatabaseManager
    face_detector: FaceDetectorProvider
    anti_spoofing: AntiSpoofingProvider
    photo_storage: PhotoStorageProvider
    quality_service: ImageQualityService
    challenge_service: ActiveChallengeService
    buffer_store: TransientSessionBufferStore
    session_service: CaptureSessionService
    pipeline_service: LivenessPipelineService


_container: ServiceContainer | None = None
_container_lock = threading.Lock()


def build_service_container(
    settings: Settings | None = None,
    face_detector: FaceDetectorProvider | None = None,
    anti_spoofing: AntiSpoofingProvider | None = None,
    photo_storage: PhotoStorageProvider | None = None,
) -> ServiceContainer:
    """Construct a complete ServiceContainer from settings and optional provider overrides."""
    cfg = settings or get_settings()
    db_mgr = get_db_manager(cfg)

    detector = face_detector or MediaPipeFaceDetector(
        model_path=cfg.face_landmarker_model_path,
        expected_sha256=cfg.face_landmarker_sha256,
        verify_checksum=cfg.verify_model_checksum,
    )
    pad_provider = anti_spoofing or MiniFASNetOnnxProvider(
        primary_model_path=cfg.primary_model_path,
        primary_scale=cfg.primary_model_scale,
        primary_sha256=cfg.primary_model_sha256,
        secondary_model_path=cfg.secondary_model_path,
        secondary_scale=cfg.secondary_model_scale,
        secondary_sha256=cfg.secondary_model_sha256,
        verify_checksum=cfg.verify_model_checksum,
        execution_provider=cfg.onnx_execution_provider,
        intra_op_threads=cfg.onnx_intra_op_threads,
        inter_op_threads=cfg.onnx_inter_op_threads,
    )
    storage = photo_storage or LocalPrivatePhotoStorage(base_dir=cfg.photo_storage_dir)

    quality_svc = ImageQualityService(cfg)
    challenge_svc = ActiveChallengeService(cfg)
    buf_store = TransientSessionBufferStore(max_sampled_frames=cfg.passive_sample_frames_max)
    session_svc = CaptureSessionService(
        settings=cfg,
        challenge_service=challenge_svc,
        buffer_store=buf_store,
    )
    pipeline_svc = LivenessPipelineService(
        settings=cfg,
        session_service=session_svc,
        quality_service=quality_svc,
        challenge_service=challenge_svc,
        face_detector=detector,
        anti_spoofing=pad_provider,
        photo_storage=storage,
    )

    return ServiceContainer(
        settings=cfg,
        db_manager=db_mgr,
        face_detector=detector,
        anti_spoofing=pad_provider,
        photo_storage=storage,
        quality_service=quality_svc,
        challenge_service=challenge_svc,
        buffer_store=buf_store,
        session_service=session_svc,
        pipeline_service=pipeline_svc,
    )


def get_container() -> ServiceContainer:
    """Return active singleton ServiceContainer."""
    global _container
    with _container_lock:
        if _container is None:
            _container = build_service_container()
        return _container


def set_container(container: ServiceContainer | None) -> None:
    """Override or reset the global ServiceContainer (used in tests and lifespan)."""
    global _container
    with _container_lock:
        _container = container


def get_db_session(
    container: ServiceContainer = Depends(get_container),
) -> Generator[Session, None, None]:
    """Yield a transactional SQLAlchemy session."""
    yield from container.db_manager.get_session()


def get_session_repository(db: Session = Depends(get_db_session)) -> SessionRepository:
    """Provide a SessionRepository bound to the current request's DB session."""
    return SessionRepository(db)


class SlidingWindowRateLimiter:
    """Thread-safe per-client sliding-window rate limiter."""

    def __init__(self, max_requests_per_minute: int = 60) -> None:
        self.max_requests = max_requests_per_minute
        self.window_seconds = 60.0
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, client_key: str) -> None:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            q = self._hits[client_key]
            while q and q[0] <= cutoff:
                q.popleft()
            if len(q) >= self.max_requests:
                raise AppError(
                    code=ErrorCode.RATE_LIMIT_EXCEEDED,
                    message="Too many requests. Please wait before retrying.",
                    status_code=429,
                )
            q.append(now)


_rate_limiter: SlidingWindowRateLimiter | None = None


def enforce_rate_limit(
    request: Request,
    container: ServiceContainer = Depends(get_container),
) -> None:
    """FastAPI dependency enforcing per-IP sliding-window rate limits."""
    global _rate_limiter
    limit = container.settings.rate_limit_requests_per_minute
    if _rate_limiter is None or _rate_limiter.max_requests != limit:
        _rate_limiter = SlidingWindowRateLimiter(max_requests_per_minute=limit)

    client_host = request.client.host if request.client else "unknown"
    _rate_limiter.check(client_host)
