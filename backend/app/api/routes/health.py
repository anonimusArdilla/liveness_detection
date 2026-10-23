"""Health and readiness endpoints (/api/v1/health)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status

from app import __version__
from app.api.dependencies import ServiceContainer, get_container
from app.schemas.liveness import HealthResponse, ModelStatusSchema

router = APIRouter(prefix="/health", tags=["Health & Readiness"])


def _collect_statuses(container: ServiceContainer) -> tuple[bool, bool, bool, list[ModelStatusSchema]]:
    db_ok = container.db_manager.check_health()
    storage_ok = container.photo_storage.is_ready()
    models = [
        container.face_detector.status(),
        *container.anti_spoofing.model_statuses(),
    ]
    all_models_ready = all(m.ready for m in models) and container.anti_spoofing.is_ready()
    overall_ready = db_ok and storage_ok and all_models_ready
    return overall_ready, db_ok, storage_ok, models


@router.get(
    "",
    response_model=HealthResponse,
    summary="Liveness probe and overall component status",
)
def health_check(
    container: ServiceContainer = Depends(get_container),
) -> HealthResponse:
    """Report process liveness and detailed readiness of models, database, and storage."""
    overall_ready, db_ok, storage_ok, models = _collect_statuses(container)
    return HealthResponse(
        status="ok",
        ready=overall_ready,
        version=__version__,
        models=models,
        database_ready=db_ok,
        storage_ready=storage_ok,
    )


@router.get(
    "/ready",
    response_model=HealthResponse,
    summary="Readiness probe verifying all models, checksums, DB, and storage are operational",
)
def readiness_check(
    response: Response,
    container: ServiceContainer = Depends(get_container),
) -> HealthResponse:
    """Return HTTP 200 when ready to process validations, or HTTP 503 when any model/asset is missing."""
    overall_ready, db_ok, storage_ok, models = _collect_statuses(container)
    if not overall_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="ready" if overall_ready else "not_ready",
        ready=overall_ready,
        version=__version__,
        models=models,
        database_ready=db_ok,
        storage_ready=storage_ok,
    )
