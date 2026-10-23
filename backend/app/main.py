"""FastAPI application entrypoint for the Local AI-Powered Facial Liveness Detection System."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
from starlette.types import ASGIApp

from app import __version__
from app.api.dependencies import get_container
from app.api.routes import health_router, sessions_router
from app.core.config import Settings, get_settings
from app.core.errors import ErrorCode, register_exception_handlers
from app.core.logging import configure_logging


class SecurityAndPayloadLimitMiddleware(BaseHTTPMiddleware):
    """Enforces maximum request body size in memory and adds defensive HTTP security headers."""

    def __init__(self, app: ASGIApp, max_request_bytes: int) -> None:
        super().__init__(app)
        self.max_request_bytes = max_request_bytes

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_request_bytes:
                    return JSONResponse(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        content={
                            "error": {
                                "code": ErrorCode.PAYLOAD_TOO_LARGE.value,
                                "message": (
                                    f"Request body exceeds maximum allowed size of "
                                    f"{self.max_request_bytes} bytes."
                                ),
                                "details": {"max_request_bytes": self.max_request_bytes},
                            }
                        },
                    )
            except ValueError:
                pass

        body = await request.body()
        if len(body) > self.max_request_bytes:
            return JSONResponse(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                content={
                    "error": {
                        "code": ErrorCode.PAYLOAD_TOO_LARGE.value,
                        "message": (
                            f"Request body exceeds maximum allowed size of {self.max_request_bytes} bytes."
                        ),
                        "details": {"max_request_bytes": self.max_request_bytes},
                    }
                },
            )

        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize logging, database schema, and computer-vision models on startup."""
    settings = get_settings()
    logger = configure_logging(settings.log_level)
    container = get_container()
    logger.info(
        "Liveness detection service starting (version=%s, ready=%s)",
        __version__,
        container.pipeline_service.is_ready(),
    )
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    cfg = settings or get_settings()
    configure_logging(cfg.log_level)

    app = FastAPI(
        title="Facial Liveness Detection Service",
        description=(
            "Local AI-powered facial liveness detection combining MediaPipe 3D landmark "
            "active challenges with ONNX Runtime MiniFASNet passive anti-spoofing."
        ),
        version=__version__,
        lifespan=lifespan,
    )

    app.add_middleware(
        SecurityAndPayloadLimitMiddleware,
        max_request_bytes=cfg.max_request_bytes,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Photo-Token"],
    )

    register_exception_handlers(app)

    api_v1 = APIRouter(prefix="/api/v1")
    api_v1.include_router(health_router)
    api_v1.include_router(sessions_router)
    app.include_router(api_v1)

    return app


app = create_app()
