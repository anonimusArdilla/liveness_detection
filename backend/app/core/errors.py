"""Structured application exceptions and FastAPI error handlers."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class ErrorCode(StrEnum):
    """Stable machine-readable error codes returned by the API."""

    VALIDATION_ERROR = "VALIDATION_ERROR"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    SESSION_EXPIRED = "SESSION_EXPIRED"
    SESSION_INVALID_STATE = "SESSION_INVALID_STATE"
    SESSION_NONCE_MISMATCH = "SESSION_NONCE_MISMATCH"
    CHALLENGE_MISMATCH = "CHALLENGE_MISMATCH"
    FRAME_ORDER_INVALID = "FRAME_ORDER_INVALID"
    IMAGE_DECODE_FAILED = "IMAGE_DECODE_FAILED"
    PHOTO_NOT_FOUND = "PHOTO_NOT_FOUND"
    UNAUTHORIZED_ACCESS = "UNAUTHORIZED_ACCESS"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AppError(Exception):
    """Base application exception with a structured error code and HTTP status."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


def register_exception_handlers(app: FastAPI) -> None:
    """Attach structured exception handlers that scrub sensitive payload data."""

    @app.exception_handler(AppError)
    async def handle_app_error(_request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code.value,
                    "message": exc.message,
                    "details": exc.details,
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        # Scrub raw input values (such as base64 frames or nonces) from validation errors
        safe_errors = [
            {
                "loc": [str(part) for part in err.get("loc", ())],
                "msg": err.get("msg", "Invalid field"),
                "type": err.get("type", "value_error"),
            }
            for err in exc.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "code": ErrorCode.VALIDATION_ERROR.value,
                    "message": "Request validation failed.",
                    "details": {"fields": safe_errors},
                }
            },
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_request: Request, _exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": ErrorCode.INTERNAL_ERROR.value,
                    "message": "An unexpected internal error occurred. Validation failed closed.",
                    "details": {},
                }
            },
        )
