"""Capture-session and liveness validation routes (/api/v1/sessions)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, Query, status
from fastapi.responses import Response

from app.api.dependencies import (
    ServiceContainer,
    enforce_rate_limit,
    get_container,
    get_session_repository,
)
from app.core.errors import AppError, ErrorCode
from app.repositories.session_repository import SessionRepository
from app.schemas.liveness import (
    CompleteSessionRequest,
    CreateSessionRequest,
    CreateSessionResponse,
    CurrentChallengeResponse,
    SessionResultResponse,
    SubmitChallengeFramesRequest,
    SubmitChallengeFramesResponse,
)

router = APIRouter(
    prefix="/sessions",
    tags=["Capture Sessions"],
    dependencies=[Depends(enforce_rate_limit)],
)


@router.post(
    "",
    response_model=CreateSessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new short-lived facial liveness capture session",
)
def create_capture_session(
    payload: CreateSessionRequest | None = None,
    container: ServiceContainer = Depends(get_container),
    repo: SessionRepository = Depends(get_session_repository),
) -> CreateSessionResponse:
    """Create a unique session, one-time nonce, expiration timestamp, and randomized challenge sequence."""
    if not container.pipeline_service.is_ready():
        raise AppError(
            code=ErrorCode.SERVICE_UNAVAILABLE,
            message="Liveness validation service is not ready: required models or storage are unavailable.",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    reg_ref = payload.registration_ref if payload else None
    return container.session_service.create_session(repo=repo, registration_ref=reg_ref)


@router.get(
    "/{session_id}/challenge",
    response_model=CurrentChallengeResponse,
    summary="Retrieve the current active challenge instruction for a session",
)
def get_current_challenge(
    session_id: str,
    container: ServiceContainer = Depends(get_container),
    repo: SessionRepository = Depends(get_session_repository),
) -> CurrentChallengeResponse:
    """Return the active challenge step without revealing subsequent uncompleted challenges."""
    return container.session_service.get_current_challenge(repo=repo, session_id=session_id)


@router.post(
    "/{session_id}/challenge/frames",
    response_model=SubmitChallengeFramesResponse,
    summary="Submit a bounded batch of frames for the current active challenge",
)
def submit_challenge_frames(
    session_id: str,
    payload: SubmitChallengeFramesRequest,
    container: ServiceContainer = Depends(get_container),
    repo: SessionRepository = Depends(get_session_repository),
) -> SubmitChallengeFramesResponse:
    """Evaluate image quality and 3D facial landmark movement for the current challenge."""
    return container.pipeline_service.submit_challenge_frames(
        repo=repo,
        session_id=session_id,
        request=payload,
    )


@router.post(
    "/{session_id}/complete",
    response_model=SessionResultResponse,
    summary="Complete the capture session and run passive ONNX anti-spoofing verification",
)
def complete_capture_session(
    session_id: str,
    payload: CompleteSessionRequest,
    container: ServiceContainer = Depends(get_container),
    repo: SessionRepository = Depends(get_session_repository),
) -> SessionResultResponse:
    """Finalize the session, run passive MiniFASNet ONNX inference, and persist accepted photo if valid."""
    return container.pipeline_service.complete_session(
        repo=repo,
        session_id=session_id,
        request=payload,
    )


@router.get(
    "/{session_id}",
    response_model=SessionResultResponse,
    summary="Retrieve the status and validation outcome of a capture session",
)
def get_session_outcome(
    session_id: str,
    container: ServiceContainer = Depends(get_container),
    repo: SessionRepository = Depends(get_session_repository),
) -> SessionResultResponse:
    """Return the current or final decision and verification metadata for a session."""
    return container.session_service.build_session_result(repo=repo, session_id=session_id)


@router.get(
    "/{session_id}/photo",
    summary="Retrieve the accepted photograph for a VALIDATED session using its access token",
)
def get_validated_photograph(
    session_id: str,
    x_photo_token: str | None = Header(default=None, alias="X-Photo-Token"),
    token: str | None = Query(default=None, description="Optional query fallback for X-Photo-Token"),
    container: ServiceContainer = Depends(get_container),
    repo: SessionRepository = Depends(get_session_repository),
) -> Response:
    """Protected endpoint serving the accepted JPEG photograph only when a valid token is provided."""
    resolved_token = x_photo_token or token
    if not resolved_token:
        raise AppError(
            code=ErrorCode.UNAUTHORIZED_ACCESS,
            message="Missing X-Photo-Token header or token parameter.",
            status_code=status.HTTP_403_FORBIDDEN,
        )

    jpeg_bytes = container.pipeline_service.get_validated_photo_bytes(
        repo=repo,
        session_id=session_id,
        access_token=resolved_token,
    )
    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, private",
            "Pragma": "no-cache",
            "X-Content-Type-Options": "nosniff",
        },
    )
