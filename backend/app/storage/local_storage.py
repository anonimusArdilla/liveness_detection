"""Local private filesystem storage provider for accepted capture photographs."""

from __future__ import annotations

import hashlib
import re
import secrets
from pathlib import Path

import cv2
import numpy as np

from app.core.errors import AppError, ErrorCode
from app.storage.base import PhotoStorageProvider

SAFE_KEY_PATTERN = re.compile(r"^[a-f0-9]{40}\.jpg$")


class LocalPrivatePhotoStorage(PhotoStorageProvider):
    """Stores accepted photographs in a non-public server directory with opaque keys.

    Security & privacy controls:
    - Never uses client-supplied filenames.
    - Generates an opaque 40-hex-character key bound to a random salt and session_id.
    - Strictly validates storage keys before reading or deleting to prevent path traversal.
    - Encodes fresh JPEG bytes from the validated numpy array to strip any input EXIF metadata.
    """

    def __init__(self, base_dir: Path, jpeg_quality: int = 92) -> None:
        self.base_dir = base_dir.resolve()
        self.jpeg_quality = max(70, min(100, jpeg_quality))
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def is_ready(self) -> bool:
        """Verify that the private storage directory exists and is writable."""
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            probe = self.base_dir / ".readiness_probe"
            probe.write_bytes(b"ok")
            probe.unlink(missing_ok=True)
            return True
        except Exception:
            return False

    def save_validated_photo(self, session_id: str, bgr_image: np.ndarray) -> str:
        """Re-encode the accepted frame as clean JPEG (stripping metadata) and persist."""
        salt = secrets.token_hex(16)
        digest = hashlib.sha256(f"{session_id}:{salt}".encode()).hexdigest()[:40]
        storage_key = f"{digest}.jpg"

        ok, encoded = cv2.imencode(
            ".jpg",
            bgr_image,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            raise AppError(
                code=ErrorCode.INTERNAL_ERROR,
                message="Failed to encode accepted photograph.",
                status_code=500,
            )

        target_path = self._resolve_safe_path(storage_key)
        try:
            target_path.write_bytes(encoded.tobytes())
        finally:
            del encoded
        return storage_key

    def load_validated_photo(self, storage_key: str) -> bytes:
        """Read JPEG bytes for a validated storage key."""
        target_path = self._resolve_safe_path(storage_key)
        if not target_path.is_file():
            raise AppError(
                code=ErrorCode.PHOTO_NOT_FOUND,
                message="Accepted photograph was not found.",
                status_code=404,
            )
        return target_path.read_bytes()

    def delete_photo(self, storage_key: str) -> None:
        """Remove a stored photograph if it exists."""
        target_path = self._resolve_safe_path(storage_key)
        target_path.unlink(missing_ok=True)

    def _resolve_safe_path(self, storage_key: str) -> Path:
        if not SAFE_KEY_PATTERN.match(storage_key):
            raise AppError(
                code=ErrorCode.PHOTO_NOT_FOUND,
                message="Invalid storage key format.",
                status_code=404,
            )
        candidate = (self.base_dir / storage_key).resolve()
        if candidate.parent != self.base_dir:
            raise AppError(
                code=ErrorCode.PHOTO_NOT_FOUND,
                message="Invalid storage key resolution.",
                status_code=404,
            )
        return candidate
