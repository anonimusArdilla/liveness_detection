"""Abstract storage interface for persisting and retrieving accepted photographs."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class PhotoStorageProvider(ABC):
    """Protocol for storing and retrieving validated capture photographs."""

    @abstractmethod
    def is_ready(self) -> bool:
        """Verify whether the underlying storage medium is writable and accessible."""

    @abstractmethod
    def save_validated_photo(self, session_id: str, bgr_image: np.ndarray) -> str:
        """Encode and persist an accepted BGR photograph, returning an opaque storage key."""

    @abstractmethod
    def load_validated_photo(self, storage_key: str) -> bytes:
        """Load the encoded JPEG bytes for a previously persisted storage key."""

    @abstractmethod
    def delete_photo(self, storage_key: str) -> None:
        """Delete a stored photograph if required by retention policies."""
