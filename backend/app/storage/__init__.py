"""Storage providers package."""

from app.storage.base import PhotoStorageProvider
from app.storage.local_storage import LocalPrivatePhotoStorage

__all__ = ["LocalPrivatePhotoStorage", "PhotoStorageProvider"]
