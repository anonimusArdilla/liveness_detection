"""Privacy-preserving logging configuration with sensitive-field redaction."""

from __future__ import annotations

import logging
import re
import sys

REDACTED = "[REDACTED]"
SENSITIVE_PATTERNS = [
    re.compile(r'("nonce"\s*:\s*)"[^"]+"', re.IGNORECASE),
    re.compile(r'("session_nonce"\s*:\s*)"[^"]+"', re.IGNORECASE),
    re.compile(r'("photo_access_token"\s*:\s*)"[^"]+"', re.IGNORECASE),
    re.compile(r'("image_base64"\s*:\s*)"[^"]+"', re.IGNORECASE),
    re.compile(r"(data:image/[a-zA-Z0-9.+-]+;base64,)[A-Za-z0-9+/=]+"),
]


class PrivacyRedactingFilter(logging.Filter):
    """Filter that scrubs biometric base64 payloads, nonces, and tokens from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        redacted_msg = msg
        for pattern in SENSITIVE_PATTERNS:
            redacted_msg = pattern.sub(rf"\1{REDACTED}", redacted_msg)
        if redacted_msg != msg:
            record.msg = redacted_msg
            record.args = ()
        return True


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure application-wide structured logging with privacy redaction."""
    logger = logging.getLogger("liveness")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
        handler.setFormatter(formatter)
        handler.addFilter(PrivacyRedactingFilter())
        logger.addHandler(handler)
        logger.propagate = False

    return logger


def get_logger(name: str = "liveness") -> logging.Logger:
    """Retrieve a child logger under the privacy-filtered liveness namespace."""
    configure_logging()
    if name == "liveness" or name.startswith("liveness."):
        return logging.getLogger(name)
    return logging.getLogger(f"liveness.{name}")
