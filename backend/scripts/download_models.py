"""Script to download and verify SHA-256 checksums of required vision and anti-spoofing models."""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = REPO_ROOT / "models"


@dataclass(frozen=True)
class ModelAsset:
    """Metadata and integrity hash for an upstream model file."""

    filename: str
    url: str
    expected_sha256: str
    license_spdx: str
    source_repo: str
    description: str


MODEL_ASSETS: tuple[ModelAsset, ...] = (
    ModelAsset(
        filename="MiniFASNetV2.onnx",
        url="https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx",
        expected_sha256="b32929adc2d9c34b9486f8c4c7bc97c1b69bc0ea9befefc380e4faae4e463907",
        license_spdx="Apache-2.0",
        source_repo="https://github.com/minivision-ai/Silent-Face-Anti-Spoofing (ONNX export: https://github.com/yakhyo/face-anti-spoofing)",
        description="Primary passive anti-spoofing model (2.7_80x80_MiniFASNetV2, input 1x3x80x80 BGR float32)",
    ),
    ModelAsset(
        filename="MiniFASNetV1SE.onnx",
        url="https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV1SE.onnx",
        expected_sha256="ebab7f90c7833fbccd46d3a555410e78d969db5438e169b6524be444862b3676",
        license_spdx="Apache-2.0",
        source_repo="https://github.com/minivision-ai/Silent-Face-Anti-Spoofing (ONNX export: https://github.com/yakhyo/face-anti-spoofing)",
        description="Secondary passive anti-spoofing model (4_0_0_80x80_MiniFASNetV1SE, input 1x3x80x80 BGR float32)",
    ),
    ModelAsset(
        filename="face_landmarker.task",
        url="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        expected_sha256="64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
        license_spdx="Apache-2.0",
        source_repo="https://developers.google.com/mediapipe/solutions/vision/face_landmarker",
        description="Google MediaPipe Face Landmarker bundle (478 3D landmarks + 4x4 facial transformation matrix)",
    ),
)


def compute_sha256(path: Path) -> str:
    """Compute hex SHA-256 digest of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def ensure_models(models_dir: Path, force: bool = False) -> bool:
    """Download missing model assets and verify all SHA-256 digests."""
    models_dir.mkdir(parents=True, exist_ok=True)
    all_ok = True

    for asset in MODEL_ASSETS:
        target = models_dir / asset.filename
        if target.exists() and not force:
            digest = compute_sha256(target)
            if digest.lower() == asset.expected_sha256.lower():
                print(f"[OK] {asset.filename} already present and SHA-256 verified ({digest[:12]}...)")
                continue
            print(
                f"[WARN] {asset.filename} checksum mismatch ({digest} != {asset.expected_sha256}); re-downloading..."
            )

        print(f"[DOWNLOAD] Fetching {asset.filename} from {asset.url} ...")
        try:
            req = urllib.request.Request(asset.url, headers={"User-Agent": "liveness-detection-setup/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            digest = hashlib.sha256(data).hexdigest()
            if digest.lower() != asset.expected_sha256.lower():
                print(
                    f"[ERROR] Downloaded {asset.filename} failed SHA-256 check! "
                    f"Expected {asset.expected_sha256}, got {digest}",
                    file=sys.stderr,
                )
                all_ok = False
                continue
            target.write_bytes(data)
            print(f"[OK] Saved {asset.filename} ({len(data):,} bytes, sha256={digest})")
        except Exception as exc:
            print(f"[ERROR] Failed to download {asset.filename}: {exc}", file=sys.stderr)
            all_ok = False

    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and verify facial liveness model weights.")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Target directory for model weights (default: <repo>/models)",
    )
    parser.add_argument("--force", action="store_true", help="Force re-download even if present")
    args = parser.parse_args()

    ok = ensure_models(args.models_dir.resolve(), force=args.force)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
