"""Offline evaluation utility for computing FAR (APCER) and FRR (BPCER) on labeled sample directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from app.core.config import get_settings
from app.vision.face_detector import MediaPipeFaceDetector
from app.vision.providers.minifasnet_onnx import MiniFASNetOnnxProvider

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def evaluate_dataset(
    genuine_dir: Path,
    spoof_dir: Path,
    threshold: float,
) -> dict[str, object]:
    """Evaluate genuine and spoof image folders and report FAR, FRR, and ACER metrics."""
    settings = get_settings()
    detector = MediaPipeFaceDetector(
        model_path=settings.face_landmarker_model_path,
        expected_sha256=settings.face_landmarker_sha256,
        verify_checksum=settings.verify_model_checksum,
    )
    pad = MiniFASNetOnnxProvider(
        primary_model_path=settings.primary_model_path,
        primary_scale=settings.primary_model_scale,
        primary_sha256=settings.primary_model_sha256,
        secondary_model_path=settings.secondary_model_path,
        secondary_scale=settings.secondary_model_scale,
        secondary_sha256=settings.secondary_model_sha256,
        verify_checksum=settings.verify_model_checksum,
    )

    genuine_scores: list[float] = []
    spoof_scores: list[float] = []
    skipped_no_face = 0

    for _label, folder, bucket in [
        ("genuine", genuine_dir, genuine_scores),
        ("spoof", spoof_dir, spoof_scores),
    ]:
        if not folder.is_dir():
            continue
        for img_path in sorted(folder.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            faces = detector.detect_faces(bgr)
            if len(faces) != 1:
                skipped_no_face += 1
                continue
            pred = pad.predict_frame(bgr, faces[0].bounding_box)
            bucket.append(pred.real_probability)

    detector.close()

    total_genuine = len(genuine_scores)
    total_spoof = len(spoof_scores)
    false_rejections = sum(1 for s in genuine_scores if s < threshold)
    false_acceptances = sum(1 for s in spoof_scores if s >= threshold)

    frr = round(false_rejections / total_genuine, 4) if total_genuine > 0 else None
    far = round(false_acceptances / total_spoof, 4) if total_spoof > 0 else None
    acer = round((frr + far) / 2.0, 4) if (frr is not None and far is not None) else None

    return {
        "threshold": threshold,
        "model": pad.active_model_label,
        "genuine_count": total_genuine,
        "spoof_count": total_spoof,
        "skipped_invalid_face_count": skipped_no_face,
        "false_rejections": false_rejections,
        "false_acceptances": false_acceptances,
        "FRR_BPCER": frr,
        "FAR_APCER": far,
        "ACER": acer,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate passive anti-spoofing FAR and FRR on labeled genuine/spoof directories."
    )
    parser.add_argument(
        "--genuine-dir", type=Path, required=True, help="Directory of genuine live face images"
    )
    parser.add_argument(
        "--spoof-dir", type=Path, required=True, help="Directory of presentation attack images"
    )
    parser.add_argument("--threshold", type=float, default=0.65, help="Real probability threshold")
    args = parser.parse_args()

    summary = evaluate_dataset(args.genuine_dir, args.spoof_dir, args.threshold)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
