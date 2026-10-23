"""Computer-vision components for face detection, landmarks, and passive anti-spoofing."""

from app.vision.anti_spoofing import (
    AntiSpoofingProvider,
    BatchSpoofEvaluation,
    FrameSpoofScore,
)
from app.vision.face_detector import FaceDetectorProvider, MediaPipeFaceDetector
from app.vision.face_landmarks import (
    BoundingBox,
    FaceAnalysis,
    HeadPose,
    Point3D,
    analyze_face_landmarks,
    estimate_head_pose,
)
from app.vision.providers.minifasnet_onnx import MiniFASNetOnnxProvider

__all__ = [
    "AntiSpoofingProvider",
    "BatchSpoofEvaluation",
    "BoundingBox",
    "FaceAnalysis",
    "FaceDetectorProvider",
    "FrameSpoofScore",
    "HeadPose",
    "MediaPipeFaceDetector",
    "MiniFASNetOnnxProvider",
    "Point3D",
    "analyze_face_landmarks",
    "estimate_head_pose",
]
