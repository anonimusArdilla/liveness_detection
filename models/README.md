# Model Weights Directory

Model weights are excluded from version control (`.gitignore`) to keep the repository lightweight and ensure explicit provenance and integrity verification.

## Required Models

| File | Architecture / Role | Upstream Source | License | Expected SHA-256 |
| :--- | :--- | :--- | :--- | :--- |
| `MiniFASNetV2.onnx` | `2.7_80x80_MiniFASNetV2` (Primary passive anti-spoofing, scale `2.7`, `1x3x80x80` BGR `float32`) | [`minivision-ai/Silent-Face-Anti-Spoofing`](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) via [`yakhyo/face-anti-spoofing`](https://github.com/yakhyo/face-anti-spoofing) | Apache-2.0 | `b32929adc2d9c34b9486f8c4c7bc97c1b69bc0ea9befefc380e4faae4e463907` |
| `MiniFASNetV1SE.onnx` | `4_0_0_80x80_MiniFASNetV1SE` (Secondary multi-scale fusion passive anti-spoofing, scale `4.0`, `1x3x80x80` BGR `float32`) | [`minivision-ai/Silent-Face-Anti-Spoofing`](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) via [`yakhyo/face-anti-spoofing`](https://github.com/yakhyo/face-anti-spoofing) | Apache-2.0 | `ebab7f90c7833fbccd46d3a555410e78d969db5438e169b6524be444862b3676` |
| `face_landmarker.task` | Google MediaPipe Face Landmarker (478 3D facial landmarks + 4x4 facial transformation matrix) | [Google MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) | Apache-2.0 | `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` |

## Automated Download & SHA-256 Verification

From the `backend/` directory, run:

```powershell
uv run python scripts/download_models.py
```

If any required model file is missing or fails SHA-256 verification on startup, `/api/v1/health/ready` reports `"ready": false` and the liveness pipeline fails closed (`VALIDATION_UNAVAILABLE`).
