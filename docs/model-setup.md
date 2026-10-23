# Model Setup, Provenance, and Preprocessing

## 1. Upstream Provenance & Licensing

This project integrates genuine, pretrained **MiniFASNet** anti-spoofing models originally developed by Minivision AI alongside Google's **MediaPipe Face Landmarker**:

| Model File | Upstream Architecture | Upstream Repository | License | Expected SHA-256 |
| :--- | :--- | :--- | :--- | :--- |
| `MiniFASNetV2.onnx` | `2.7_80x80_MiniFASNetV2` | [`minivision-ai/Silent-Face-Anti-Spoofing`](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) (ONNX export via [`yakhyo/face-anti-spoofing`](https://github.com/yakhyo/face-anti-spoofing)) | Apache-2.0 | `b32929adc2d9c34b9486f8c4c7bc97c1b69bc0ea9befefc380e4faae4e463907` |
| `MiniFASNetV1SE.onnx` | `4_0_0_80x80_MiniFASNetV1SE` | [`minivision-ai/Silent-Face-Anti-Spoofing`](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) (ONNX export via [`yakhyo/face-anti-spoofing`](https://github.com/yakhyo/face-anti-spoofing)) | Apache-2.0 | `ebab7f90c7833fbccd46d3a555410e78d969db5438e169b6524be444862b3676` |
| `face_landmarker.task` | MediaPipe Face Landmarker v1 (float16) | [Google MediaPipe Vision Models](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) | Apache-2.0 | `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` |

## 2. Acquisition and Integrity Verification

From the `backend/` directory, run:

```powershell
# Windows PowerShell
.\.venv\Scripts\python.exe scripts/download_models.py
```

```bash
# Linux / macOS
./.venv/bin/python scripts/download_models.py
```

The script downloads all three assets into `<repo>/models/` and verifies each file against its pinned SHA-256 digest. On backend startup, `MediaPipeFaceDetector` and `MiniFASNetOnnxProvider` independently re-verify the SHA-256 digests (`LIVENESS_VERIFY_MODEL_CHECKSUM=true`) before loading the models into memory.

## 3. Verified Preprocessing Specification

Inspection of upstream `minivision-ai/Silent-Face-Anti-Spoofing` (`src/generate_patches.py`, `src/anti_spoof_predict.py`, and `src/data_io/functional.py`) established the following exact requirements implemented in `backend/app/vision/providers/minifasnet_onnx.py`:

1. **Multi-Scale Context Cropping:**
   - `MiniFASNetV2.onnx` was trained with bounding-box expansion scale **`2.7`** (`2.7_80x80_MiniFASNetV2`).
   - `MiniFASNetV1SE.onnx` was trained with bounding-box expansion scale **`4.0`** (`4_0_0_80x80_MiniFASNetV1SE`) to capture broader background/screen-border context around the face.
   - When expanding around the face center `(cx, cy)` by `effective_scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, scale)`, any out-of-bounds edge is shifted inward onto the opposite edge so the square aspect ratio is preserved before resizing to `80x80`.
2. **Color Space & Numerical Range:**
   - Input patches must remain in **BGR** channel order (`cv2` native format).
   - Upstream `src/data_io/functional.py` (`to_tensor`) converts `uint8` `[0, 255]` arrays to `float32` `[0.0, 255.0]` **without** dividing by `255.0`. Dividing by `255.0` invalidates the BatchNorm statistics of the pretrained weights.
   - Tensor layout is `NCHW` with shape `(1, 3, 80, 80)`.
3. **Output Class Mapping & Multi-Scale Fusion:**
   - Each model outputs raw logits of shape `(1, 3)`.
   - After applying softmax across axis `1`, probability vectors from `MiniFASNetV2` (scale `2.7`) and `MiniFASNetV1SE` (scale `4.0`) are averaged.
   - Class index **`1`** represents **Real (Genuine Live Face)**; class indices **`0`** and **`2`** represent **Presentation Attacks (Print / Replay Spoofs)**.

## 4. CPU Inference Requirements & Measured Performance

- **Default Execution Provider:** `CPUExecutionProvider` (no GPU or CUDA required). Optional GPU execution can be enabled by installing `onnxruntime-gpu` and setting `LIVENESS_ONNX_EXECUTION_PROVIDER=CUDAExecutionProvider`.
- **Measured Local Benchmark (Windows x86_64 CPU, 640x480 frames, 25 iterations via `scripts/benchmark_pipeline.py`):**
  - Model initialization time (all 3 models + warmup): **`88.56 ms`**
  - Python heap delta: **`6.03 MB`** current / **`11.08 MB`** peak
  - OpenCV Image Quality check: **`1.82 ms`** mean (`1.64 ms` median, `2.33 ms` p95)
  - MediaPipe FaceLandmarker (478 3D landmarks + transformation matrix): **`11.31 ms`** mean (`11.02 ms` median, `13.18 ms` p95)
  - `MiniFASNetV2.onnx` single-model inference: **`1.40 ms`** mean (`1.35 ms` median, `1.84 ms` p95)
  - `MiniFASNetV2.onnx` + `MiniFASNetV1SE.onnx` multi-scale ensemble inference: **`2.37 ms`** mean (`2.30 ms` median, `3.16 ms` p95)
