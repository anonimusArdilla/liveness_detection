# Local AI-Powered Facial Liveness Detection System

A modular, privacy-preserving facial liveness detection system combining **server-validated active 3D facial landmark challenges** (MediaPipe Face Landmarker) with **local CPU passive anti-spoofing** (MiniFASNetV2 + MiniFASNetV1SE multi-scale ONNX Runtime ensemble).

## 1. Purpose and Scope

- **Primary Objective:** Verify whether a live person appears to be physically present in front of a camera while detecting common presentation attacks (printed photographs and prerecorded video replayed on mobile/desktop screens).
- **Strict Scope Boundary:** This system implements **only** facial capture guidance, image-quality validation, active + passive liveness verification, and protected storage of the accepted portrait. It does **not** perform identity verification, document matching, or 1:1 / 1:N facial recognition.
- **Privacy First:** All video streams and intermediate challenge frames are processed strictly in memory and immediately released. Raw videos are never recorded or stored. Only the single accepted frontal photograph is persisted (stripped of EXIF metadata) after `VALIDATED` confirmation.

---

## 2. Architecture Overview

- **Backend (`backend/`):** Python 3.11, FastAPI, Pydantic v2, OpenCV, MediaPipe Tasks (`FaceLandmarker`), ONNX Runtime (`MiniFASNetV2` + `MiniFASNetV1SE`), SQLAlchemy 2.0, and Alembic (SQLite by default, PostgreSQL-compatible).
- **Frontend (`frontend/`):** React 18, TypeScript (strict), and Vite.
- **Documentation (`docs/`):**
  - [Architecture Overview](docs/architecture.md)
  - [REST API Contract (`/api/v1`)](docs/api.md)
  - [Security & Privacy Controls](docs/security-and-privacy.md)
  - [Model Setup, Provenance & Preprocessing](docs/model-setup.md)
  - [Evaluation Guide & Known Limitations](docs/evaluation.md)

---

## 3. Prerequisites and Tested Versions

| Component | Tested Version | Notes |
| :--- | :--- | :--- |
| **Python** | `3.11.16` (`>=3.11,<3.13`) | Managed via `uv` or standard `venv` |
| **uv** (optional) | `0.12.19` | Fast Python environment and package manager |
| **Node.js / npm** | `v24.21.0` / `11.19.0` | Compatible with Node.js `>=20.x` |
| **ONNX Runtime** | `1.31.0` | Runs on `CPUExecutionProvider` by default |
| **MediaPipe** | `0.10.35` | Uses `mediapipe.tasks.python.vision.FaceLandmarker` |
| **OpenCV** | `4.11.0.86` | Image decoding, brightness, and Laplacian sharpness |

---

## 4. Quickstart: Model Acquisition, Backend & Frontend Startup

### Step 1: Clone / Enter Repository and Configure Environment Files

**Windows PowerShell:**
```powershell
Copy-Item backend\.env.example backend\.env
Copy-Item frontend\.env.example frontend\.env
```

**Linux / macOS:**
```bash
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
```

### Step 2: Set Up Backend Python Virtual Environment

#### Option A: Using `uv` (Recommended)

**Windows PowerShell:**
```powershell
Set-Location backend
uv sync --python 3.11
```

**Linux / macOS:**
```bash
cd backend
uv sync --python 3.11
```

#### Option B: Using Standard `venv` + `pip`

**Windows PowerShell:**
```powershell
Set-Location backend
python -m venv .venv
.\.venv\Scripts\pip install --upgrade pip
.\.venv\Scripts\pip install -r requirements.txt
```

**Linux / macOS:**
```bash
cd backend
python3.11 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt
```

### Step 3: Download and Verify Pretrained Model Weights

Model weights are not committed to Git. Run the download script to fetch `MiniFASNetV2.onnx`, `MiniFASNetV1SE.onnx`, and `face_landmarker.task` (all **Apache-2.0** licensed) into `models/` and verify their pinned SHA-256 checksums:

**Windows PowerShell (from `backend/`):**
```powershell
.\.venv\Scripts\python.exe scripts/download_models.py
```

**Linux / macOS (from `backend/`):**
```bash
./.venv/bin/python scripts/download_models.py
```

### Step 4: Initialize Database and Run Migrations

**Windows PowerShell (from `backend/`):**
```powershell
.\.venv\Scripts\alembic.exe upgrade head
```

**Linux / macOS (from `backend/`):**
```bash
./.venv/bin/alembic upgrade head
```

### Step 5: Start the Backend Server

**Windows PowerShell (from `backend/`):**
```powershell
.\.venv\Scripts\uvicorn.exe app.main:app --host 127.0.0.1 --port 8000 --reload
```

**Linux / macOS (from `backend/`):**
```bash
./.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Verify readiness at `http://127.0.0.1:8000/api/v1/health/ready`.

### Step 6: Install and Start the React Frontend

Open a second terminal at the repository root:

**Windows PowerShell / Linux / macOS:**
```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173` (or `http://localhost:5173`) in your browser.

---

## 5. Docker Compose Startup

To launch both services in containers (automatically downloading and verifying model weights on startup):

```bash
docker compose up --build
```

- Frontend UI: `http://localhost:5173`
- Backend API & OpenAPI Docs: `http://localhost:8000/docs`

---

## 6. Environment Configuration Reference

### Backend (`backend/.env` — prefixed with `LIVENESS_`)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `LIVENESS_HOST` | `127.0.0.1` | Backend bind address |
| `LIVENESS_PORT` | `8000` | Backend port |
| `LIVENESS_LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `LIVENESS_CORS_ORIGINS` | `["http://localhost:5173","http://127.0.0.1:5173"]` | Explicit allowed CORS origins |
| `LIVENESS_DATABASE_URL` | `sqlite:///./data/liveness.db` | SQLAlchemy DB URL (SQLite or PostgreSQL `postgresql+psycopg://...`) |
| `LIVENESS_PHOTO_STORAGE_DIR` | `./storage_data/accepted_photos` | Private directory for accepted portraits |
| `LIVENESS_FACE_LANDMARKER_MODEL_PATH` | `../models/face_landmarker.task` | MediaPipe FaceLandmarker model bundle path |
| `LIVENESS_PRIMARY_MODEL_PATH` | `../models/MiniFASNetV2.onnx` | Primary passive anti-spoofing ONNX model |
| `LIVENESS_PRIMARY_MODEL_SCALE` | `2.7` | Bounding box expansion scale for `MiniFASNetV2` |
| `LIVENESS_SECONDARY_MODEL_PATH` | `../models/MiniFASNetV1SE.onnx` | Secondary passive anti-spoofing ONNX model |
| `LIVENESS_SECONDARY_MODEL_SCALE` | `4.0` | Bounding box expansion scale for `MiniFASNetV1SE` |
| `LIVENESS_VERIFY_MODEL_CHECKSUM` | `true` | Enforce SHA-256 verification of model files on startup |
| `LIVENESS_ONNX_EXECUTION_PROVIDER` | `CPUExecutionProvider` | ONNX Runtime execution provider |
| `LIVENESS_ONNX_INTRA_OP_THREADS` | `2` | CPU intra-op threads for ONNX Runtime |
| `LIVENESS_MIN_IMAGE_WIDTH` / `HEIGHT` | `320` / `240` | Minimum allowed frame dimensions |
| `LIVENESS_MIN_BRIGHTNESS` / `MAX` | `45.0` / `225.0` | Mean grayscale brightness bounds (`0..255`) |
| `LIVENESS_MIN_BLUR_LAPLACIAN_VAR` | `35.0` | Minimum Laplacian variance sharpness score |
| `LIVENESS_MIN_FACE_AREA_RATIO` / `MAX` | `0.04` / `0.75` | Allowed face bounding-box area ratio |
| `LIVENESS_MAX_CENTER_OFFSET_RATIO` | `0.28` | Maximum normalized face offset from frame center |
| `LIVENESS_SESSION_TTL_SECONDS` | `180` | Session expiration lifetime in seconds |
| `LIVENESS_CHALLENGE_COUNT` | `3` | Number of active challenge steps per session |
| `LIVENESS_CENTER_YAW_MAX_DEG` | `12.0` | Max absolute yaw angle (deg) for `CENTER_FACE` |
| `LIVENESS_TURN_LEFT_MIN_YAW_DEG` | `-13.0` | Required negative yaw angle (deg) for `TURN_LEFT` |
| `LIVENESS_TURN_RIGHT_MIN_YAW_DEG` | `13.0` | Required positive yaw angle (deg) for `TURN_RIGHT` |
| `LIVENESS_SPOOF_REAL_SCORE_THRESHOLD` | `0.65` | Minimum mean `Real` probability across sampled frames |
| `LIVENESS_SPOOF_MIN_FRAME_SCORE` | `0.45` | Minimum single-frame `Real` probability |
| `LIVENESS_MAX_REQUEST_BYTES` | `6291456` | Maximum HTTP request payload size (6 MiB) |
| `LIVENESS_RATE_LIMIT_REQUESTS_PER_MINUTE` | `60` | Per-IP sliding-window rate limit |

---

## 7. Running All Tests, Linters, and Benchmarks

### Backend Linting, Type Checking, and Automated Tests

**Windows PowerShell:**
```powershell
Set-Location backend
.\.venv\Scripts\ruff.exe check app tests scripts
.\.venv\Scripts\mypy.exe app tests scripts
.\.venv\Scripts\pytest.exe -v
```

**Linux / macOS:**
```bash
cd backend
./.venv/bin/ruff check app tests scripts
./.venv/bin/mypy app tests scripts
./.venv/bin/pytest -v
```

### CPU Latency & Memory Benchmark

```powershell
Set-Location backend
.\.venv\Scripts\python.exe scripts/benchmark_pipeline.py
```

### Frontend Linting, Type Checking, Unit Tests, and Build

```bash
cd frontend
npm run lint
npm run typecheck
npm test
npm run build
```

---

## 8. Example Capture Lifecycle Flow

1. **Check Readiness:** `GET /api/v1/health/ready` returns `200 OK` (`"ready": true`).
2. **Create Session:** `POST /api/v1/sessions` with `{"registration_ref": "REG-2026-001"}` returns `session_id`, `session_nonce`, and step `0` (`CENTER_FACE`).
3. **Submit Active Challenge Frames:** For each step (`CENTER_FACE` -> randomized `TURN_LEFT` / `TURN_RIGHT`), the client samples 3 frames at `180 ms` intervals and sends `POST /api/v1/sessions/{session_id}/challenge/frames` with monotonically increasing `timestamp_ms` values.
4. **Finalize & Run Passive Anti-Spoofing:** Once `all_challenges_completed == true`, the client calls `POST /api/v1/sessions/{session_id}/complete` with `session_nonce`.
5. **Decision & Persistence:** The backend runs multi-scale `MiniFASNetV2` + `MiniFASNetV1SE` ONNX inference across the session's sampled frames. Only if `outcome == "VALIDATED"` does it persist the clean JPEG portrait and return a `photo_access_token` and `verification_signature`.

---

## 9. Troubleshooting

- **`503 SERVICE_UNAVAILABLE` on `/api/v1/health/ready` or Session Creation:**
  - One or more model files (`MiniFASNetV2.onnx`, `MiniFASNetV1SE.onnx`, `face_landmarker.task`) are missing from `models/` or failed SHA-256 verification. Run `python scripts/download_models.py --force`.
- **Browser Camera Permission Denied (`NotAllowedError`):**
  - Ensure you access the frontend via `http://localhost:5173` or `http://127.0.0.1:5173` (browsers block `navigator.mediaDevices.getUserMedia` on insecure non-localhost origins). Reset camera permissions in the browser's site settings and click **Retry Camera & Start Session**.
- **Camera Unavailable (`NotReadableError`):**
  - Another application (Zoom, Teams, OBS) has exclusive lock on the webcam. Close the other application and click Retry.
- **`TURN_LEFT` / `TURN_RIGHT` Challenge Not Passing:**
  - Keep your face visible inside the frame while turning your head slowly (~15–25 degrees) in the requested direction under even front lighting.

---

## 10. Security Considerations and Known Limitations

- This application is an engineering baseline and is **not** certified under ISO/IEC 30107-3.
- Passive anti-spoofing thresholds (`LIVENESS_SPOOF_REAL_SCORE_THRESHOLD`) must be calibrated using representative genuine and spoof samples from your target hardware (`scripts/evaluate_models.py`).
- Standard monocular RGB cameras cannot guarantee defense against high-grade 3D masks or driver/browser-level virtual camera injection without hardware depth sensors and device attestation. See [`docs/evaluation.md`](docs/evaluation.md) for details.
