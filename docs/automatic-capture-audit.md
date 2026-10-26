# Facial Liveness Automatic Capture Workflow — Architecture & System Audit

**Date:** 2026-10-26  
**Auditor:** Senior Full-Stack & Computer-Vision Security Architect  
**Document Status:** Complete & Verified against Source Repository  

---

## 1. Executive Summary

This audit assesses the facial liveness detection application to transition from a manual capture workflow (where users must click a button to capture and submit frames for each challenge) to an **automated, guided, hybrid capture workflow**.

The application currently enforces active challenge verification (head turns, smiles, blinks, eyebrow raises) alongside passive presentation attack detection (PAD via MiniFASNet ONNX). This audit analyzes the current implementation, identifies existing components and constraints, and establishes the foundation for the hybrid architecture implementation.

---

## 2. Repository & Runtime Inspection

### 2.1 Technology Stack & Versions
- **Backend:**
  - Runtime: Python 3.11 / 3.12 (managed via `uv` / `pyproject.toml`).
  - Framework: FastAPI 0.115 + Pydantic v2.9 + Pydantic-Settings.
  - Database & ORM: SQLAlchemy 2.0 (SQLite local / PostgreSQL production) + Alembic migrations (`backend/migrations/`).
  - Vision & Machine Learning:
    - Google MediaPipe `0.10.35` (FaceLandmarker Task API using `models/face_landmarker.task` with 478 3D landmarks, blendshapes, transformation matrices).
    - ONNX Runtime `1.19.2` (MiniFASNetV2 and MiniFASNetV1SE models in `models/`).
    - OpenCV Contrib `4.10.0.84`.
- **Frontend:**
  - Framework: React 18.3.1 + TypeScript 5.6.3 + Vite 5.4.10.
  - Testing: Vitest 2.1.4 + Testing Library (`@testing-library/react` 16.0.1, `jsdom` 25.0.1).
  - Styling: Vanilla CSS (`frontend/src/styles/index.css`) with responsive design system and SVG overlays.
  - Dependencies: Zero external ML libraries in browser (strictly vanilla React and DOM Canvas API).

### 2.2 Entry Points & Configuration
- Backend Entry: `backend/app/main.py` (`create_app` factory with CORS, security headers, rate limiting, and exception handlers).
- Frontend Entry: `frontend/src/main.tsx` mounting `frontend/src/app/App.tsx`.
- Central Configuration: `backend/app/core/config.py` (`Settings` loaded from `LIVENESS_*` environment variables).

---

## 3. Tracing the Current Capture Flow

### 3.1 Session Creation & Authorization
1. Frontend calls `POST /api/v1/sessions` with optional `registration_ref`.
2. Backend generates a 32-byte cryptographic nonce (`session_nonce`) and stores its SHA-256 hash (`nonce_hash`).
3. `ActiveChallengeService.generate_challenge_sequence()` generates a 4-step sequence:
   - Step 0 (Mandatory): `CENTER_FACE` (establishes frontal, neutral reference frame).
   - Steps 1, 2, 3: Cryptographically sampled without replacement from `[TURN_LEFT, TURN_RIGHT, SMILE, BLINK, RAISE_EYEBROWS]`.
4. Only Step 0 is returned in `current_challenge` (progressive disclosure); future steps are never disclosed ahead of time.

### 3.2 Frame Acquisition & Current Manual Trigger
1. User clicks `"Enable Camera & Start Liveness Check"`. Camera initializes via `navigator.mediaDevices.getUserMedia`.
2. Frontend renders live video and displays current challenge instructions (`ChallengeInstructions.tsx`).
3. **Current Manual Bottleneck:** The user must manually click `"Capture & Verify Step"` (`data-testid="verify-challenge-btn"`).
4. `useLivenessSession.ts` captures 3 frames at 180ms intervals onto an offscreen canvas, runs client-side usability checks (`captureAndAssessVideoFrame` for resolution, brightness, blur), base64-encodes them, and submits them to `POST /api/v1/sessions/{session_id}/challenge/frames`.

### 3.3 Server-Side Challenge Verification & MediaPipe Lifecycle
1. Request arrives with `session_nonce`, `challenge_id`, and frames.
2. Nonce hash and timestamps are validated (must be strictly monotonic).
3. **MediaPipe Lifecycle:** `MediaPipeFaceDetector` is a thread-safe singleton inside `ServiceContainer`. It is **reused across all requests** (not recreated), protected by a thread lock.
4. Frames are decoded and passed to MediaPipe FaceLandmarker:
   - Evaluates head pose (Euler angles yaw/pitch/roll).
   - Extracts 52 facial blendshapes (mouthSmile, eyeBlink, browUp).
5. `ActiveChallengeService.evaluate_challenge_batch()` evaluates whether the batch meets challenge criteria:
   - `CENTER_FACE`: Yaw $\le 12^\circ$, Pitch $\le 16^\circ$, and neutral expression ($\text{smile} \le 0.35, \text{brow} \le 0.35$).
   - `TURN_LEFT`: Yaw $\le -13^\circ$.
   - `TURN_RIGHT`: Yaw $\ge +13^\circ$.
   - `SMILE`: Mouth smile score $\ge 0.55$.
   - `RAISE_EYEBROWS`: Brow raise score $\ge 0.50$.
   - `BLINK`: Temporal transition requiring both an eyes-closed frame ($\ge 0.50$) and an eyes-open frame ($\le 0.30$) within the batch.
6. If the challenge passes:
   - Best representative frame is saved in `TransientSessionBufferStore` (RAM only).
   - Step is marked completed; session advances `current_step_index`.
   - Next challenge instruction is returned.
7. If the challenge fails:
   - Attempts increment. `MAX_ATTEMPTS_PER_CHALLENGE = 8`.
   - Feedback hint is returned to the user.

### 3.4 Final Decision & Passive PAD
1. When `all_challenges_completed == True`, frontend calls `POST /api/v1/sessions/{session_id}/complete`.
2. Backend pops transient buffers from RAM.
3. Evaluates representative frames through MiniFASNet ONNX models (multi-scale fusion).
4. Only if aggregate score $\ge 0.65$ and min frame score $\ge 0.45$:
   - Frontal frame is saved to private disk (`LocalPrivatePhotoStorage`) with 40-hex random key.
   - HMAC/SHA-256 verification signature is computed.
   - Photo access token is generated.
   - Session marked `VALIDATED`.
5. All intermediate frames in RAM are purged.

---

## 4. Existing Components & Security Controls

| Component | Status | Security / Functional Properties |
| :--- | :--- | :--- |
| `MediaPipeFaceDetector` | Verified & Reused | Singleton instance, SHA-256 verified bundle, thread-locked, landmark + blendshape extraction. |
| `MiniFASNetOnnxProvider` | Verified & Reused | Multi-scale ONNX execution (V2 + V1SE), fail-closed error handling. |
| `ActiveChallengeService` | Verified & Extended | CSPRNG sequencing, neutral CENTER_FACE gate, multi-gesture evaluation. |
| `TransientSessionBufferStore` | Verified | In-memory only, bounded frame count, immediate purge on session termination. |
| `LocalPrivatePhotoStorage` | Verified | Strips EXIF, writes outside web root, access gated by session-bound token. |
| Nonce & Progressive Disclosure | Verified | SHA-256 hashed nonces; future steps hidden until prior step passes. |
| Client-side Usability Guard | Verified | Canvas-based brightness, resolution, and sharpness heuristics for instant feedback. |

---

## 5. Gaps & Bottlenecks for Automatic Capture

1. **Manual Capture Dependency:** The current UI requires clicking `"Capture & Verify Step"`. An automated capture loop must operate hands-free while maintaining framing and responsiveness.
2. **Batch Isolation vs. Temporal Continuity:** Currently, the backend evaluates each HTTP batch in isolation. If a user blinks across batch boundaries or turns progressively, each isolated batch may fail if it captures only a partial movement.
3. **Attempt Limit Exhaustion:** `MAX_ATTEMPTS_PER_CHALLENGE = 8` was calibrated for manual clicks. In an automated capture loop submitting batches at 2-3 Hz, 8 attempts would be exhausted in under 3 seconds, prematurely failing the session.
4. **Rate Limiting Budget:** Default sliding-window rate limit is 60 req/min. Continuous batch submissions can easily breach 60 req/min if not properly budgeted or configured.
5. **Frontend State Machine Granularity:** The UI currently only defines 8 broad phases (`IDLE`, `CHECKING_SERVICE`, `READY_FOR_CHALLENGE`, `CAPTURING_BATCH`, `FINALIZING`, `COMPLETED`, `SERVICE_UNAVAILABLE`, `ERROR`). It lacks fine-grained states for `WAITING_FOR_FACE`, `CHALLENGE_ACTIVE`, `ACTION_DETECTED`, `SUBMITTING_EVIDENCE`, `CHALLENGE_VERIFIED`.

---

## 6. Architecture Decision (Pattern Selection)

- **Pattern A (Browser Model):** Requires downloading `@mediapipe/tasks-vision` (~4MB bundle), running WebAssembly/WebGL in the browser, duplicating 3D landmark math, and managing cross-browser GPU/WASM bugs. Rejected per constraint: *"Do not introduce a second facial-analysis model into the browser merely to satisfy the requirement."*
- **Pattern B (Server-Driven Incremental Detection with Hybrid Client Guidance):** Selected.
  - Frontend performs lightweight, zero-dependency canvas usability & stability analysis.
  - Controlled-frequency frame sampling with bounded rolling memory buffers.
  - Automatic submission with in-flight concurrency locks (`isSubmittingRef`).
  - Backend maintains authoritative temporal state across batches for the active challenge step.
  - Automatic step progression upon backend confirmation.

---

## 7. Implementation Plan & Modified Files

1. **Backend:**
   - Extend `TransientSessionBufferStore` and `CaptureSessionService` to track rolling temporal history for the active challenge.
   - Adjust `MAX_ATTEMPTS_PER_CHALLENGE` and rate limit defaults to support automated streaming cadence.
   - Update tests in `test_challenge_and_session.py` and `test_api_workflow.py` to match the 4-step sequence (`TOTAL_CHALLENGE_STEPS = 4`).
2. **Frontend:**
   - Implement the explicit 12-state automatic capture state machine in `useLivenessSession.ts`.
   - Implement continuous frame assessment and automatic batch submission in `useLivenessSession.ts`.
   - Update `ChallengeInstructions.tsx` and `CameraPreview.tsx` to provide visual feedback for automatic capture (analyzing indicator, progress ring/pulse) while retaining manual fallback.
   - Update frontend tests in `App.test.tsx` to verify the automated workflow.
3. **Verification:**
   - Execute full backend test suite (`pytest`).
   - Execute frontend test suite (`vitest`).
   - Validate end-to-end integration and security bounds.
