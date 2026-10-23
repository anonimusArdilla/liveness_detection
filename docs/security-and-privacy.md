# Security and Privacy Architecture

## 1. Biometric Data Minimization & In-Memory Processing

- **Zero Raw Video Retention:** Neither the frontend nor the backend records or stores video files (`.mp4`, `.webm`, `.avi`).
- **In-Memory Frame Decoding:** Incoming base64 JPEG/PNG frames are decoded directly in memory using `np.frombuffer` and `cv2.imdecode`. No temporary files are spooled to disk.
- **Explicit Buffer Release:** Decoded frame arrays (`np.ndarray`), MediaPipe `mp.Image` wrappers, and ONNX input tensors are explicitly deleted in `finally` blocks immediately after each operation.
- **Bounded Transient Session Cache (`TransientSessionBufferStore`):** During a `PENDING` session, at most `LIVENESS_PASSIVE_SAMPLE_FRAMES_MAX` (default `5`) representative frames and one frontal candidate frame are held in RAM. As soon as `complete_session` finishes, a session fails, or a session expires, `TransientSessionBufferStore` explicitly frees all numpy arrays.
- **Conditional Portrait Persistence:** The accepted frontal photograph is written to disk **only** after all quality checks, active challenges, and passive ONNX anti-spoofing checks succeed (`outcome == "VALIDATED"`). Rejected or abandoned sessions leave zero image files on disk.
- **EXIF Stripping & Opaque Keys:** Accepted portraits are re-encoded from raw pixel arrays into fresh JPEGs (stripping any camera EXIF/GPS metadata) and saved under a 40-character hexadecimal SHA-256 key (`^[a-f0-9]{40}\.jpg$`) in a private directory outside the frontend web root.

## 2. Persisted Metadata Inventory

The SQLite/PostgreSQL database (`capture_sessions` and `challenge_steps` tables) retains only:

- `session_id` (UUIDv4) and optional `registration_ref`
- `nonce_hash` (SHA-256 digest of the one-time session nonce; raw nonce is never stored)
- `outcome`, `created_at`, `expires_at`, `completed_at`
- `quality_passed`, `challenges_passed`, `passive_spoof_passed`
- `aggregate_real_score`, `min_frame_real_score`, `frames_evaluated`, `passive_model_name`
- `reason` and non-biometric `quality_issues_json`
- `photo_storage_key` (internal filename, never returned by the API) and `photo_token_hash` (SHA-256 digest of the photo access token)
- `verification_signature`

## 3. Log Redaction & Error Safety

- `PrivacyRedactingFilter` in `backend/app/core/logging.py` automatically scrubs `image_base64`, `data:image/...;base64,...`, `session_nonce`, and `photo_access_token` patterns from all log records.
- FastAPI `RequestValidationError` handlers strip raw input values so malformed base64 frames or nonces are never echoed back in HTTP 422 error payloads or logs.

## 4. Anti-Replay & Fail-Closed Controls

- **Server-Controlled Progressive Challenges:** Challenge sequences are generated using `secrets.SystemRandom()` on the server and disclosed one step at a time.
- **Monotonic Timestamp Enforcement:** Every submitted frame must carry a strictly increasing `timestamp_ms` across the session. Out-of-order or replayed frames are rejected with `FRAME_ORDER_INVALID`.
- **Constant-Time Secret Comparisons:** Session nonces and photo access tokens are verified using `secrets.compare_digest` against stored SHA-256 hashes.
- **Fail-Closed Guarantee:** Missing model files, SHA-256 checksum mismatches, or runtime inference exceptions immediately mark the service as not ready (`503 SERVICE_UNAVAILABLE`) or transition the session to `VALIDATION_UNAVAILABLE`. The system never defaults to a positive validation result on error.
