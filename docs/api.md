# REST API Contract (`/api/v1`)

All endpoints return JSON (except `GET /api/v1/sessions/{session_id}/photo`, which returns `image/jpeg`).

## Error Response Envelope

Every error response uses a consistent JSON structure:

```json
{
  "error": {
    "code": "SESSION_EXPIRED",
    "message": "Capture session has expired. Please start a new session.",
    "details": {}
  }
}
```

### Machine-Readable Error Codes

| Error Code | HTTP Status | Description |
| :--- | :--- | :--- |
| `VALIDATION_ERROR` | `400` / `422` | Malformed request schema or frame count outside `[min_frames_per_batch, max_frames_per_batch]`. |
| `IMAGE_DECODE_FAILED` | `400` | Base64 payload could not be decoded as a valid 3-channel color image. |
| `FRAME_ORDER_INVALID` | `400` | Frame `timestamp_ms` values are not strictly increasing or replay an earlier timestamp. |
| `SESSION_NONCE_MISMATCH` | `403` | Invalid `session_nonce` for the session. |
| `UNAUTHORIZED_ACCESS` | `403` | Missing or invalid `X-Photo-Token` when requesting an accepted photograph. |
| `SESSION_NOT_FOUND` | `404` | Session ID does not exist. |
| `PHOTO_NOT_FOUND` | `404` | No accepted photograph exists for the session (e.g., session was not `VALIDATED`). |
| `CHALLENGE_MISMATCH` | `409` | Submitted `challenge_id` does not match the current active challenge step. |
| `SESSION_INVALID_STATE` | `409` | Session is already in a terminal outcome state and cannot be reused or re-finalized. |
| `SESSION_EXPIRED` | `410` | Session exceeded `LIVENESS_SESSION_TTL_SECONDS`. |
| `PAYLOAD_TOO_LARGE` | `413` | Request body exceeds `LIVENESS_MAX_REQUEST_BYTES`. |
| `RATE_LIMIT_EXCEEDED` | `429` | Client IP exceeded `LIVENESS_RATE_LIMIT_REQUESTS_PER_MINUTE`. |
| `SERVICE_UNAVAILABLE` | `503` | Required vision/anti-spoofing models or storage are unavailable (fail-closed). |
| `INTERNAL_ERROR` | `500` | Unexpected server error; validation fails closed. |

---

## Endpoints

### 1. `GET /api/v1/health` and `GET /api/v1/health/ready`

Reports service liveness and readiness. `/api/v1/health/ready` returns HTTP `200 OK` only when all ONNX and MediaPipe models are loaded, SHA-256 verified, and database/storage checks pass; otherwise it returns HTTP `503 Service Unavailable`.

- **Response (`200 OK` / `503 Service Unavailable`):**
  ```json
  {
    "status": "ready",
    "ready": true,
    "version": "1.0.0",
    "models": [
      {
        "name": "MediaPipe-FaceLandmarker",
        "ready": true,
        "provider": "CPUExecutionProvider (TFLite XNNPACK)",
        "path": "face_landmarker.task",
        "checksum_verified": true,
        "reason": null
      },
      {
        "name": "MiniFASNetV2 (scale 2.7)",
        "ready": true,
        "provider": "CPUExecutionProvider",
        "path": "MiniFASNetV2.onnx",
        "checksum_verified": true,
        "reason": null
      },
      {
        "name": "MiniFASNetV1SE (scale 4.0)",
        "ready": true,
        "provider": "CPUExecutionProvider",
        "path": "MiniFASNetV1SE.onnx",
        "checksum_verified": true,
        "reason": null
      }
    ],
    "database_ready": true,
    "storage_ready": true
  }
  ```

---

### 2. `POST /api/v1/sessions`

Creates a new short-lived capture session, one-time cryptographic nonce, and randomized challenge sequence.

- **Request (`application/json`):**
  ```json
  {
    "registration_ref": "REG-2026-001"
  }
  ```
- **Response (`201 Created`):**
  ```json
  {
    "session_id": "52f0c812-748b-4a31-98ab-1988d7e01f23",
    "session_nonce": "9mY8x...",
    "outcome": "PENDING",
    "expires_at": "2026-10-22T21:35:00Z",
    "total_challenges": 3,
    "current_challenge": {
      "challenge_id": "a1b2c3d4-...",
      "step_index": 0,
      "total_steps": 3,
      "challenge_type": "CENTER_FACE",
      "instruction": "Keep your face centered and look straight at the camera.",
      "completed": false
    }
  }
  ```
- **Idempotency & Retry:** Non-idempotent; each call creates a distinct session.

---

### 3. `GET /api/v1/sessions/{session_id}/challenge`

Retrieves the currently active challenge step without revealing future steps.

- **Response (`200 OK`):** `CurrentChallengeResponse`

---

### 4. `POST /api/v1/sessions/{session_id}/challenge/frames`

Submits a bounded batch of frames (`2..10` frames by default) for the current challenge step.

- **Request (`application/json`):**
  ```json
  {
    "session_nonce": "9mY8x...",
    "challenge_id": "a1b2c3d4-...",
    "frames": [
      {
        "timestamp_ms": 1761168000100,
        "image_base64": "data:image/jpeg;base64,/9j/4AAQ..."
      },
      {
        "timestamp_ms": 1761168000320,
        "image_base64": "data:image/jpeg;base64,/9j/4AAQ..."
      }
    ]
  }
  ```
- **Validation Rules:**
  - `session_nonce` must match the session's SHA-256 `nonce_hash`.
  - `challenge_id` must match the active uncompleted step.
  - `frames` timestamps must be strictly increasing across the entire session.
- **Response (`200 OK`):** Returns `challenge_passed`, `quality_report` (including actionable `issues`), and `next_challenge` if the step succeeded.

---

### 5. `POST /api/v1/sessions/{session_id}/complete`

Finalizes the session after all active challenges have been completed. Executes passive ONNX MiniFASNet inference across sampled session frames, persists the accepted frontal portrait if and only if `outcome == "VALIDATED"`, and purges all in-memory frame buffers.

- **Request (`application/json`):**
  ```json
  {
    "session_nonce": "9mY8x..."
  }
  ```
- **Response (`200 OK`):** `SessionResultResponse` (includes `photo_access_token` and `verification_signature` when `outcome == "VALIDATED"`).

---

### 6. `GET /api/v1/sessions/{session_id}`

Retrieves the outcome and metadata of a session (without re-exposing the one-time `photo_access_token`).

---

### 7. `GET /api/v1/sessions/{session_id}/photo`

Returns the accepted JPEG portrait for a `VALIDATED` session.

- **Authentication:** Requires `X-Photo-Token: <photo_access_token>` header (or `?token=<photo_access_token>` query parameter).

---

## Integration into an External Registration Backend

1. The registration frontend obtains a `registration_ref` from the main registration backend and passes it to `POST /api/v1/sessions`.
2. Never trust the frontend's claim that liveness succeeded. When the user submits their registration form, the **registration backend** must call `GET /api/v1/sessions/{session_id}` server-to-server and verify:
   - `outcome == "VALIDATED"` and `validated == true`
   - `registration_ref` matches the expected registration transaction ID
   - `completed_at` is recent
   - Optional: verify `verification_signature` (`SHA256(session_id | registration_ref | outcome | completed_at | nonce_hash)`).
