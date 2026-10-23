# Evaluation Methodology and Known Limitations

## 1. Security Calibration & Evaluation Procedure

Passive anti-spoofing thresholds (`LIVENESS_SPOOF_REAL_SCORE_THRESHOLD`, default `0.65`, and `LIVENESS_SPOOF_MIN_FRAME_SCORE`, default `0.45`) are baseline operating points that **must be calibrated** against representative samples captured from your target deployment cameras and lighting environments.

### Running Quantitative Offline Evaluation

Collect authorized test images organized into two directories (`genuine/` and `spoof/`) and run:

```powershell
Set-Location backend
.\.venv\Scripts\python.exe scripts/evaluate_models.py `
  --genuine-dir C:\path\to\eval_dataset\genuine `
  --spoof-dir C:\path\to\eval_dataset\spoof `
  --threshold 0.65
```

The script outputs a JSON report containing:

- `genuine_count` and `spoof_count` (plus `skipped_invalid_face_count`)
- `FRR_BPCER` (False Rejection Rate / Bona Fide Presentation Classification Error Rate): proportion of genuine live samples rejected (`real_probability < threshold`)
- `FAR_APCER` (False Acceptance Rate / Attack Presentation Classification Error Rate): proportion of spoof samples accepted (`real_probability >= threshold`)
- `ACER` (Average Classification Error Rate): `(FRR_BPCER + FAR_APCER) / 2`

### Upstream Sample Sanity Check

When evaluated on the reference sample pair published in upstream `minivision-ai/Silent-Face-Anti-Spoofing/images/sample/`:

- `image_T1.jpg` (Genuine sample): `MiniFASNetV2` score = `1.0000`, `MiniFASNetV1SE` score = `0.9997` (Ensemble `real_score = 0.9999` -> **Accepted**).
- `image_F1.jpg` (Spoof sample): `MiniFASNetV2` score = `0.0069`, `MiniFASNetV1SE` score = `0.1232` (Ensemble `real_score = 0.0650` -> **Rejected**).

*(Note: A 2-image sanity check verifies preprocessing correctness only and is not a statistically significant security evaluation.)*

## 2. Recommended Evaluation Matrix

Before production deployment, evaluate across a structured matrix of at least 100+ genuine and 100+ presentation-attack captures covering:

1. **Print Attacks:** Matte and glossy printed photographs (A4 and passport sizes) held flat, curved, or tilted to mimic head turns.
2. **Mobile Phone Replay Attacks:** Static photos and prerecorded high-frame-rate videos displayed on OLED/LCD smartphones at varying screen brightness levels.
3. **Monitor & Tablet Replay Attacks:** Prerecorded videos displayed on high-resolution laptops, tablets, and desktop monitors where screen bezels may fall outside the `2.7` crop scale (mitigated by the `4.0` scale `MiniFASNetV1SE` model, which captures wider context).
4. **Environmental & Camera Variations:**
   - Low-end 720p webcams vs. 1080p/4K USB cameras and mobile front cameras.
   - Warm indoor lighting, fluorescent office lighting, dim environments, and side lighting.
   - Users wearing prescription eyeglasses (reflections on lenses), sunglasses, hats, or facial hair.
5. **Accessibility & Edge-Case Genuine Users:**
   - Genuine users who turn their head slowly or have limited cervical range of motion (adjust `LIVENESS_TURN_LEFT_MIN_YAW_DEG`, `LIVENESS_TURN_RIGHT_MIN_YAW_DEG`, and `LIVENESS_SESSION_TTL_SECONDS`).
   - Interrupted captures, network retries, and repeated attempts.

## 3. Known Limitations

- **Baseline Status:** This system is a baseline implementation and is **not** ISO/IEC 30107-3 (PAD) certified.
- **RGB Camera Constraints:** Ordinary monocular 2D RGB webcams lack true hardware depth (IR structured light / Time-of-Flight). High-precision 3D silicone masks or studio-grade bezel-less OLED replays under controlled lighting can degrade passive 2D CNN accuracy.
- **Camera-Feed Injection / Virtual Cameras:** Because frame capture occurs in the browser via `navigator.mediaDevices.getUserMedia()`, software virtual cameras (OBS VirtualCam, driver-level video injection, or compromised browser environments) cannot be prevented purely by image-level liveness models. High-assurance deployments should combine this pipeline with device attestation, WebRTC stream metadata inspection, and server-side anomaly detection.
