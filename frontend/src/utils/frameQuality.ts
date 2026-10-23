import type { QualityIssue } from "../types/liveness";

export interface ClientFrameCheckResult {
  passed: boolean;
  width: number;
  height: number;
  brightness: number;
  sharpness: number;
  issues: QualityIssue[];
  imageBase64: string;
}

/**
 * Usability-only client-side image quality check and JPEG encoder.
 *
 * Security note: This function is strictly for immediate UI feedback and frame sampling.
 * All security-relevant quality, landmark, and anti-spoofing checks are enforced by the backend.
 */
export function captureAndAssessVideoFrame(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  minWidth = 320,
  minHeight = 240,
  minBrightness = 45,
  maxBrightness = 225,
  minSharpness = 8,
): ClientFrameCheckResult {
  const width = video.videoWidth || 640;
  const height = video.videoHeight || 480;
  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const issues: QualityIssue[] = [];

  if (width < minWidth || height < minHeight) {
    issues.push({
      code: "RESOLUTION_TOO_LOW",
      message: `Camera resolution (${width}x${height}) is below the required minimum (${minWidth}x${minHeight}).`,
    });
  }

  let brightness = 128;
  let sharpness = 50;
  let imageBase64 = "";

  if (ctx) {
    ctx.drawImage(video, 0, 0, width, height);
    const imgData = ctx.getImageData(0, 0, width, height);
    const data = imgData.data;

    let lumSum = 0;
    let diffSum = 0;
    let sampleCount = 0;
    const stride = 16; // Sample every 4th pixel for low CPU overhead

    for (let i = 0; i < data.length - stride; i += stride) {
      const r = data[i] ?? 0;
      const g = data[i + 1] ?? 0;
      const b = data[i + 2] ?? 0;
      const lum = 0.299 * r + 0.587 * g + 0.114 * b;

      const rNext = data[i + stride] ?? 0;
      const gNext = data[i + stride + 1] ?? 0;
      const bNext = data[i + stride + 2] ?? 0;
      const lumNext = 0.299 * rNext + 0.587 * gNext + 0.114 * bNext;

      lumSum += lum;
      diffSum += Math.abs(lum - lumNext);
      sampleCount += 1;
    }

    if (sampleCount > 0) {
      brightness = Math.round((lumSum / sampleCount) * 10) / 10;
      sharpness = Math.round((diffSum / sampleCount) * 10) / 10;
    }

    if (brightness < minBrightness) {
      issues.push({
        code: "LIGHTING_TOO_DARK",
        message: "Lighting is too dim. Please move to a brighter area.",
      });
    } else if (brightness > maxBrightness) {
      issues.push({
        code: "LIGHTING_TOO_BRIGHT",
        message: "Lighting is too bright or overexposed. Avoid direct glare.",
      });
    }

    if (sharpness < minSharpness) {
      issues.push({
        code: "IMAGE_TOO_BLURRY",
        message: "Camera image appears blurry. Hold still and ensure your camera is in focus.",
      });
    }

    imageBase64 = canvas.toDataURL("image/jpeg", 0.86);
    // Clear transient canvas pixels immediately after encoding
    ctx.clearRect(0, 0, width, height);
  }

  return {
    passed: issues.length === 0 && imageBase64.length > 0,
    width,
    height,
    brightness,
    sharpness,
    issues,
    imageBase64,
  };
}
