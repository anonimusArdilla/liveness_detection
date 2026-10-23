import { useCallback, useEffect, useRef, useState } from "react";

export type CameraState =
  | "IDLE"
  | "REQUESTING"
  | "ACTIVE"
  | "PERMISSION_DENIED"
  | "NO_CAMERA"
  | "CAMERA_IN_USE"
  | "UNSUPPORTED"
  | "INIT_ERROR";

export interface UseCameraResult {
  cameraState: CameraState;
  cameraError: string | null;
  videoRef: React.RefObject<HTMLVideoElement>;
  startCamera: () => Promise<boolean>;
  stopCamera: () => void;
}

export function useCamera(): UseCameraResult {
  const [cameraState, setCameraState] = useState<CameraState>("IDLE");
  const [cameraError, setCameraError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      for (const track of streamRef.current.getTracks()) {
        track.stop();
      }
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraState((prev) => (prev === "ACTIVE" || prev === "REQUESTING" ? "IDLE" : prev));
  }, []);

  const startCamera = useCallback(async (): Promise<boolean> => {
    stopCamera();
    setCameraError(null);

    if (
      typeof navigator === "undefined" ||
      !navigator.mediaDevices ||
      typeof navigator.mediaDevices.getUserMedia !== "function"
    ) {
      setCameraState("UNSUPPORTED");
      setCameraError(
        "Your browser does not support camera access (navigator.mediaDevices.getUserMedia). Please use a modern browser over HTTPS or localhost.",
      );
      return false;
    }

    setCameraState("REQUESTING");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 640, min: 320 },
          height: { ideal: 480, min: 240 },
        },
        audio: false,
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        try {
          await videoRef.current.play();
        } catch {
          // Ignore autoplay promise rejection in headless/test DOM environments
        }
      }
      setCameraState("ACTIVE");
      return true;
    } catch (err: unknown) {
      const domErr = err as { name?: string; message?: string };
      const errName = domErr?.name ?? "";

      if (errName === "NotAllowedError" || errName === "PermissionDeniedError") {
        setCameraState("PERMISSION_DENIED");
        setCameraError(
          "Camera permission was denied. Please grant camera access in your browser address bar settings and click Retry.",
        );
      } else if (errName === "NotFoundError" || errName === "DevicesNotFoundError") {
        setCameraState("NO_CAMERA");
        setCameraError(
          "No camera device was detected on this system. Connect a webcam and click Retry.",
        );
      } else if (errName === "NotReadableError" || errName === "TrackStartError") {
        setCameraState("CAMERA_IN_USE");
        setCameraError(
          "Your camera is currently in use by another application. Close other video apps and click Retry.",
        );
      } else {
        setCameraState("INIT_ERROR");
        setCameraError(
          domErr?.message
            ? `Failed to initialize camera: ${domErr.message}`
            : "An unexpected error occurred while initializing the camera.",
        );
      }
      return false;
    }
  }, [stopCamera]);

  useEffect(() => {
    const handlePageHide = () => {
      stopCamera();
    };
    window.addEventListener("pagehide", handlePageHide);
    return () => {
      window.removeEventListener("pagehide", handlePageHide);
      stopCamera();
    };
  }, [stopCamera]);

  return {
    cameraState,
    cameraError,
    videoRef,
    startCamera,
    stopCamera,
  };
}
