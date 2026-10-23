import React from "react";
import type { CameraState } from "../hooks/useCamera";

export interface CameraPermissionPromptProps {
  cameraState: CameraState;
  cameraError: string | null;
  isStarting: boolean;
  onStartCapture: () => void;
}

export const CameraPermissionPrompt: React.FC<CameraPermissionPromptProps> = ({
  cameraState,
  cameraError,
  isStarting,
  onStartCapture,
}) => {
  const isErrorState =
    cameraState === "PERMISSION_DENIED" ||
    cameraState === "NO_CAMERA" ||
    cameraState === "CAMERA_IN_USE" ||
    cameraState === "UNSUPPORTED" ||
    cameraState === "INIT_ERROR";

  return (
    <div className="camera-prompt-box" role="region" aria-label="Camera initialization">
      <h3>Camera Access Required</h3>
      <p>
        To verify physical presence, this system performs a brief active head-movement check and
        local passive anti-spoofing analysis. Raw video is processed in memory and never stored.
      </p>

      {isErrorState && cameraError && (
        <div className="alert alert-error" role="alert">
          <strong>
            {cameraState === "PERMISSION_DENIED" && "Camera Permission Denied: "}
            {cameraState === "NO_CAMERA" && "No Camera Found: "}
            {cameraState === "CAMERA_IN_USE" && "Camera Unavailable: "}
            {cameraState === "UNSUPPORTED" && "Browser Not Supported: "}
            {cameraState === "INIT_ERROR" && "Camera Initialization Error: "}
          </strong>
          <span>{cameraError}</span>
        </div>
      )}

      <ul className="checklist-guidelines">
        <li>Position yourself in a well-lit area without strong backlighting.</li>
        <li>Ensure only your face is visible in the camera frame.</li>
        <li>Keep your face centered inside the oval guide and follow the on-screen prompts.</li>
      </ul>

      <button
        type="button"
        className="btn btn-primary"
        disabled={isStarting}
        onClick={onStartCapture}
      >
        {isStarting
          ? "Initializing Camera & Session..."
          : isErrorState
            ? "Retry Camera & Start Session"
            : "Enable Camera & Start Liveness Check"}
      </button>
    </div>
  );
};
