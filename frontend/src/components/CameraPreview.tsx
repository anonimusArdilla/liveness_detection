import React from "react";
import type { ChallengeType } from "../types/liveness";

export interface CameraPreviewProps {
  videoRef: React.RefObject<HTMLVideoElement>;
  isActive: boolean;
  currentChallengeType: ChallengeType | null;
  qualityPassed: boolean;
}

export const CameraPreview: React.FC<CameraPreviewProps> = ({
  videoRef,
  isActive,
  currentChallengeType,
  qualityPassed,
}) => {
  return (
    <div className="camera-preview-container" aria-label="Live camera preview">
      <video
        ref={videoRef}
        className="camera-video-element"
        autoPlay
        playsInline
        muted
        data-testid="live-camera-video"
      />
      {isActive && (
        <svg
          className="face-guide-overlay"
          viewBox="0 0 640 480"
          preserveAspectRatio="none"
          aria-hidden="true"
        >
          <defs>
            <mask id="oval-cutout">
              <rect width="640" height="480" fill="white" />
              <ellipse cx="320" cy="235" rx="135" ry="170" fill="black" />
            </mask>
          </defs>
          <rect
            width="640"
            height="480"
            fill="rgba(15, 23, 42, 0.45)"
            mask="url(#oval-cutout)"
          />
          <ellipse
            cx="320"
            cy="235"
            rx="135"
            ry="170"
            fill="none"
            stroke={qualityPassed ? "#16a34a" : "#cbd5e1"}
            strokeWidth="3"
            strokeDasharray={currentChallengeType === "CENTER_FACE" ? "none" : "10 6"}
          />
        </svg>
      )}
    </div>
  );
};
