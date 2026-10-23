import React from "react";
import type { QualityIssue, QualityReport } from "../types/liveness";

export interface QualityGuidanceBannerProps {
  issues: QualityIssue[];
  qualityReport: QualityReport | null;
}

export const QualityGuidanceBanner: React.FC<QualityGuidanceBannerProps> = ({
  issues,
  qualityReport,
}) => {
  return (
    <div className="quality-guidance-panel" aria-live="polite">
      <div className="quality-header">
        <strong>Image Quality & Positioning Guidance</strong>
        {qualityReport && (
          <span className="quality-metrics-pill">
            {qualityReport.width}x{qualityReport.height} | Brightness: {qualityReport.brightness} |
            Sharpness: {qualityReport.blur_score}
            {qualityReport.head_pose
              ? ` | Yaw: ${qualityReport.head_pose.yaw.toFixed(1)}°`
              : ""}
          </span>
        )}
      </div>

      {issues.length > 0 ? (
        <ul className="quality-issues-list" role="alert" aria-label="Actionable quality corrections">
          {issues.map((issue) => (
            <li key={issue.code} className="quality-issue-item">
              <span className="issue-badge">{issue.code.replace(/_/g, " ")}</span>{" "}
              <span>{issue.message}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="quality-ok-text">
          {qualityReport
            ? "Image quality and face positioning passed server checks."
            : "Keep your face clearly visible inside the center guide."}
        </p>
      )}
    </div>
  );
};
