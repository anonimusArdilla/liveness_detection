import React from "react";
import { livenessApiClient } from "../services/livenessApi";
import type { SessionResultResponse } from "../types/liveness";

export interface ValidationResultCardProps {
  result: SessionResultResponse;
  onRetry: () => void;
}

function getFailureCategoryLabel(outcome: SessionResultResponse["outcome"]): string {
  switch (outcome) {
    case "QUALITY_FAILED":
      return "Image Quality / Positioning Check Failed";
    case "CHALLENGE_FAILED":
      return "Active Head-Movement Challenge Failed";
    case "SPOOF_SUSPECTED":
      return "Liveness Failed — Presentation Attack (Spoof) Suspected";
    case "VALIDATION_UNAVAILABLE":
      return "Validation Service Unavailable (Failed Closed)";
    case "EXPIRED":
      return "Capture Session Expired";
    default:
      return "Validation Rejected";
  }
}

export const ValidationResultCard: React.FC<ValidationResultCardProps> = ({ result, onRetry }) => {
  const isSuccess = result.outcome === "VALIDATED" && result.validated === true;
  const photoUrl =
    isSuccess && result.photo_available && result.photo_access_token
      ? livenessApiClient.getProtectedPhotoUrl(result.session_id, result.photo_access_token)
      : null;

  return (
    <div
      className={`result-card ${isSuccess ? "result-card-success" : "result-card-rejected"}`}
      role="region"
      aria-labelledby="validation-result-heading"
      data-testid="validation-result-card"
    >
      <div className="result-header">
        <span className={`outcome-badge ${isSuccess ? "badge-success" : "badge-danger"}`}>
          {result.outcome}
        </span>
        <h3 id="validation-result-heading">
          {isSuccess
            ? "Physical Liveness Confirmed by Server"
            : getFailureCategoryLabel(result.outcome)}
        </h3>
      </div>

      <p className="result-reason">{result.reason}</p>

      <div className="result-checks-grid">
        <div className="result-check-item">
          <span className="check-label">Image Quality</span>
          <span className={result.quality_passed ? "check-pass" : "check-fail"}>
            {result.quality_passed ? "PASSED" : "FAILED"}
          </span>
        </div>
        <div className="result-check-item">
          <span className="check-label">Active Landmark Challenges</span>
          <span className={result.challenges_passed ? "check-pass" : "check-fail"}>
            {result.challenges_passed ? "PASSED" : "FAILED"}
          </span>
        </div>
        <div className="result-check-item">
          <span className="check-label">Passive Anti-Spoofing (ONNX)</span>
          <span className={result.passive_spoof_passed ? "check-pass" : "check-fail"}>
            {result.passive_spoof_passed ? "PASSED" : "FAILED"}
          </span>
        </div>
      </div>

      {result.passive_report && (
        <div className="passive-telemetry-box">
          <strong>Passive Model Evaluation ({result.passive_report.model_name}):</strong>{" "}
          Aggregate Real Score:{" "}
          <code>{(result.passive_report.aggregate_real_score * 100).toFixed(1)}%</code> | Min Frame
          Score: <code>{(result.passive_report.min_frame_real_score * 100).toFixed(1)}%</code> |
          Policy Threshold: <code>{(result.passive_report.threshold_applied * 100).toFixed(1)}%</code>{" "}
          ({result.passive_report.frames_evaluated} frames evaluated)
        </div>
      )}

      {result.quality_issues.length > 0 && (
        <div className="result-quality-issues">
          <strong>Quality Issues Recorded:</strong>
          <ul>
            {result.quality_issues.map((issue) => (
              <li key={issue.code}>
                <code>{issue.code}</code>: {issue.message}
              </li>
            ))}
          </ul>
        </div>
      )}

      {photoUrl && (
        <div className="accepted-photo-preview">
          <h4>Accepted Registration Portrait</h4>
          <img
            src={photoUrl}
            alt="Validated applicant capture"
            className="validated-portrait-img"
          />
          {result.verification_signature && (
            <p className="signature-caption">
              Verification Signature: <code>{result.verification_signature.slice(0, 24)}...</code>
            </p>
          )}
        </div>
      )}

      <div className="result-actions">
        <button type="button" className="btn btn-primary" onClick={onRetry}>
          {isSuccess ? "Start Another Verification" : "Retry Liveness Capture"}
        </button>
      </div>
    </div>
  );
};
