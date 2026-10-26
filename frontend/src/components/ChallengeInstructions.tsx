import React from "react";
import type { ChallengeInstruction } from "../types/liveness";

export interface ChallengeInstructionsProps {
  challenge: ChallengeInstruction;
  feedbackMessage: string | null;
  isSubmitting: boolean;
  onCaptureStep: () => void;
  onAbandon: () => void;
}

export const ChallengeInstructions: React.FC<ChallengeInstructionsProps> = ({
  challenge,
  feedbackMessage,
  isSubmitting,
  onCaptureStep,
  onAbandon,
}) => {
  return (
    <div className="challenge-panel" role="region" aria-label="Active liveness challenge">
      <div className="challenge-badge-row">
        <span className="step-pill">
          Challenge Step {challenge.step_index + 1} of {challenge.total_steps}
        </span>
        <span className="challenge-type-tag">{challenge.challenge_type.replace(/_/g, " ")}</span>
      </div>

      <div className="auto-capture-badge" aria-live="polite">
        <span className="auto-pulse-dot" aria-hidden="true" />
        <span>Hands-free automatic capture active — perform action naturally</span>
      </div>

      <h3 className="challenge-instruction-heading" data-testid="challenge-instruction-text">
        {challenge.instruction}
      </h3>

      {feedbackMessage && feedbackMessage !== challenge.instruction && (
        <p className="challenge-feedback-note" role="status">
          {feedbackMessage}
        </p>
      )}

      <div className="challenge-actions">
        <button
          type="button"
          className="btn btn-primary"
          disabled={isSubmitting}
          onClick={onCaptureStep}
          data-testid="verify-challenge-btn"
        >
          {isSubmitting ? "Capturing & Validating Frames..." : "Capture Now (Manual Fallback)"}
        </button>
        <button
          type="button"
          className="btn btn-secondary"
          disabled={isSubmitting}
          onClick={onAbandon}
        >
          Cancel Capture
        </button>
      </div>
    </div>
  );
};
