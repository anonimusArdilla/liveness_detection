import React from "react";
import type { WorkflowPhase } from "../hooks/useLivenessSession";
import type { LivenessOutcome } from "../types/liveness";

export interface CaptureProgressStatusProps {
  phase: WorkflowPhase;
  outcome: LivenessOutcome | null;
  completedSteps: number;
  totalSteps: number;
  sessionId: string | null;
}

export const CaptureProgressStatus: React.FC<CaptureProgressStatusProps> = ({
  phase,
  outcome,
  completedSteps,
  totalSteps,
  sessionId,
}) => {
  const progressPercent =
    totalSteps > 0 ? Math.min(100, Math.round((completedSteps / totalSteps) * 100)) : 0;

  return (
    <div className="progress-status-bar" role="status" aria-live="polite">
      <div className="progress-meta-row">
        <span>
          <strong>Workflow Status:</strong> {phase.replace(/_/g, " ")}
          {outcome ? ` (${outcome})` : ""}
        </span>
        <span>
          <strong>Completed Challenges:</strong> {completedSteps} / {totalSteps}
        </span>
        {sessionId && (
          <span className="session-id-pill" title={sessionId}>
            Session: {sessionId.slice(0, 8)}...
          </span>
        )}
      </div>
      <div
        className="progress-track"
        role="progressbar"
        aria-valuenow={completedSteps}
        aria-valuemin={0}
        aria-valuemax={totalSteps}
        aria-label="Active challenge completion progress"
      >
        <div className="progress-fill" style={{ width: `${progressPercent}%` }} />
      </div>
    </div>
  );
};
