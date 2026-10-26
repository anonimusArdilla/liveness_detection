import React, { useCallback, useState } from "react";
import { CameraPermissionPrompt } from "../components/CameraPermissionPrompt";
import { CameraPreview } from "../components/CameraPreview";
import { CaptureProgressStatus } from "../components/CaptureProgressStatus";
import { ChallengeInstructions } from "../components/ChallengeInstructions";
import { ErrorBanner } from "../components/ErrorBanner";
import { QualityGuidanceBanner } from "../components/QualityGuidanceBanner";
import { RegistrationForm } from "../components/RegistrationForm";
import { ValidationResultCard } from "../components/ValidationResultCard";
import { useCamera } from "../hooks/useCamera";
import { useLivenessSession } from "../hooks/useLivenessSession";
import type { RegistrationFormData } from "../types/liveness";

export const App: React.FC = () => {
  const [formData, setFormData] = useState<RegistrationFormData>({
    fullName: "",
    email: "",
    registrationRef: "REG-2026-001",
  });

  const { cameraState, cameraError, videoRef, startCamera, stopCamera } = useCamera();
  const {
    phase,
    sessionId,
    outcome,
    currentChallenge,
    completedSteps,
    totalSteps,
    qualityIssues,
    latestQualityReport,
    feedbackMessage,
    errorMessage,
    errorCode,
    finalResult,
    initializeSession,
    captureAndSubmitCurrentChallenge,
    abandonSession,
  } = useLivenessSession(stopCamera, videoRef);

  const handleStartCaptureFlow = useCallback(async () => {
    const camOk = await startCamera();
    if (!camOk) {
      return;
    }
    const sessionOk = await initializeSession(formData.registrationRef);
    if (!sessionOk) {
      stopCamera();
    }
  }, [formData.registrationRef, initializeSession, startCamera, stopCamera]);

  const handleCaptureStep = useCallback(async () => {
    await captureAndSubmitCurrentChallenge(videoRef.current);
  }, [captureAndSubmitCurrentChallenge, videoRef]);

  const handleRetryOrReset = useCallback(() => {
    abandonSession();
  }, [abandonSession]);

  const isSessionActive =
    phase === "CHECKING_SERVICE" ||
    phase === "READY_FOR_CHALLENGE" ||
    phase === "CHALLENGE_ACTIVE" ||
    phase === "ACTION_DETECTED" ||
    phase === "CAPTURING_EVIDENCE" ||
    phase === "CAPTURING_BATCH" ||
    phase === "SUBMITTING_EVIDENCE" ||
    phase === "CHALLENGE_VERIFIED";

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="header-inner">
          <div>
            <h1 className="app-title">Facial Liveness Verification</h1>
            <p className="app-subtitle">
              Local CPU AI Anti-Spoofing (MiniFASNet ONNX) &amp; Server-Validated 3D Landmark
              Challenges
            </p>
          </div>
          <span className="privacy-pill">Zero Raw Video Retention • In-Memory Processing</span>
        </div>
      </header>

      <main className="app-main">
        <RegistrationForm
          formData={formData}
          disabled={isSessionActive || phase === "FINALIZING"}
          onChange={setFormData}
        />

        <section className="card capture-card" aria-labelledby="liveness-capture-heading">
          <h2 id="liveness-capture-heading" className="section-title">
            2. Live Presence &amp; Anti-Spoofing Verification
          </h2>

          <CaptureProgressStatus
            phase={phase}
            outcome={outcome}
            completedSteps={completedSteps}
            totalSteps={totalSteps}
            sessionId={sessionId}
          />

          {errorMessage && (
            <ErrorBanner
              code={errorCode}
              message={errorMessage}
              onRetry={handleRetryOrReset}
            />
          )}

          {phase === "COMPLETED" && finalResult ? (
            <ValidationResultCard result={finalResult} onRetry={handleRetryOrReset} />
          ) : (
            <div className="capture-workspace">
              <div className="camera-column">
                <CameraPreview
                  videoRef={videoRef}
                  isActive={cameraState === "ACTIVE"}
                  currentChallengeType={currentChallenge?.challenge_type ?? null}
                  qualityPassed={Boolean(latestQualityReport?.passed)}
                />
                {cameraState === "ACTIVE" && (
                  <QualityGuidanceBanner
                    issues={qualityIssues}
                    qualityReport={latestQualityReport}
                  />
                )}
              </div>

              <div className="controls-column">
                {phase === "IDLE" ||
                phase === "SERVICE_UNAVAILABLE" ||
                phase === "ERROR" ||
                cameraState !== "ACTIVE" ? (
                  <CameraPermissionPrompt
                    cameraState={cameraState}
                    cameraError={cameraError}
                    isStarting={phase === "CHECKING_SERVICE" || cameraState === "REQUESTING"}
                    onStartCapture={handleStartCaptureFlow}
                  />
                ) : null}

                {cameraState === "ACTIVE" && currentChallenge && (
                  <ChallengeInstructions
                    challenge={currentChallenge}
                    feedbackMessage={feedbackMessage}
                    isSubmitting={
                      phase === "SUBMITTING_EVIDENCE" ||
                      phase === "CAPTURING_BATCH" ||
                      phase === "FINALIZING"
                    }
                    onCaptureStep={handleCaptureStep}
                    onAbandon={handleRetryOrReset}
                  />
                )}

                {phase === "FINALIZING" && (
                  <div className="finalizing-box" role="status" aria-live="polite">
                    <strong>Evaluating Passive Anti-Spoofing Model...</strong>
                    <p>
                      Verifying multi-scale MiniFASNet ONNX predictions and session integrity on the
                      backend.
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
};
