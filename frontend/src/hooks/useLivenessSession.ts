import { useCallback, useRef, useState } from "react";
import { LivenessApiClientError, livenessApiClient } from "../services/livenessApi";
import type {
  ChallengeInstruction,
  FrameInput,
  HealthResponse,
  LivenessOutcome,
  QualityIssue,
  QualityReport,
  SessionResultResponse,
} from "../types/liveness";
import { captureAndAssessVideoFrame } from "../utils/frameQuality";

export type WorkflowPhase =
  | "IDLE"
  | "CHECKING_SERVICE"
  | "READY_FOR_CHALLENGE"
  | "CAPTURING_BATCH"
  | "FINALIZING"
  | "COMPLETED"
  | "SERVICE_UNAVAILABLE"
  | "ERROR";

export interface UseLivenessSessionResult {
  phase: WorkflowPhase;
  healthStatus: HealthResponse | null;
  sessionId: string | null;
  outcome: LivenessOutcome | null;
  currentChallenge: ChallengeInstruction | null;
  completedSteps: number;
  totalSteps: number;
  qualityIssues: QualityIssue[];
  latestQualityReport: QualityReport | null;
  feedbackMessage: string | null;
  errorMessage: string | null;
  errorCode: string | null;
  finalResult: SessionResultResponse | null;
  initializeSession: (registrationRef?: string) => Promise<boolean>;
  captureAndSubmitCurrentChallenge: (videoElement: HTMLVideoElement | null) => Promise<void>;
  abandonSession: () => void;
}

const SAMPLE_INTERVAL_MS = Number(import.meta.env.VITE_FRAME_SAMPLE_INTERVAL_MS ?? 180);
const FRAMES_PER_BATCH = Number(import.meta.env.VITE_FRAMES_PER_CHALLENGE_BATCH ?? 3);

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function useLivenessSession(onFinishCleanupCamera: () => void): UseLivenessSessionResult {
  const [phase, setPhase] = useState<WorkflowPhase>("IDLE");
  const [healthStatus, setHealthStatus] = useState<HealthResponse | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [outcome, setOutcome] = useState<LivenessOutcome | null>(null);
  const [currentChallenge, setCurrentChallenge] = useState<ChallengeInstruction | null>(null);
  const [completedSteps, setCompletedSteps] = useState<number>(0);
  const [totalSteps, setTotalSteps] = useState<number>(3);
  const [qualityIssues, setQualityIssues] = useState<QualityIssue[]>([]);
  const [latestQualityReport, setLatestQualityReport] = useState<QualityReport | null>(null);
  const [feedbackMessage, setFeedbackMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [errorCode, setErrorCode] = useState<string | null>(null);
  const [finalResult, setFinalResult] = useState<SessionResultResponse | null>(null);

  const sessionNonceRef = useRef<string | null>(null);
  const monotonicTimestampRef = useRef<number>(Date.now());
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const nextMonotonicTimestamp = useCallback((): number => {
    const now = Date.now();
    const next = Math.max(now, monotonicTimestampRef.current + 25);
    monotonicTimestampRef.current = next;
    return next;
  }, []);

  const abandonSession = useCallback(() => {
    sessionNonceRef.current = null;
    setPhase("IDLE");
    setSessionId(null);
    setOutcome(null);
    setCurrentChallenge(null);
    setCompletedSteps(0);
    setQualityIssues([]);
    setLatestQualityReport(null);
    setFeedbackMessage(null);
    setErrorMessage(null);
    setErrorCode(null);
    setFinalResult(null);
    onFinishCleanupCamera();
  }, [onFinishCleanupCamera]);

  const initializeSession = useCallback(async (registrationRef?: string): Promise<boolean> => {
    setPhase("CHECKING_SERVICE");
    setErrorMessage(null);
    setErrorCode(null);
    setQualityIssues([]);
    setLatestQualityReport(null);
    setFinalResult(null);
    setFeedbackMessage("Verifying backend model readiness...");

    try {
      const readiness = await livenessApiClient.checkReadiness();
      setHealthStatus(readiness);
      if (!readiness.ready) {
        const unreadyReasons = readiness.models
          .filter((m) => !m.ready)
          .map((m) => `${m.name}: ${m.reason ?? "not ready"}`)
          .join("; ");
        setPhase("SERVICE_UNAVAILABLE");
        setErrorCode("SERVICE_UNAVAILABLE");
        setErrorMessage(
          unreadyReasons
            ? `Liveness verification is unavailable because required models are not ready (${unreadyReasons}).`
            : "Liveness verification service is currently not ready.",
        );
        return false;
      }

      const created = await livenessApiClient.createSession(registrationRef);
      sessionNonceRef.current = created.session_nonce;
      setSessionId(created.session_id);
      setOutcome(created.outcome);
      setCurrentChallenge(created.current_challenge);
      setCompletedSteps(0);
      setTotalSteps(created.total_challenges);
      setFeedbackMessage(created.current_challenge.instruction);
      setPhase("READY_FOR_CHALLENGE");
      return true;
    } catch (err: unknown) {
      if (err instanceof LivenessApiClientError) {
        setPhase(err.status === 503 ? "SERVICE_UNAVAILABLE" : "ERROR");
        setErrorCode(err.code);
        setErrorMessage(err.message);
      } else {
        setPhase("SERVICE_UNAVAILABLE");
        setErrorCode("BACKEND_UNREACHABLE");
        setErrorMessage(
          "Could not reach the liveness verification backend. Ensure the FastAPI server is running.",
        );
      }
      return false;
    }
  }, []);

  const captureAndSubmitCurrentChallenge = useCallback(
    async (videoElement: HTMLVideoElement | null): Promise<void> => {
      if (!sessionId || !sessionNonceRef.current || !currentChallenge) {
        setErrorMessage("No active capture session. Please start a new session.");
        setPhase("ERROR");
        return;
      }
      if (!videoElement) {
        setErrorMessage("Camera video stream is not active.");
        return;
      }

      setPhase("CAPTURING_BATCH");
      setErrorMessage(null);
      setErrorCode(null);

      if (!offscreenCanvasRef.current) {
        offscreenCanvasRef.current = document.createElement("canvas");
      }
      const canvas = offscreenCanvasRef.current;
      const frames: FrameInput[] = [];
      let clientIssues: QualityIssue[] = [];

      for (let i = 0; i < FRAMES_PER_BATCH; i += 1) {
        const check = captureAndAssessVideoFrame(videoElement, canvas);
        if (check.issues.length > 0) {
          clientIssues = check.issues;
        }
        frames.push({
          timestamp_ms: nextMonotonicTimestamp(),
          image_base64: check.imageBase64,
        });
        if (i < FRAMES_PER_BATCH - 1) {
          await sleep(SAMPLE_INTERVAL_MS);
        }
      }

      if (clientIssues.length > 0) {
        setQualityIssues(clientIssues);
      }

      try {
        const submitResp = await livenessApiClient.submitChallengeFrames(sessionId, {
          session_nonce: sessionNonceRef.current,
          challenge_id: currentChallenge.challenge_id,
          frames,
        });

        setOutcome(submitResp.outcome);
        setLatestQualityReport(submitResp.quality_report);
        setQualityIssues(submitResp.quality_report.issues);
        setFeedbackMessage(submitResp.feedback_message);

        // If the session transitioned to a terminal failure state during challenge evaluation
        if (
          submitResp.outcome === "QUALITY_FAILED" ||
          submitResp.outcome === "CHALLENGE_FAILED" ||
          submitResp.outcome === "VALIDATION_UNAVAILABLE" ||
          submitResp.outcome === "EXPIRED"
        ) {
          const outcomeDetails = await livenessApiClient.getSessionOutcome(sessionId);
          setFinalResult(outcomeDetails);
          sessionNonceRef.current = null;
          onFinishCleanupCamera();
          setPhase("COMPLETED");
          return;
        }

        if (!submitResp.challenge_passed) {
          setPhase("READY_FOR_CHALLENGE");
          return;
        }

        const nextStepCount = submitResp.step_index + 1;
        setCompletedSteps(nextStepCount);

        if (!submitResp.all_challenges_completed && submitResp.next_challenge) {
          setCurrentChallenge(submitResp.next_challenge);
          setFeedbackMessage(submitResp.next_challenge.instruction);
          setPhase("READY_FOR_CHALLENGE");
          return;
        }

        // All active challenges passed -> request final passive anti-spoofing decision from backend
        setCurrentChallenge(null);
        setPhase("FINALIZING");
        setFeedbackMessage("Running passive anti-spoofing verification on captured frames...");

        const finalOutcome = await livenessApiClient.completeSession(
          sessionId,
          sessionNonceRef.current,
        );
        sessionNonceRef.current = null;
        setOutcome(finalOutcome.outcome);
        setFinalResult(finalOutcome);
        setFeedbackMessage(finalOutcome.reason);
        onFinishCleanupCamera();
        setPhase("COMPLETED");
      } catch (err: unknown) {
        if (err instanceof LivenessApiClientError) {
          setErrorCode(err.code);
          setErrorMessage(err.message);
          if (
            err.code === "SESSION_EXPIRED" ||
            err.code === "SESSION_INVALID_STATE" ||
            err.code === "SERVICE_UNAVAILABLE"
          ) {
            sessionNonceRef.current = null;
            onFinishCleanupCamera();
            setPhase("ERROR");
            return;
          }
        } else {
          setErrorCode("NETWORK_ERROR");
          setErrorMessage("Network error while communicating with the liveness backend.");
        }
        setPhase("READY_FOR_CHALLENGE");
      }
    },
    [currentChallenge, nextMonotonicTimestamp, onFinishCleanupCamera, sessionId],
  );

  return {
    phase,
    healthStatus,
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
  };
}
