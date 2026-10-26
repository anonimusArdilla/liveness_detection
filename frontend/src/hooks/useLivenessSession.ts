import { useCallback, useEffect, useRef, useState } from "react";
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
  | "INITIALIZING"
  | "CHECKING_SERVICE"
  | "WAITING_FOR_FACE"
  | "READY_FOR_CHALLENGE"
  | "CHALLENGE_ACTIVE"
  | "ACTION_DETECTED"
  | "CAPTURING_EVIDENCE"
  | "CAPTURING_BATCH"
  | "SUBMITTING_EVIDENCE"
  | "CHALLENGE_VERIFIED"
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
  isAutoCaptureActive: boolean;
  initializeSession: (registrationRef?: string) => Promise<boolean>;
  captureAndSubmitCurrentChallenge: (videoElement: HTMLVideoElement | null) => Promise<void>;
  abandonSession: () => void;
}

const SAMPLE_INTERVAL_MS = Number(import.meta.env.VITE_FRAME_SAMPLE_INTERVAL_MS ?? 180);
const FRAMES_PER_BATCH = Number(import.meta.env.VITE_FRAMES_PER_CHALLENGE_BATCH ?? 3);

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function useLivenessSession(
  onFinishCleanupCamera: () => void,
  videoRef?: React.RefObject<HTMLVideoElement>,
): UseLivenessSessionResult {
  const [phase, setPhase] = useState<WorkflowPhase>("IDLE");
  const [healthStatus, setHealthStatus] = useState<HealthResponse | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [outcome, setOutcome] = useState<LivenessOutcome | null>(null);
  const [currentChallenge, setCurrentChallenge] = useState<ChallengeInstruction | null>(null);
  const [completedSteps, setCompletedSteps] = useState<number>(0);
  const [totalSteps, setTotalSteps] = useState<number>(4);
  const [qualityIssues, setQualityIssues] = useState<QualityIssue[]>([]);
  const [latestQualityReport, setLatestQualityReport] = useState<QualityReport | null>(null);
  const [feedbackMessage, setFeedbackMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [errorCode, setErrorCode] = useState<string | null>(null);
  const [finalResult, setFinalResult] = useState<SessionResultResponse | null>(null);
  const [isAutoCaptureActive, setIsAutoCaptureActive] = useState<boolean>(true);

  const sessionNonceRef = useRef<string | null>(null);
  const monotonicTimestampRef = useRef<number>(Date.now());
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const isSubmittingRef = useRef<boolean>(false);
  const batchBufferRef = useRef<FrameInput[]>([]);
  const isSessionTerminatedRef = useRef<boolean>(false);

  const nextMonotonicTimestamp = useCallback((): number => {
    const now = Date.now();
    const next = Math.max(now, monotonicTimestampRef.current + 25);
    monotonicTimestampRef.current = next;
    return next;
  }, []);

  const abandonSession = useCallback(() => {
    isSessionTerminatedRef.current = true;
    isSubmittingRef.current = false;
    batchBufferRef.current = [];
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
    setIsAutoCaptureActive(false);
    onFinishCleanupCamera();
  }, [onFinishCleanupCamera]);

  const initializeSession = useCallback(async (registrationRef?: string): Promise<boolean> => {
    isSessionTerminatedRef.current = false;
    isSubmittingRef.current = false;
    batchBufferRef.current = [];
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
      setIsAutoCaptureActive(true);
      setPhase("CHALLENGE_ACTIVE");
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

  // Authoritative batch evaluation pipeline
  const processChallengeBatch = useCallback(
    async (frames: FrameInput[], targetChallenge: ChallengeInstruction): Promise<void> => {
      if (!sessionId || !sessionNonceRef.current || isSessionTerminatedRef.current) {
        return;
      }

      isSubmittingRef.current = true;
      setPhase("SUBMITTING_EVIDENCE");
      setErrorMessage(null);
      setErrorCode(null);

      try {
        const submitResp = await livenessApiClient.submitChallengeFrames(sessionId, {
          session_nonce: sessionNonceRef.current,
          challenge_id: targetChallenge.challenge_id,
          frames,
        });

        if (isSessionTerminatedRef.current) {
          isSubmittingRef.current = false;
          return;
        }

        setOutcome(submitResp.outcome);
        setLatestQualityReport(submitResp.quality_report);
        setQualityIssues(submitResp.quality_report.issues);
        setFeedbackMessage(submitResp.feedback_message);

        // Terminal outcomes (quality exceeded, attempts exceeded, expired)
        if (
          submitResp.outcome === "QUALITY_FAILED" ||
          submitResp.outcome === "CHALLENGE_FAILED" ||
          submitResp.outcome === "VALIDATION_UNAVAILABLE" ||
          submitResp.outcome === "EXPIRED"
        ) {
          isSessionTerminatedRef.current = true;
          const outcomeDetails = await livenessApiClient.getSessionOutcome(sessionId);
          setFinalResult(outcomeDetails);
          sessionNonceRef.current = null;
          isSubmittingRef.current = false;
          batchBufferRef.current = [];
          onFinishCleanupCamera();
          setPhase("COMPLETED");
          return;
        }

        // Active challenge incomplete -> continue sampling with updated server guidance
        if (!submitResp.challenge_passed) {
          isSubmittingRef.current = false;
          setPhase("CHALLENGE_ACTIVE");
          return;
        }

        // Challenge verified by server!
        setPhase("CHALLENGE_VERIFIED");
        const nextStepCount = submitResp.step_index + 1;
        setCompletedSteps(nextStepCount);
        batchBufferRef.current = [];

        // Advance to next challenge sequentially if more challenges remain
        if (!submitResp.all_challenges_completed && submitResp.next_challenge) {
          setCurrentChallenge(submitResp.next_challenge);
          setFeedbackMessage(submitResp.next_challenge.instruction);
          if (!isSessionTerminatedRef.current) {
            isSubmittingRef.current = false;
            setPhase("CHALLENGE_ACTIVE");
          }
          return;
        }

        // All challenges verified -> finalize session and trigger passive anti-spoofing
        setCurrentChallenge(null);
        setPhase("FINALIZING");
        setFeedbackMessage("Running passive anti-spoofing verification on captured frames...");

        const finalOutcome = await livenessApiClient.completeSession(
          sessionId,
          sessionNonceRef.current,
        );
        isSessionTerminatedRef.current = true;
        sessionNonceRef.current = null;
        isSubmittingRef.current = false;
        setOutcome(finalOutcome.outcome);
        setFinalResult(finalOutcome);
        setFeedbackMessage(finalOutcome.reason);
        onFinishCleanupCamera();
        setPhase("COMPLETED");
      } catch (err: unknown) {
        if (isSessionTerminatedRef.current) {
          isSubmittingRef.current = false;
          return;
        }

        if (err instanceof LivenessApiClientError) {
          setErrorCode(err.code);
          setErrorMessage(err.message);
          if (
            err.code === "SESSION_EXPIRED" ||
            err.code === "SESSION_INVALID_STATE" ||
            err.code === "SERVICE_UNAVAILABLE"
          ) {
            isSessionTerminatedRef.current = true;
            sessionNonceRef.current = null;
            isSubmittingRef.current = false;
            onFinishCleanupCamera();
            setPhase("ERROR");
            return;
          }
        } else {
          setErrorCode("NETWORK_ERROR");
          setErrorMessage("Network error while communicating with the liveness backend.");
        }
        isSubmittingRef.current = false;
        setPhase("CHALLENGE_ACTIVE");
      }
    },
    [onFinishCleanupCamera, sessionId],
  );

  // Manual fallback trigger (compatible with existing tests and accessibility)
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
      if (isSubmittingRef.current) {
        return;
      }

      isSubmittingRef.current = true;
      setPhase("CAPTURING_BATCH");
      setErrorMessage(null);
      setErrorCode(null);

      if (!offscreenCanvasRef.current) {
        offscreenCanvasRef.current = document.createElement("canvas");
      }
      const canvas = offscreenCanvasRef.current;
      const manualFrames: FrameInput[] = [];
      let clientIssues: QualityIssue[] = [];

      for (let i = 0; i < FRAMES_PER_BATCH; i += 1) {
        const check = captureAndAssessVideoFrame(videoElement, canvas);
        if (check.issues.length > 0) {
          clientIssues = check.issues;
        }
        manualFrames.push({
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

      isSubmittingRef.current = false;
      await processChallengeBatch(manualFrames, currentChallenge);
    },
    [currentChallenge, nextMonotonicTimestamp, processChallengeBatch, sessionId],
  );

  // Continuous, automated frame sampling and batch transmission loop
  useEffect(() => {
    if (
      !isAutoCaptureActive ||
      isSessionTerminatedRef.current ||
      !sessionId ||
      !currentChallenge ||
      (phase !== "CHALLENGE_ACTIVE" && phase !== "READY_FOR_CHALLENGE") ||
      !videoRef?.current
    ) {
      return;
    }

    let isMounted = true;
    const video = videoRef.current;

    if (!offscreenCanvasRef.current) {
      offscreenCanvasRef.current = document.createElement("canvas");
    }
    const canvas = offscreenCanvasRef.current;

    const intervalId = setInterval(() => {
      if (
        !isMounted ||
        isSubmittingRef.current ||
        isSessionTerminatedRef.current ||
        !video ||
        (video.paused && !video.srcObject) ||
        video.ended
      ) {
        return;
      }

      const frameWidth = video.videoWidth || 640;
      const frameHeight = video.videoHeight || 480;
      if (frameWidth === 0 || frameHeight === 0) {
        return;
      }

      const check = captureAndAssessVideoFrame(video, canvas);
      if (check.issues.length > 0) {
        setQualityIssues(check.issues);
      } else {
        setQualityIssues([]);
      }

      // Add sampled frame to in-memory batch buffer
      batchBufferRef.current.push({
        timestamp_ms: nextMonotonicTimestamp(),
        image_base64: check.imageBase64,
      });

      // Once batch threshold is reached, dispatch for server verification
      if (batchBufferRef.current.length >= FRAMES_PER_BATCH) {
        const batch = batchBufferRef.current.splice(0, FRAMES_PER_BATCH);
        processChallengeBatch(batch, currentChallenge).catch((err) => {
          console.error("Automated batch verification error:", err);
        });
      }
    }, SAMPLE_INTERVAL_MS);

    return () => {
      isMounted = false;
      clearInterval(intervalId);
    };
  }, [
    currentChallenge,
    isAutoCaptureActive,
    nextMonotonicTimestamp,
    phase,
    processChallengeBatch,
    sessionId,
    videoRef,
  ]);

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
    isAutoCaptureActive,
    initializeSession,
    captureAndSubmitCurrentChallenge,
    abandonSession,
  };
}
