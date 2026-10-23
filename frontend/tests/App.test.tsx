import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "../src/app/App";
import { livenessApiClient } from "../src/services/livenessApi";
import type {
  CreateSessionResponse,
  HealthResponse,
  SessionResultResponse,
  SubmitChallengeFramesResponse,
} from "../src/types/liveness";

const mockTrackStop = vi.fn();

function createMockMediaStream(): MediaStream {
  return {
    getTracks: () => [{ stop: mockTrackStop } as unknown as MediaStreamTrack],
  } as unknown as MediaStream;
}

const READY_HEALTH: HealthResponse = {
  status: "ready",
  ready: true,
  version: "1.0.0",
  models: [
    {
      name: "MediaPipe-FaceLandmarker",
      ready: true,
      provider: "CPUExecutionProvider",
      path: "face_landmarker.task",
      checksum_verified: true,
      reason: null,
    },
    {
      name: "MiniFASNetV2 (scale 2.7)",
      ready: true,
      provider: "CPUExecutionProvider",
      path: "MiniFASNetV2.onnx",
      checksum_verified: true,
      reason: null,
    },
  ],
  database_ready: true,
  storage_ready: true,
};

const CREATED_SESSION: CreateSessionResponse = {
  session_id: "sess-11112222-3333",
  session_nonce: "nonce-secret-1234567890123456",
  outcome: "PENDING",
  expires_at: "2026-10-22T22:00:00Z",
  total_challenges: 2,
  current_challenge: {
    challenge_id: "chal-step-0",
    step_index: 0,
    total_steps: 2,
    challenge_type: "CENTER_FACE",
    instruction: "Keep your face centered and look straight at the camera.",
    completed: false,
  },
};

beforeEach(() => {
  mockTrackStop.mockClear();
  HTMLMediaElement.prototype.play = vi.fn().mockResolvedValue(undefined);
  HTMLCanvasElement.prototype.getContext = vi.fn().mockReturnValue({
    drawImage: vi.fn(),
    getImageData: () => ({
      data: new Uint8ClampedArray(640 * 480 * 4).fill(130),
    }),
    clearRect: vi.fn(),
  });
  HTMLCanvasElement.prototype.toDataURL = vi
    .fn()
    .mockReturnValue("data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/");
});

describe("Facial Liveness Verification Frontend", () => {
  it("handles camera permission denial and preserves registration form data", async () => {
    const user = userEvent.setup();
    const deniedError = new Error("Permission denied");
    deniedError.name = "NotAllowedError";

    Object.defineProperty(navigator, "mediaDevices", {
      writable: true,
      configurable: true,
      value: {
        getUserMedia: vi.fn().mockRejectedValue(deniedError),
      },
    });

    render(<App />);

    const nameInput = screen.getByLabelText(/Full Name/i) as HTMLInputElement;
    const emailInput = screen.getByLabelText(/Email Address/i) as HTMLInputElement;
    await user.type(nameInput, "Alice Security");
    await user.type(emailInput, "alice@example.com");

    const startBtn = screen.getByRole("button", {
      name: /Enable Camera & Start Liveness Check/i,
    });
    await user.click(startBtn);

    await waitFor(() => {
      expect(screen.getByText(/Camera Permission Denied:/i)).toBeInTheDocument();
    });

    // Registration data must remain intact after camera error
    expect(nameInput.value).toBe("Alice Security");
    expect(emailInput.value).toBe("alice@example.com");
  });

  it("handles missing camera device (NotFoundError) and camera initialization failure", async () => {
    const user = userEvent.setup();
    const notFoundErr = new Error("Requested device not found");
    notFoundErr.name = "NotFoundError";

    const getUserMediaMock = vi
      .fn()
      .mockRejectedValueOnce(notFoundErr)
      .mockRejectedValueOnce(new Error("Hardware bus timeout"));

    Object.defineProperty(navigator, "mediaDevices", {
      writable: true,
      configurable: true,
      value: { getUserMedia: getUserMediaMock },
    });

    render(<App />);

    await user.click(
      screen.getByRole("button", { name: /Enable Camera & Start Liveness Check/i }),
    );
    await waitFor(() => {
      expect(screen.getByText(/No Camera Found:/i)).toBeInTheDocument();
    });

    // Retry and trigger generic initialization error
    await user.click(screen.getByRole("button", { name: /Retry Camera & Start Session/i }));
    await waitFor(() => {
      expect(screen.getByText(/Camera Initialization Error:/i)).toBeInTheDocument();
      expect(screen.getByText(/Hardware bus timeout/i)).toBeInTheDocument();
    });
  });

  it("handles backend service-unavailable state when required models are missing", async () => {
    const user = userEvent.setup();
    Object.defineProperty(navigator, "mediaDevices", {
      writable: true,
      configurable: true,
      value: {
        getUserMedia: vi.fn().mockResolvedValue(createMockMediaStream()),
      },
    });

    vi.spyOn(livenessApiClient, "checkReadiness").mockResolvedValue({
      ...READY_HEALTH,
      ready: false,
      models: [
        {
          name: "MiniFASNetV2 (scale 2.7)",
          ready: false,
          provider: "CPUExecutionProvider",
          path: "MiniFASNetV2.onnx",
          checksum_verified: false,
          reason: "Model weight file not found",
        },
      ],
    });

    render(<App />);
    await user.click(
      screen.getByRole("button", { name: /Enable Camera & Start Liveness Check/i }),
    );

    await waitFor(() => {
      expect(screen.getByTestId("error-banner")).toHaveTextContent(
        /required models are not ready/i,
      );
    });
    expect(mockTrackStop).toHaveBeenCalled();
  });

  it("completes sequential challenges and renders server-confirmed VALIDATED result", async () => {
    const user = userEvent.setup();
    Object.defineProperty(navigator, "mediaDevices", {
      writable: true,
      configurable: true,
      value: {
        getUserMedia: vi.fn().mockResolvedValue(createMockMediaStream()),
      },
    });

    vi.spyOn(livenessApiClient, "checkReadiness").mockResolvedValue(READY_HEALTH);
    vi.spyOn(livenessApiClient, "createSession").mockResolvedValue(CREATED_SESSION);

    const step1Response: SubmitChallengeFramesResponse = {
      session_id: CREATED_SESSION.session_id,
      outcome: "PENDING",
      challenge_passed: true,
      step_index: 0,
      total_steps: 2,
      feedback_message: "Challenge 'CENTER_FACE' verified.",
      quality_report: {
        passed: true,
        width: 640,
        height: 480,
        brightness: 130,
        blur_score: 120,
        face_count: 1,
        face_area_ratio: 0.18,
        center_offset_ratio: 0.02,
        bounding_box: { x: 200, y: 120, width: 220, height: 220 },
        head_pose: { yaw: 1.0, pitch: 0.5, roll: 0.0 },
        issues: [],
      },
      all_challenges_completed: false,
      next_challenge: {
        challenge_id: "chal-step-1",
        step_index: 1,
        total_steps: 2,
        challenge_type: "TURN_LEFT",
        instruction: "Slowly turn your head to your left.",
        completed: false,
      },
    };

    const step2Response: SubmitChallengeFramesResponse = {
      ...step1Response,
      step_index: 1,
      feedback_message: "Challenge 'TURN_LEFT' verified.",
      all_challenges_completed: true,
      next_challenge: null,
    };

    const submitSpy = vi
      .spyOn(livenessApiClient, "submitChallengeFrames")
      .mockResolvedValueOnce(step1Response)
      .mockResolvedValueOnce(step2Response);

    const validatedResult: SessionResultResponse = {
      session_id: CREATED_SESSION.session_id,
      registration_ref: "REG-2026-001",
      outcome: "VALIDATED",
      validated: true,
      created_at: "2026-10-22T21:00:00Z",
      expires_at: "2026-10-22T21:03:00Z",
      completed_at: "2026-10-22T21:00:25Z",
      reason: "Facial liveness validated: image quality, active challenges, and passive PAD passed.",
      quality_passed: true,
      challenges_passed: true,
      passive_spoof_passed: true,
      passive_report: {
        passed: true,
        aggregate_real_score: 0.96,
        min_frame_real_score: 0.93,
        frames_evaluated: 2,
        threshold_applied: 0.65,
        model_name: "MiniFASNetV2+MiniFASNetV1SE",
      },
      quality_issues: [],
      photo_available: true,
      photo_access_token: "tok-accepted-photo-123",
      verification_signature: "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
    };

    vi.spyOn(livenessApiClient, "completeSession").mockResolvedValue(validatedResult);

    render(<App />);

    await user.click(
      screen.getByRole("button", { name: /Enable Camera & Start Liveness Check/i }),
    );

    // Step 1 instruction displayed
    await waitFor(() => {
      expect(screen.getByTestId("challenge-instruction-text")).toHaveTextContent(
        /Keep your face centered/i,
      );
    });

    await user.click(screen.getByTestId("verify-challenge-btn"));

    // Step 2 instruction displayed progressively after backend confirms step 1
    await waitFor(() => {
      expect(screen.getByTestId("challenge-instruction-text")).toHaveTextContent(
        /Slowly turn your head to your left/i,
      );
    });

    await user.click(screen.getByTestId("verify-challenge-btn"));

    // Final VALIDATED card displayed
    await waitFor(() => {
      expect(screen.getByTestId("validation-result-card")).toHaveTextContent("VALIDATED");
      expect(
        screen.getByText(/Physical Liveness Confirmed by Server/i),
      ).toBeInTheDocument();
    });
    expect(submitSpy).toHaveBeenCalledTimes(2);
    expect(mockTrackStop).toHaveBeenCalled();
  });

  it("distinguishes SPOOF_SUSPECTED rejection from quality failure and allows retry", async () => {
    const user = userEvent.setup();
    Object.defineProperty(navigator, "mediaDevices", {
      writable: true,
      configurable: true,
      value: {
        getUserMedia: vi.fn().mockResolvedValue(createMockMediaStream()),
      },
    });

    vi.spyOn(livenessApiClient, "checkReadiness").mockResolvedValue(READY_HEALTH);
    vi.spyOn(livenessApiClient, "createSession").mockResolvedValue({
      ...CREATED_SESSION,
      total_challenges: 1,
      current_challenge: {
        ...CREATED_SESSION.current_challenge,
        total_steps: 1,
      },
    });

    vi.spyOn(livenessApiClient, "submitChallengeFrames").mockResolvedValue({
      session_id: CREATED_SESSION.session_id,
      outcome: "PENDING",
      challenge_passed: true,
      step_index: 0,
      total_steps: 1,
      feedback_message: "Challenge 'CENTER_FACE' verified.",
      quality_report: {
        passed: true,
        width: 640,
        height: 480,
        brightness: 125,
        blur_score: 95,
        face_count: 1,
        face_area_ratio: 0.2,
        center_offset_ratio: 0.03,
        bounding_box: { x: 200, y: 120, width: 220, height: 220 },
        head_pose: { yaw: 0, pitch: 0, roll: 0 },
        issues: [],
      },
      all_challenges_completed: true,
      next_challenge: null,
    });

    vi.spyOn(livenessApiClient, "completeSession").mockResolvedValue({
      session_id: CREATED_SESSION.session_id,
      registration_ref: "REG-2026-001",
      outcome: "SPOOF_SUSPECTED",
      validated: false,
      created_at: "2026-10-22T21:00:00Z",
      expires_at: "2026-10-22T21:03:00Z",
      completed_at: "2026-10-22T21:00:20Z",
      reason:
        "Passive anti-spoofing check failed: potential presentation attack (printed photo or screen replay) detected.",
      quality_passed: true,
      challenges_passed: true,
      passive_spoof_passed: false,
      passive_report: {
        passed: false,
        aggregate_real_score: 0.08,
        min_frame_real_score: 0.05,
        frames_evaluated: 1,
        threshold_applied: 0.65,
        model_name: "MiniFASNetV2+MiniFASNetV1SE",
      },
      quality_issues: [],
      photo_available: false,
      photo_access_token: null,
      verification_signature: "deadbeef",
    });

    render(<App />);
    const nameInput = screen.getByLabelText(/Full Name/i) as HTMLInputElement;
    await user.type(nameInput, "Bob Spoofer");

    await user.click(
      screen.getByRole("button", { name: /Enable Camera & Start Liveness Check/i }),
    );
    await waitFor(() => {
      expect(screen.getByTestId("verify-challenge-btn")).toBeInTheDocument();
    });

    await user.click(screen.getByTestId("verify-challenge-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("validation-result-card")).toHaveTextContent(
        "SPOOF_SUSPECTED",
      );
      expect(
        screen.getByText(/Presentation Attack \(Spoof\) Suspected/i),
      ).toBeInTheDocument();
    });

    // Click Retry and confirm registration form input is still preserved
    await user.click(screen.getByRole("button", { name: /Retry Liveness Capture/i }));
    expect(nameInput.value).toBe("Bob Spoofer");
    expect(
      screen.getByRole("button", { name: /Enable Camera & Start Liveness Check/i }),
    ).toBeInTheDocument();
  });
});
