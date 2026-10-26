export type LivenessOutcome =
  | "PENDING"
  | "QUALITY_FAILED"
  | "CHALLENGE_FAILED"
  | "SPOOF_SUSPECTED"
  | "VALIDATION_UNAVAILABLE"
  | "VALIDATED"
  | "EXPIRED";

export type ChallengeType =
  | "CENTER_FACE"
  | "TURN_LEFT"
  | "TURN_RIGHT"
  | "SMILE"
  | "BLINK"
  | "RAISE_EYEBROWS";

export type QualityIssueCode =
  | "RESOLUTION_TOO_LOW"
  | "RESOLUTION_TOO_HIGH"
  | "NO_FACE_DETECTED"
  | "MULTIPLE_FACES_DETECTED"
  | "FACE_TOO_SMALL"
  | "FACE_TOO_CLOSE"
  | "FACE_NOT_CENTERED"
  | "LIGHTING_TOO_DARK"
  | "LIGHTING_TOO_BRIGHT"
  | "IMAGE_TOO_BLURRY";

export interface QualityIssue {
  code: QualityIssueCode;
  message: string;
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface HeadPose {
  yaw: number;
  pitch: number;
  roll: number;
}

export interface QualityReport {
  passed: boolean;
  width: number;
  height: number;
  brightness: number;
  blur_score: number;
  face_count: number;
  face_area_ratio: number | null;
  center_offset_ratio: number | null;
  bounding_box: BoundingBox | null;
  head_pose: HeadPose | null;
  issues: QualityIssue[];
}

export interface ModelStatus {
  name: string;
  ready: boolean;
  provider: string;
  path: string | null;
  checksum_verified: boolean;
  reason: string | null;
}

export interface HealthResponse {
  status: string;
  ready: boolean;
  version: string;
  models: ModelStatus[];
  database_ready: boolean;
  storage_ready: boolean;
}

export interface ChallengeInstruction {
  challenge_id: string;
  step_index: number;
  total_steps: number;
  challenge_type: ChallengeType;
  instruction: string;
  completed: boolean;
}

export interface CreateSessionResponse {
  session_id: string;
  session_nonce: string;
  outcome: LivenessOutcome;
  expires_at: string;
  total_challenges: number;
  current_challenge: ChallengeInstruction;
}

export interface CurrentChallengeResponse {
  session_id: string;
  outcome: LivenessOutcome;
  expires_at: string;
  all_challenges_completed: boolean;
  current_challenge: ChallengeInstruction | null;
}

export interface FrameInput {
  timestamp_ms: number;
  image_base64: string;
}

export interface SubmitChallengeFramesRequest {
  session_nonce: string;
  challenge_id: string;
  frames: FrameInput[];
}

export interface SubmitChallengeFramesResponse {
  session_id: string;
  outcome: LivenessOutcome;
  challenge_passed: boolean;
  step_index: number;
  total_steps: number;
  feedback_message: string;
  quality_report: QualityReport;
  all_challenges_completed: boolean;
  next_challenge: ChallengeInstruction | null;
}

export interface PassiveSpoofReport {
  passed: boolean;
  aggregate_real_score: number;
  min_frame_real_score: number;
  frames_evaluated: number;
  threshold_applied: number;
  model_name: string;
}

export interface SessionResultResponse {
  session_id: string;
  registration_ref: string | null;
  outcome: LivenessOutcome;
  validated: boolean;
  created_at: string;
  expires_at: string;
  completed_at: string | null;
  reason: string;
  quality_passed: boolean;
  challenges_passed: boolean;
  passive_spoof_passed: boolean;
  passive_report: PassiveSpoofReport | null;
  quality_issues: QualityIssue[];
  photo_available: boolean;
  photo_access_token: string | null;
  verification_signature: string | null;
}

export interface ApiErrorDetail {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

export interface RegistrationFormData {
  fullName: string;
  email: string;
  registrationRef: string;
}
