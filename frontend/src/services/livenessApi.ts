import type {
  ApiErrorDetail,
  CreateSessionResponse,
  CurrentChallengeResponse,
  HealthResponse,
  SessionResultResponse,
  SubmitChallengeFramesRequest,
  SubmitChallengeFramesResponse,
} from "../types/liveness";

const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000/api/v1";

export function getApiBaseUrl(): string {
  const configured = import.meta.env.VITE_API_BASE_URL as string | undefined;
  return (configured && configured.trim() ? configured.trim() : DEFAULT_API_BASE_URL).replace(
    /\/+$/,
    "",
  );
}

export class LivenessApiClientError extends Error {
  public readonly code: string;
  public readonly status: number;
  public readonly details: Record<string, unknown>;

  constructor(code: string, message: string, status: number, details: Record<string, unknown> = {}) {
    super(message);
    this.name = "LivenessApiClientError";
    this.code = code;
    this.status = status;
    this.details = details;
  }
}

async function parseJsonResponse<T>(response: Response): Promise<T> {
  let payload: unknown = null;
  try {
    payload = await response.json();
  } catch {
    throw new LivenessApiClientError(
      "NETWORK_PARSE_ERROR",
      `Unexpected non-JSON response from server (HTTP ${response.status}).`,
      response.status,
    );
  }

  if (!response.ok) {
    const errEnvelope = payload as { error?: ApiErrorDetail } | null;
    const err = errEnvelope?.error;
    throw new LivenessApiClientError(
      err?.code ?? `HTTP_${response.status}`,
      err?.message ?? `Request failed with HTTP status ${response.status}.`,
      response.status,
      err?.details ?? {},
    );
  }

  return payload as T;
}

export const livenessApiClient = {
  async checkReadiness(): Promise<HealthResponse> {
    const response = await fetch(`${getApiBaseUrl()}/health/ready`, {
      method: "GET",
      headers: { Accept: "application/json" },
    });
    // Both 200 and 503 return structured HealthResponse JSON
    const data = (await response.json()) as HealthResponse;
    return data;
  },

  async createSession(registrationRef?: string): Promise<CreateSessionResponse> {
    const response = await fetch(`${getApiBaseUrl()}/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({
        registration_ref: registrationRef && registrationRef.trim() ? registrationRef.trim() : null,
      }),
    });
    return parseJsonResponse<CreateSessionResponse>(response);
  },

  async getCurrentChallenge(sessionId: string): Promise<CurrentChallengeResponse> {
    const response = await fetch(
      `${getApiBaseUrl()}/sessions/${encodeURIComponent(sessionId)}/challenge`,
      {
        method: "GET",
        headers: { Accept: "application/json" },
      },
    );
    return parseJsonResponse<CurrentChallengeResponse>(response);
  },

  async submitChallengeFrames(
    sessionId: string,
    payload: SubmitChallengeFramesRequest,
  ): Promise<SubmitChallengeFramesResponse> {
    const response = await fetch(
      `${getApiBaseUrl()}/sessions/${encodeURIComponent(sessionId)}/challenge/frames`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify(payload),
      },
    );
    return parseJsonResponse<SubmitChallengeFramesResponse>(response);
  },

  async completeSession(
    sessionId: string,
    sessionNonce: string,
  ): Promise<SessionResultResponse> {
    const response = await fetch(
      `${getApiBaseUrl()}/sessions/${encodeURIComponent(sessionId)}/complete`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ session_nonce: sessionNonce }),
      },
    );
    return parseJsonResponse<SessionResultResponse>(response);
  },

  async getSessionOutcome(sessionId: string): Promise<SessionResultResponse> {
    const response = await fetch(
      `${getApiBaseUrl()}/sessions/${encodeURIComponent(sessionId)}`,
      {
        method: "GET",
        headers: { Accept: "application/json" },
      },
    );
    return parseJsonResponse<SessionResultResponse>(response);
  },

  getProtectedPhotoUrl(sessionId: string, photoAccessToken: string): string {
    return `${getApiBaseUrl()}/sessions/${encodeURIComponent(sessionId)}/photo?token=${encodeURIComponent(photoAccessToken)}`;
  },
};
