import React from "react";

export interface ErrorBannerProps {
  code: string | null;
  message: string;
  onRetry: () => void;
}

export const ErrorBanner: React.FC<ErrorBannerProps> = ({ code, message, onRetry }) => {
  return (
    <div className="alert alert-error" role="alert" data-testid="error-banner">
      <div className="alert-body">
        <strong>{code ? `Error (${code}): ` : "Error: "}</strong>
        <span>{message}</span>
      </div>
      <button type="button" className="btn btn-secondary btn-sm" onClick={onRetry}>
        Start New Capture Session
      </button>
    </div>
  );
};
