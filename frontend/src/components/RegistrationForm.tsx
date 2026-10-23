import React from "react";
import type { RegistrationFormData } from "../types/liveness";

export interface RegistrationFormProps {
  formData: RegistrationFormData;
  disabled: boolean;
  onChange: (updated: RegistrationFormData) => void;
}

export const RegistrationForm: React.FC<RegistrationFormProps> = ({
  formData,
  disabled,
  onChange,
}) => {
  return (
    <section className="card registration-card" aria-labelledby="registration-heading">
      <h2 id="registration-heading" className="section-title">
        1. Applicant Details
      </h2>
      <p className="section-subtitle">
        Your registration details remain preserved in memory even if camera permissions or liveness
        checks need to be retried.
      </p>
      <div className="form-grid">
        <div className="form-field">
          <label htmlFor="applicant-full-name">Full Name</label>
          <input
            id="applicant-full-name"
            type="text"
            value={formData.fullName}
            disabled={disabled}
            placeholder="Jane Doe"
            onChange={(e) => onChange({ ...formData, fullName: e.target.value })}
          />
        </div>
        <div className="form-field">
          <label htmlFor="applicant-email">Email Address</label>
          <input
            id="applicant-email"
            type="email"
            value={formData.email}
            disabled={disabled}
            placeholder="jane.doe@example.com"
            onChange={(e) => onChange({ ...formData, email: e.target.value })}
          />
        </div>
        <div className="form-field">
          <label htmlFor="applicant-reg-ref">Registration Reference ID</label>
          <input
            id="applicant-reg-ref"
            type="text"
            value={formData.registrationRef}
            disabled={disabled}
            placeholder="REG-2026-001"
            pattern="^[a-zA-Z0-9_.:-]*$"
            onChange={(e) =>
              onChange({
                ...formData,
                registrationRef: e.target.value.replace(/[^a-zA-Z0-9_.:-]/g, ""),
              })
            }
          />
        </div>
      </div>
    </section>
  );
};
