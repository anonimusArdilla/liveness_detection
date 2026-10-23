"""Initial schema for capture_sessions and challenge_steps.

Revision ID: 0001_initial_schema
Revises: None
Create Date: 2026-10-22 21:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "capture_sessions",
        sa.Column("session_id", sa.String(length=64), primary_key=True),
        sa.Column("registration_ref", sa.String(length=128), nullable=True),
        sa.Column("nonce_hash", sa.String(length=64), nullable=False),
        sa.Column("outcome", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("current_step_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_steps", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("last_frame_timestamp_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("quality_passed", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("challenges_passed", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("passive_spoof_passed", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("aggregate_real_score", sa.Float(), nullable=True),
        sa.Column("min_frame_real_score", sa.Float(), nullable=True),
        sa.Column("frames_evaluated", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("passive_model_name", sa.String(length=128), nullable=True),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("quality_issues_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("photo_storage_key", sa.String(length=256), nullable=True),
        sa.Column("photo_token_hash", sa.String(length=64), nullable=True),
        sa.Column("verification_signature", sa.String(length=64), nullable=True),
    )
    op.create_index(
        "ix_capture_sessions_registration_ref", "capture_sessions", ["registration_ref"]
    )
    op.create_index("ix_capture_sessions_outcome", "capture_sessions", ["outcome"])

    op.create_table(
        "challenge_steps",
        sa.Column("challenge_id", sa.String(length=64), primary_key=True),
        sa.Column(
            "session_id",
            sa.String(length=64),
            sa.ForeignKey("capture_sessions.session_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("step_index", sa.Integer(), nullable=False),
        sa.Column("challenge_type", sa.String(length=32), nullable=False),
        sa.Column("instruction", sa.String(length=256), nullable=False),
        sa.Column("completed", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_index("ix_challenge_steps_session_id", "challenge_steps", ["session_id"])


def downgrade() -> None:
    op.drop_index("ix_challenge_steps_session_id", table_name="challenge_steps")
    op.drop_table("challenge_steps")
    op.drop_index("ix_capture_sessions_outcome", table_name="capture_sessions")
    op.drop_index("ix_capture_sessions_registration_ref", table_name="capture_sessions")
    op.drop_table("capture_sessions")
