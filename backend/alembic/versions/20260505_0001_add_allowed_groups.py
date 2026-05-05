"""add allowed_groups

Revision ID: 20260505_0001
Revises: 20260502_0001
Create Date: 2026-05-05
"""

from alembic import op
import sqlalchemy as sa

revision = "20260505_0001"
down_revision = "20260502_0001"
branch_labels = None
depends_on = None

def upgrade() -> None:
    # Use server_default='[]' to ensure existing rows get an empty array rather than null
    op.add_column("documents", sa.Column("allowed_groups", sa.JSON(), server_default='[]', nullable=False))

def downgrade() -> None:
    op.drop_column("documents", "allowed_groups")
