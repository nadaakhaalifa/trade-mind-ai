"""create environments table

Revision ID: eb9258d726b5
Revises: 68468fb451e5
Create Date: 2026-04-22 17:54:23.298087

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "eb9258d726b5"
down_revision: Union[str, Sequence[str], None] = "68468fb451e5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "environments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("asset_universe", sa.JSON(), nullable=True),
        sa.Column("action_space", sa.String(), nullable=False),
        sa.Column("fee_model", sa.JSON(), nullable=True),
        sa.Column("slippage_model", sa.JSON(), nullable=True),
        sa.Column("reward_function", sa.String(), nullable=False),
        sa.Column("risk_constraints", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_environments_id"), "environments", ["id"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_environments_id"), table_name="environments")
    op.drop_table("environments")