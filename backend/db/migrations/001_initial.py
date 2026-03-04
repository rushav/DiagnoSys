"""Initial migration — create problems, domains, problem_domains tables."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "problems",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("title", sa.String(1000), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("source", sa.String(50), nullable=False),
        sa.Column("source_url", sa.String(2000), nullable=False, unique=True),
        sa.Column("solution_status", sa.String(20), nullable=True),
        sa.Column("confidence_score", sa.Float, nullable=True),
        sa.Column("raw_data", JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_problems_solution_status", "problems", ["solution_status"])
    op.create_index("ix_problems_confidence_score", "problems", ["confidence_score"])
    op.create_index("ix_problems_created_at", "problems", ["created_at"])
    op.create_index("ix_problems_source", "problems", ["source"])
    op.create_index("ix_problems_source_url", "problems", ["source_url"])

    op.create_table(
        "domains",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(100), nullable=False, unique=True),
    )

    op.create_table(
        "problem_domains",
        sa.Column("problem_id", UUID(as_uuid=True), sa.ForeignKey("problems.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("domain_id", sa.Integer, sa.ForeignKey("domains.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("confidence", sa.Float, nullable=True),
    )
    op.create_index("ix_problem_domains_domain_id", "problem_domains", ["domain_id"])
    op.create_index("ix_problem_domains_problem_id", "problem_domains", ["problem_id"])


def downgrade():
    op.drop_table("problem_domains")
    op.drop_table("domains")
    op.drop_table("problems")
