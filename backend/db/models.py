"""
models.py — DiagnoSys Database Models
SQLAlchemy models with UUID PKs, JSONB, and proper indexes.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Column, DateTime, Float, ForeignKey, Index, Integer,
    String, Text, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class Problem(Base):
    __tablename__ = "problems"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(1000), nullable=False)
    description = Column(Text, nullable=True)
    source = Column(String(50), nullable=False)          # stack_exchange | github | reddit
    source_url = Column(String(2000), nullable=False, unique=True)
    solution_status = Column(String(20), nullable=True)  # unsolved | partial | adequate
    confidence_score = Column(Float, nullable=True)
    raw_data = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    domains = relationship("ProblemDomain", back_populates="problem", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_problems_solution_status", "solution_status"),
        Index("ix_problems_confidence_score", "confidence_score"),
        Index("ix_problems_created_at", "created_at"),
        Index("ix_problems_source", "source"),
        Index("ix_problems_source_url", "source_url"),
    )

    def __repr__(self):
        return f"<Problem id={self.id} source={self.source} status={self.solution_status}>"


class Domain(Base):
    __tablename__ = "domains"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True)

    problems = relationship("ProblemDomain", back_populates="domain")

    def __repr__(self):
        return f"<Domain name={self.name}>"


class ProblemDomain(Base):
    __tablename__ = "problem_domains"

    problem_id = Column(UUID(as_uuid=True), ForeignKey("problems.id", ondelete="CASCADE"), primary_key=True)
    domain_id = Column(Integer, ForeignKey("domains.id", ondelete="CASCADE"), primary_key=True)
    confidence = Column(Float, nullable=True)

    problem = relationship("Problem", back_populates="domains")
    domain = relationship("Domain", back_populates="problems")

    __table_args__ = (
        Index("ix_problem_domains_domain_id", "domain_id"),
        Index("ix_problem_domains_problem_id", "problem_id"),
    )
