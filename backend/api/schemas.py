"""
schemas.py — DiagnoSys API Schemas
Pydantic v2 schemas matching PRD API contracts.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class DomainSchema(BaseModel):
    name: str
    confidence: Optional[float] = None

    model_config = {"from_attributes": True}


class ProblemResponse(BaseModel):
    id: uuid.UUID
    title: str
    description: Optional[str] = None
    source: str
    source_url: str
    solution_status: Optional[str] = None
    confidence_score: Optional[float] = None
    domains: List[DomainSchema] = []
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ProblemListResponse(BaseModel):
    items: List[ProblemResponse]
    total: int
    page: int
    page_size: int
    pages: int


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    context: str = Field(default="", max_length=2000)


class ClassifyResponse(BaseModel):
    domains: List[str]
    confidence: float
    model_version: str
