"""
problems.py — DiagnoSys API Routes
GET /api/problems with filtering, pagination, and Redis caching.
"""

import hashlib
import json
import math
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from dependencies import get_db, get_redis
from schemas import ProblemListResponse, ProblemResponse, DomainSchema

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../db"))
from models import Domain, Problem, ProblemDomain

router = APIRouter(prefix="/api/problems", tags=["problems"])
CACHE_TTL = 300  # 5 minutes


def _cache_key(page, page_size, domain, status, min_confidence) -> str:
    key = f"problems:{page}:{page_size}:{domain}:{status}:{min_confidence}"
    return hashlib.sha256(key.encode()).hexdigest()


@router.get("", response_model=ProblemListResponse)
async def list_problems(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    domain: Optional[str] = Query(None),
    status: Optional[str] = Query(None, pattern="^(unsolved|partial|adequate)$"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0),
    db: AsyncSession = Depends(get_db),
    redis=Depends(get_redis),
):
    cache_key = _cache_key(page, page_size, domain, status, min_confidence)
    if redis:
        cached = await redis.get(cache_key)
        if cached:
            return ProblemListResponse(**json.loads(cached))

    query = select(Problem).options(selectinload(Problem.domains).selectinload(ProblemDomain.domain))

    if status:
        query = query.where(Problem.solution_status == status)
    if min_confidence is not None:
        query = query.where(Problem.confidence_score >= min_confidence)
    if domain:
        query = query.join(Problem.domains).join(ProblemDomain.domain).where(Domain.name == domain)

    count_query = select(func.count()).select_from(query.subquery())
    total = (await db.execute(count_query)).scalar_one()

    query = query.order_by(Problem.created_at.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    problems = result.scalars().unique().all()

    items = []
    for p in problems:
        domains = [
            DomainSchema(name=pd.domain.name, confidence=pd.confidence)
            for pd in p.domains if pd.domain
        ]
        items.append(ProblemResponse(
            id=p.id,
            title=p.title,
            description=p.description,
            source=p.source,
            source_url=p.source_url,
            solution_status=p.solution_status,
            confidence_score=p.confidence_score,
            domains=domains,
            created_at=p.created_at,
            updated_at=p.updated_at,
        ))

    response = ProblemListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        pages=math.ceil(total / page_size) if total else 0,
    )

    if redis:
        await redis.setex(cache_key, CACHE_TTL, response.model_dump_json())

    return response
