"""
dependencies.py — DiagnoSys API Dependencies
Redis and DB session FastAPI dependencies.
"""

import os
from typing import AsyncGenerator, Optional

import redis.asyncio as aioredis
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../db"))
from database import get_db as _get_db

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
_redis_pool: Optional[aioredis.Redis] = None


async def get_redis() -> Optional[aioredis.Redis]:
    global _redis_pool
    if _redis_pool is None:
        try:
            _redis_pool = await aioredis.from_url(REDIS_URL, decode_responses=True)
        except Exception:
            return None
    return _redis_pool


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async for session in _get_db():
        yield session
