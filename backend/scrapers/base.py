"""
base.py — DiagnoSys Backend Scrapers
BaseScraper with token bucket rate limiting, exponential backoff retry,
and PostgreSQL deduplication.
"""

import abc
import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass
class RawProblem:
    title: str
    description: str
    source: str  # "stack_exchange" | "github" | "reddit"
    source_url: str
    raw_data: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


class TokenBucket:
    """Token bucket rate limiter."""
    def __init__(self, rate: float, capacity: float):
        self.rate = rate
        self.capacity = capacity
        self._tokens = capacity
        self._last_refill = time.monotonic()

    async def acquire(self, tokens: float = 1.0):
        while True:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._last_refill = now
            if self._tokens >= tokens:
                self._tokens -= tokens
                return
            wait = (tokens - self._tokens) / self.rate
            await asyncio.sleep(wait)


class BaseScraper(abc.ABC):
    def __init__(
        self,
        rate_per_second: float = 1.0,
        bucket_capacity: float = 10.0,
        max_retries: int = 3,
    ):
        self.bucket = TokenBucket(rate_per_second, bucket_capacity)
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        return self._client

    async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Rate-limited HTTP request with exponential backoff."""
        await self.bucket.acquire()
        client = await self._get_client()
        last_error = None
        for attempt in range(self.max_retries):
            try:
                resp = await client.request(method, url, **kwargs)
                resp.raise_for_status()
                return resp
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    wait = int(e.response.headers.get("Retry-After", 60))
                    logger.warning(f"Rate limited on {url}, waiting {wait}s")
                    await asyncio.sleep(wait)
                elif e.response.status_code >= 500:
                    wait = 2 ** attempt
                    logger.warning(f"Server error {e.response.status_code} on {url}, retry in {wait}s")
                    await asyncio.sleep(wait)
                else:
                    raise
                last_error = e
            except httpx.RequestError as e:
                wait = 2 ** attempt
                logger.warning(f"Request error on {url}: {e}, retry in {wait}s")
                await asyncio.sleep(wait)
                last_error = e
        raise RuntimeError(f"Request failed after {self.max_retries} retries: {last_error}")

    @abc.abstractmethod
    async def scrape(self, **kwargs) -> List[RawProblem]:
        """Scrape problems from the source. Implement in subclasses."""
        ...

    def _dedup_key(self, source_url: str) -> str:
        return hashlib.sha256(source_url.encode()).hexdigest()

    async def filter_existing(
        self, problems: List[RawProblem], session: AsyncSession
    ) -> List[RawProblem]:
        """Filter out problems that already exist in PostgreSQL."""
        if not problems:
            return []
        urls = [p.source_url for p in problems]
        result = await session.execute(
            text("SELECT source_url FROM problems WHERE source_url = ANY(:urls)"),
            {"urls": urls},
        )
        existing = {row.source_url for row in result}
        return [p for p in problems if p.source_url not in existing]

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
