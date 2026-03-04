"""
assessor.py — DiagnoSys Quality Assessment
DeepSeekAssessor: async batch processing with retry, Redis caching, cost tracking.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from typing import Optional

import httpx
import redis.asyncio as aioredis

from prompts import build_assessment_prompt, parse_verdict

logger = logging.getLogger(__name__)

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL = 86400 * 7  # 7 days


class DeepSeekAssessor:
    def __init__(
        self,
        api_key: Optional[str] = None,
        concurrency: int = 5,
        max_retries: int = 3,
        cache_enabled: bool = True,
    ):
        self.api_key = api_key or os.environ["DEEPSEEK_API_KEY"]
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.cache_enabled = cache_enabled
        self._redis: Optional[aioredis.Redis] = None
        self._semaphore = asyncio.Semaphore(concurrency)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    async def _get_redis(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
        return self._redis

    def _cache_key(self, title: str, description: str, answers: list) -> str:
        content = json.dumps({"title": title, "desc": description[:500], "ans": answers}, sort_keys=True)
        return f"diagnosys:quality:{hashlib.sha256(content.encode()).hexdigest()}"

    async def _call_api(self, messages: list) -> dict:
        """Call DeepSeek API with exponential backoff retry."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.1,
        }
        last_error = None
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(DEEPSEEK_API_URL, headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    usage = data.get("usage", {})
                    self.total_prompt_tokens += usage.get("prompt_tokens", 0)
                    self.total_completion_tokens += usage.get("completion_tokens", 0)
                    return data
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                last_error = e
                wait = (2 ** attempt) + 1
                logger.warning(f"API error (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                await asyncio.sleep(wait)
        raise RuntimeError(f"DeepSeek API failed after {self.max_retries} retries: {last_error}")

    async def assess_single(
        self,
        title: str,
        description: str,
        answers: list[str],
        problem_id: Optional[str] = None,
    ) -> dict:
        """Assess a single problem, with Redis caching."""
        async with self._semaphore:
            cache_key = self._cache_key(title, description, answers)

            if self.cache_enabled:
                redis = await self._get_redis()
                cached = await redis.get(cache_key)
                if cached:
                    logger.debug(f"Cache hit for {problem_id or cache_key[:16]}")
                    return json.loads(cached)

            messages = build_assessment_prompt(title, description, answers)
            api_response = await self._call_api(messages)
            response_text = api_response["choices"][0]["message"]["content"]
            result = parse_verdict(response_text)

            if problem_id:
                result["problem_id"] = problem_id

            if self.cache_enabled:
                redis = await self._get_redis()
                await redis.setex(cache_key, CACHE_TTL, json.dumps(result))

            return result

    async def assess_batch(self, problems: list[dict]) -> list[dict]:
        """
        Assess a batch of problems concurrently.
        Each problem dict: {id, title, description, answers: [str]}
        """
        tasks = [
            self.assess_single(
                p["title"],
                p.get("description", ""),
                p.get("answers", []),
                p.get("id"),
            )
            for p in problems
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        output = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error(f"Error assessing problem {problems[i].get('id')}: {r}")
                output.append({"problem_id": problems[i].get("id"), "verdict": "unsolved", "error": str(r)})
            else:
                output.append(r)
        return output

    def log_cost_summary(self):
        """Log token usage and estimated cost."""
        cost = (self.total_prompt_tokens * 0.14 + self.total_completion_tokens * 0.28) / 1_000_000
        logger.info(
            f"DeepSeek usage — prompt: {self.total_prompt_tokens:,} tokens, "
            f"completion: {self.total_completion_tokens:,} tokens, "
            f"estimated cost: ${cost:.4f}"
        )

    async def close(self):
        if self._redis:
            await self._redis.aclose()
