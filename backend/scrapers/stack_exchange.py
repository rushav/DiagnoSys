"""
stack_exchange.py — DiagnoSys Backend Scrapers
Stack Exchange API v2.3 scraper for unanswered questions.
"""

import logging
import os
from typing import List

import httpx

from base import BaseScraper, RawProblem

logger = logging.getLogger(__name__)

SE_API_BASE = "https://api.stackexchange.com/2.3"
SE_SITES = [
    "stackoverflow", "serverfault", "superuser", "askubuntu",
    "unix", "apple", "dba", "devops", "softwareengineering", "datascience",
]


class StackExchangeScraper(BaseScraper):
    def __init__(self, api_key: str = None, **kwargs):
        super().__init__(rate_per_second=0.5, bucket_capacity=5.0, **kwargs)
        self.api_key = api_key or os.getenv("SE_API_KEY", "")

    async def _fetch_questions(self, site: str, page: int = 1, page_size: int = 100) -> dict:
        params = {
            "site": site,
            "order": "desc",
            "sort": "creation",
            "filter": "withbody",
            "pagesize": page_size,
            "page": page,
        }
        if self.api_key:
            params["key"] = self.api_key
        resp = await self._request("GET", f"{SE_API_BASE}/questions/unanswered", params=params)
        return resp.json()

    async def scrape(self, pages_per_site: int = 5) -> List[RawProblem]:
        """Scrape unanswered questions from SE_SITES."""
        problems = []
        for site in SE_SITES:
            for page in range(1, pages_per_site + 1):
                try:
                    data = await self._fetch_questions(site, page=page)
                    for item in data.get("items", []):
                        title = item.get("title", "")
                        body = item.get("body", "") or item.get("body_markdown", "")
                        tags = item.get("tags", [])
                        link = item.get("link", "")
                        if not link:
                            continue
                        problems.append(RawProblem(
                            title=title,
                            description=body[:5000],
                            source="stack_exchange",
                            source_url=link,
                            tags=tags,
                            raw_data={
                                "site": site,
                                "question_id": item.get("question_id"),
                                "score": item.get("score", 0),
                                "answer_count": item.get("answer_count", 0),
                                "creation_date": item.get("creation_date"),
                            },
                        ))
                    if not data.get("has_more", False):
                        break
                except Exception as e:
                    logger.error(f"Error scraping {site} page {page}: {e}")
        logger.info(f"StackExchange: scraped {len(problems)} problems")
        return problems
