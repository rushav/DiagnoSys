"""
github_issues.py — DiagnoSys Backend Scrapers
GitHub REST API v3 scraper for open issues.
"""

import logging
import os
from typing import List

from base import BaseScraper, RawProblem

logger = logging.getLogger(__name__)

GH_API_BASE = "https://api.github.com"
TARGET_LABELS = ["bug", "help wanted", "question"]
TARGET_REPOS = [
    "python/cpython", "django/django", "pallets/flask", "fastapi/fastapi",
    "pytorch/pytorch", "huggingface/transformers", "tensorflow/tensorflow",
    "kubernetes/kubernetes", "docker/compose", "redis/redis",
    "postgres/postgres", "nginx/nginx", "ansible/ansible", "hashicorp/terraform",
    "microsoft/vscode", "neovim/neovim", "rust-lang/rust", "golang/go",
]


class GitHubScraper(BaseScraper):
    def __init__(self, token: str = None, **kwargs):
        super().__init__(rate_per_second=1.0, bucket_capacity=10.0, **kwargs)
        self.token = token or os.getenv("GITHUB_TOKEN", "")

    def _headers(self) -> dict:
        h = {"Accept": "application/vnd.github.v3+json", "X-GitHub-Api-Version": "2022-11-28"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    async def _fetch_issues(self, repo: str, label: str, page: int = 1) -> list:
        url = f"{GH_API_BASE}/repos/{repo}/issues"
        params = {
            "state": "open",
            "labels": label,
            "per_page": 100,
            "page": page,
            "sort": "created",
            "direction": "desc",
        }
        resp = await self._request("GET", url, headers=self._headers(), params=params)
        return resp.json()

    async def scrape(self, pages_per_repo: int = 2) -> List[RawProblem]:
        problems = []
        for repo in TARGET_REPOS:
            for label in TARGET_LABELS:
                for page in range(1, pages_per_repo + 1):
                    try:
                        items = await self._fetch_issues(repo, label, page)
                        if not items:
                            break
                        for issue in items:
                            if issue.get("pull_request"):
                                continue  # skip PRs
                            problems.append(RawProblem(
                                title=issue.get("title", ""),
                                description=(issue.get("body") or "")[:5000],
                                source="github",
                                source_url=issue.get("html_url", ""),
                                tags=[l["name"] for l in issue.get("labels", [])],
                                raw_data={
                                    "repo": repo,
                                    "issue_number": issue.get("number"),
                                    "labels": [l["name"] for l in issue.get("labels", [])],
                                    "created_at": issue.get("created_at"),
                                },
                            ))
                    except Exception as e:
                        logger.error(f"Error scraping {repo} label={label} page={page}: {e}")
        logger.info(f"GitHub: scraped {len(problems)} issues")
        return problems
