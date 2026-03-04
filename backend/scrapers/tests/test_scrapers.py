"""
test_scrapers.py — Unit tests for DiagnoSys scrapers
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from base import BaseScraper, RawProblem, TokenBucket
from stack_exchange import StackExchangeScraper
from github_issues import GitHubScraper


# ── TokenBucket ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_token_bucket_acquire():
    bucket = TokenBucket(rate=100.0, capacity=5.0)
    await bucket.acquire()  # Should succeed immediately


# ── StackExchangeScraper ──────────────────────────────────────────────────────

MOCK_SE_RESPONSE = {
    "items": [
        {
            "title": "How to optimize PostgreSQL with 10M rows?",
            "body": "<p>I have a large table...</p>",
            "tags": ["postgresql", "performance"],
            "link": "https://stackoverflow.com/questions/1234",
            "question_id": 1234,
            "score": 5,
            "answer_count": 0,
            "creation_date": 1700000000,
        }
    ],
    "has_more": False,
}


@pytest.mark.asyncio
async def test_stack_exchange_scraper():
    scraper = StackExchangeScraper(api_key="test")
    mock_resp = MagicMock()
    mock_resp.json.return_value = MOCK_SE_RESPONSE
    mock_resp.raise_for_status = MagicMock()
    with patch.object(scraper, "_request", new=AsyncMock(return_value=mock_resp)):
        problems = await scraper.scrape(pages_per_site=1)
    assert len(problems) == len(SE_SITES_COUNT := 10)  # 10 sites × 1 problem each
    assert problems[0].source == "stack_exchange"
    assert problems[0].source_url == "https://stackoverflow.com/questions/1234"
    await scraper.close()


# ── GitHubScraper ─────────────────────────────────────────────────────────────

MOCK_GH_ISSUE = [
    {
        "title": "Memory leak in DataLoader",
        "body": "When using DataLoader with num_workers > 0...",
        "html_url": "https://github.com/pytorch/pytorch/issues/999",
        "number": 999,
        "labels": [{"name": "bug"}],
        "created_at": "2024-01-15T10:00:00Z",
        "pull_request": None,
    }
]


@pytest.mark.asyncio
async def test_github_scraper():
    scraper = GitHubScraper(token="test-token")
    mock_resp = MagicMock()
    mock_resp.json.return_value = MOCK_GH_ISSUE
    mock_resp.raise_for_status = MagicMock()
    with patch.object(scraper, "_request", new=AsyncMock(return_value=mock_resp)):
        problems = await scraper.scrape(pages_per_repo=1)
    assert any(p.source == "github" for p in problems)
    assert any("pytorch" in p.source_url for p in problems)
    await scraper.close()


@pytest.mark.asyncio
async def test_github_scraper_skips_prs():
    scraper = GitHubScraper(token="test-token")
    pr_issue = [{**MOCK_GH_ISSUE[0], "pull_request": {"url": "..."}}]
    mock_resp = MagicMock()
    mock_resp.json.return_value = pr_issue
    mock_resp.raise_for_status = MagicMock()
    with patch.object(scraper, "_request", new=AsyncMock(return_value=mock_resp)):
        problems = await scraper.scrape(pages_per_repo=1)
    assert not any("pytorch/pytorch/issues/999" in p.source_url for p in problems)
    await scraper.close()


@pytest.mark.asyncio
async def test_deduplication():
    from unittest.mock import AsyncMock as AM
    scraper = StackExchangeScraper(api_key="test")
    problems = [RawProblem("T", "D", "stack_exchange", "http://existing.url")]
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.__iter__ = MagicMock(return_value=iter([MagicMock(source_url="http://existing.url")]))
    mock_session.execute = AM(return_value=mock_result)
    filtered = await scraper.filter_existing(problems, mock_session)
    assert len(filtered) == 0
