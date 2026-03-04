"""
test_assessor.py — Unit tests for DeepSeekAssessor
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from assessor import DeepSeekAssessor
from prompts import parse_verdict, build_assessment_prompt


# ── parse_verdict tests ──────────────────────────────────────────────────────

def test_parse_verdict_adequate():
    text = "<reasoning>Well solved.</reasoning><verdict>adequate</verdict>"
    result = parse_verdict(text)
    assert result["verdict"] == "adequate"
    assert result["reasoning"] == "Well solved."


def test_parse_verdict_partial():
    text = "<reasoning>Partially answered.</reasoning><verdict>partial</verdict>"
    result = parse_verdict(text)
    assert result["verdict"] == "partial"


def test_parse_verdict_unsolved():
    text = "<reasoning>No answers.</reasoning><verdict>unsolved</verdict>"
    result = parse_verdict(text)
    assert result["verdict"] == "unsolved"


def test_parse_verdict_missing_tags():
    """Falls back to 'unsolved' when tags are missing."""
    result = parse_verdict("Some response without XML tags")
    assert result["verdict"] == "unsolved"
    assert result["reasoning"] == ""


def test_parse_verdict_invalid_verdict():
    text = "<reasoning>Analysis</reasoning><verdict>unknown</verdict>"
    result = parse_verdict(text)
    assert result["verdict"] == "unsolved"  # falls back


# ── build_assessment_prompt tests ────────────────────────────────────────────

def test_build_assessment_prompt_structure():
    msgs = build_assessment_prompt("Test title", "Test description", ["Answer 1"])
    assert msgs[0]["role"] == "system"
    assert "XML" in msgs[0]["content"]
    user_msg = msgs[-1]
    assert user_msg["role"] == "user"
    assert "Test title" in user_msg["content"]
    assert "Answer 1" in user_msg["content"]


def test_build_assessment_prompt_no_answers():
    msgs = build_assessment_prompt("Title", "Desc", [])
    user_msg = msgs[-1]
    assert "(no answers)" in user_msg["content"]


# ── DeepSeekAssessor tests ───────────────────────────────────────────────────

MOCK_API_RESPONSE = {
    "choices": [{"message": {"content": "<reasoning>Good solution.</reasoning><verdict>adequate</verdict>"}}],
    "usage": {"prompt_tokens": 100, "completion_tokens": 50},
}

MOCK_API_RESPONSE_PARTIAL = {
    "choices": [{"message": {"content": "<reasoning>Incomplete.</reasoning><verdict>partial</verdict>"}}],
    "usage": {"prompt_tokens": 80, "completion_tokens": 40},
}


@pytest.mark.asyncio
async def test_assess_single_adequate():
    assessor = DeepSeekAssessor(api_key="test-key", cache_enabled=False)
    with patch.object(assessor, "_call_api", new=AsyncMock(return_value=MOCK_API_RESPONSE)):
        result = await assessor.assess_single("Title", "Desc", ["Answer"], "prob-1")
    assert result["verdict"] == "adequate"
    assert result["problem_id"] == "prob-1"


@pytest.mark.asyncio
async def test_assess_single_partial():
    assessor = DeepSeekAssessor(api_key="test-key", cache_enabled=False)
    with patch.object(assessor, "_call_api", new=AsyncMock(return_value=MOCK_API_RESPONSE_PARTIAL)):
        result = await assessor.assess_single("Title", "Desc", [], "prob-2")
    assert result["verdict"] == "partial"


@pytest.mark.asyncio
async def test_assess_batch():
    assessor = DeepSeekAssessor(api_key="test-key", cache_enabled=False)
    problems = [
        {"id": "p1", "title": "T1", "description": "D1", "answers": ["A1"]},
        {"id": "p2", "title": "T2", "description": "D2", "answers": []},
    ]
    with patch.object(assessor, "_call_api", new=AsyncMock(return_value=MOCK_API_RESPONSE)):
        results = await assessor.assess_batch(problems)
    assert len(results) == 2
    for r in results:
        assert "verdict" in r


@pytest.mark.asyncio
async def test_assess_single_api_timeout():
    """Handles API timeout gracefully (returns error in batch)."""
    import httpx
    assessor = DeepSeekAssessor(api_key="test-key", cache_enabled=False, max_retries=1)
    with patch.object(assessor, "_call_api", new=AsyncMock(side_effect=RuntimeError("timeout"))):
        problems = [{"id": "p1", "title": "T1", "description": "D1", "answers": []}]
        results = await assessor.assess_batch(problems)
    assert results[0].get("error") is not None


@pytest.mark.asyncio
async def test_token_tracking():
    assessor = DeepSeekAssessor(api_key="test-key", cache_enabled=False)
    with patch.object(assessor, "_call_api", new=AsyncMock(return_value=MOCK_API_RESPONSE)):
        await assessor.assess_single("T", "D", ["A"])
    assert assessor.total_prompt_tokens == 100
    assert assessor.total_completion_tokens == 50
