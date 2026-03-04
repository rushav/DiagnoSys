"""
reddit.py — DiagnoSys Backend Scrapers
Reddit scraper using PRAW library.
"""

import logging
import os
from typing import List

import praw
from praw.models import Submission

from base import BaseScraper, RawProblem

logger = logging.getLogger(__name__)

TARGET_SUBREDDITS = [
    "learnprogramming", "cscareerquestions", "MachineLearning",
    "devops", "webdev", "datascience", "AskComputerScience",
]


class RedditScraper(BaseScraper):
    def __init__(self, client_id: str = None, client_secret: str = None, **kwargs):
        super().__init__(rate_per_second=1.5, bucket_capacity=10.0, **kwargs)
        self.reddit = praw.Reddit(
            client_id=client_id or os.getenv("REDDIT_CLIENT_ID", ""),
            client_secret=client_secret or os.getenv("REDDIT_CLIENT_SECRET", ""),
            user_agent="DiagnoSys/1.0 (engineering problem collector)",
            ratelimit_seconds=100,
        )

    async def scrape(self, limit_per_sub: int = 200) -> List[RawProblem]:
        """Scrape 'new' and 'hot' posts from target subreddits."""
        import asyncio
        problems = []
        loop = asyncio.get_event_loop()

        def _fetch_sub(subreddit_name: str) -> List[RawProblem]:
            sub_problems = []
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                for post in list(subreddit.new(limit=limit_per_sub // 2)) + list(subreddit.hot(limit=limit_per_sub // 2)):
                    if post.is_self and post.selftext and len(post.selftext) > 50:
                        sub_problems.append(RawProblem(
                            title=post.title,
                            description=post.selftext[:5000],
                            source="reddit",
                            source_url=f"https://reddit.com{post.permalink}",
                            tags=[subreddit_name],
                            raw_data={
                                "subreddit": subreddit_name,
                                "post_id": post.id,
                                "score": post.score,
                                "num_comments": post.num_comments,
                                "created_utc": post.created_utc,
                                "upvote_ratio": post.upvote_ratio,
                            },
                        ))
            except Exception as e:
                logger.error(f"Error scraping r/{subreddit_name}: {e}")
            return sub_problems

        for sub in TARGET_SUBREDDITS:
            sub_results = await loop.run_in_executor(None, _fetch_sub, sub)
            problems.extend(sub_results)

        logger.info(f"Reddit: scraped {len(problems)} posts")
        return problems
