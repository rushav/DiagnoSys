"""
scheduler.py — DiagnoSys Backend Scrapers
Celery beat schedule for periodic scraping.
"""

import os
from celery import Celery
from celery.schedules import crontab

CELERY_BROKER = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
CELERY_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")

app = Celery("scraper_scheduler", broker=CELERY_BROKER, backend=CELERY_BACKEND)

app.conf.beat_schedule = {
    "stack_exchange_scrape": {
        "task": "jobs.scrape_stack_exchange",
        "schedule": crontab(minute=0, hour="*/2"),  # every 2 hours
    },
    "github_scrape": {
        "task": "jobs.scrape_github",
        "schedule": crontab(minute="*/30"),  # every 30 minutes
    },
    "reddit_scrape": {
        "task": "jobs.scrape_reddit",
        "schedule": crontab(minute="*/15"),  # every 15 minutes
    },
}
app.conf.timezone = "UTC"
