import os
import time
import anthropic
from dotenv import load_dotenv
from crewai import Crew, Process

load_dotenv()

from agents import (
    ml_classifier_agent,
    ml_quality_agent,
    backend_scraper_agent,
    backend_api_agent,
    frontend_agent,
)
from tasks import (
    task_ml_classification,
    task_ml_quality,
    task_backend_scrapers,
    task_backend_api,
    task_frontend,
)

def run_with_retry(crew, max_retries=3):
    for attempt in range(max_retries):
        try:
            return crew.kickoff()
        except anthropic.RateLimitError as e:
            wait = 60 * (attempt + 1)
            print(f"[Rate limit] Waiting {wait}s before retry {attempt+1}/{max_retries}...")
            time.sleep(wait)
        except Exception as e:
            print(f"[Error] {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(30)
    raise RuntimeError("Max retries exceeded")

if __name__ == "__main__":
    print("=" * 60)
    print("DiagnoSys — Multi-Agent Build Pipeline")
    print("=" * 60)

    crew = Crew(
        agents=[
            ml_classifier_agent,
            ml_quality_agent,
            backend_scraper_agent,
            backend_api_agent,
            frontend_agent,
        ],
        tasks=[
            task_ml_classification,
            task_ml_quality,
            task_backend_scrapers,
            task_backend_api,
            task_frontend,
        ],
        process=Process.sequential,
        verbose=True,
    )

    result = run_with_retry(crew)

    print("\n" + "=" * 60)
    print("Pipeline complete. Check output/ for all reports.")
    print("=" * 60)
    print(result)
