from crewai import Task
from agents import (
    ml_classifier_agent,
    ml_quality_agent,
    backend_scraper_agent,
    backend_api_agent,
    frontend_agent,
)

# ── Phase 1 Tasks (Weeks 1-2: Setup & Proof of Concept) ─────────────────────

task_ml_classification = Task(
    description="""
    Build the complete ML classification system for DiagnoSys.

    DELIVERABLES — write every file to ml/classification/:

    1. dataset.py
       - DatasetBuilder class that loads raw problems from PostgreSQL
       - Preprocessing: clean HTML, truncate to 512 tokens, multi-hot label encoding
       - 20 domain labels: databases, backend, frontend, ml, devops, security,
         mobile, systems, networking, web, cloud, testing, performance,
         architecture, data_engineering, nlp, computer_vision, rl, robotics, other

    2. train.py
       - QLoRA fine-tuning of Llama 3.1 8B (meta-llama/Meta-Llama-3.1-8B)
       - Use PEFT + BitsAndBytes 4-bit quantisation for RTX 3080 (10GB VRAM)
       - BCEWithLogitsLoss for multi-label classification
       - Weights & Biases experiment tracking
       - Save checkpoints to ml/classification/checkpoints/

    3. inference.py
       - FastAPI app on port 8001
       - POST /ml/classify endpoint matching PRD contract:
         { "text": str, "context": str } → { "domains": [...], "confidence": float, "model_version": str }
       - Load model once at startup, keep in GPU memory
       - Batch inference support
       - <200ms latency target

    4. evaluate.py
       - Precision, recall, F1 per domain
       - Confusion matrix
       - Target: 80%+ precision on held-out validation set

    5. requirements.txt
       - All Python dependencies with pinned versions

    After writing files, output a summary report to output/ml_classification_report.txt
    describing: architecture decisions, VRAM usage estimate, training time estimate,
    and any assumptions made.
    """,
    agent=ml_classifier_agent,
    expected_output="All 5 files written to ml/classification/ + summary report in output/",
)

task_ml_quality = Task(
    description="""
    Build the solution quality assessment system using DeepSeek-V3 API.

    DELIVERABLES — write every file to ml/quality_assessment/:

    1. prompts.py
       - XML-tagged Chain-of-Thought prompt templates
       - System prompt that instructs DeepSeek to reason step by step
       - Output format: <reasoning>...</reasoning><verdict>unsolved|partial|adequate</verdict>
       - Few-shot examples for each verdict class

    2. assessor.py
       - DeepSeekAssessor class wrapping the DeepSeek API
       - Async batch processing (process N problems concurrently)
       - Exponential backoff retry logic
       - Structured XML response parsing
       - Cost tracking (log token usage per batch)
       - Cache results in Redis to avoid re-processing

    3. pipeline.py
       - Celery task: assess_problem_quality(problem_id)
       - Reads problem from PostgreSQL, calls assessor, writes verdict back
       - Celery task: batch_assess_problems(limit=100)

    4. tests/test_assessor.py
       - Unit tests with mocked DeepSeek API responses
       - Test each verdict class + edge cases (empty text, API timeout)

    5. requirements.txt

    Output report to output/ml_quality_report.txt
    """,
    agent=ml_quality_agent,
    expected_output="All files written to ml/quality_assessment/ + report in output/",
    context=[task_ml_classification],
)

task_backend_scrapers = Task(
    description="""
    Build all three data scrapers for DiagnoSys.

    DELIVERABLES — write every file to backend/scrapers/:

    1. base.py
       - BaseScraper abstract class
       - Rate limiting via token bucket algorithm
       - Exponential backoff retry (max 3 retries)
       - Deduplication check against PostgreSQL before inserting

    2. stack_exchange.py
       - StackExchangeScraper(BaseScraper)
       - Uses Stack Exchange API v2.3
       - Fetches unanswered questions from 10 top programming sites
       - Rate limit: 10K requests/day with API key
       - Extracts: title, body, tags, score, answer_count, creation_date

    3. github_issues.py
       - GitHubScraper(BaseScraper)
       - Uses GitHub REST API v3 with personal access token
       - Targets open issues with labels: bug, help-wanted, question
       - Rate limit: 5K requests/hour
       - Extracts: title, body, labels, created_at, repo full_name

    4. reddit.py
       - RedditScraper(BaseScraper)
       - Uses PRAW library
       - Targets: r/learnprogramming, r/cscareerquestions, r/MachineLearning,
         r/devops, r/webdev, r/datascience, r/AskComputerScience
       - Rate limit: 100 requests/min
       - Extracts: title, selftext, score, num_comments, created_utc

    5. scheduler.py
       - Celery beat schedule
       - stack_exchange_scrape: every 2 hours
       - github_scrape: every 30 minutes
       - reddit_scrape: every 15 minutes

    6. tests/test_scrapers.py
       - Unit tests with mocked HTTP responses for all three scrapers

    Output report to output/backend_scrapers_report.txt
    """,
    agent=backend_scraper_agent,
    expected_output="All scraper files written to backend/scrapers/ + report in output/",
)

task_backend_api = Task(
    description="""
    Build the complete FastAPI backend for DiagnoSys.

    DELIVERABLES:

    backend/db/models.py
       - SQLAlchemy models: Problem, Domain, ProblemDomain (junction)
       - Problem fields: id (UUID), title, description, source, source_url,
         solution_status, confidence_score, raw_data (JSONB), created_at, updated_at
       - Indexes on: solution_status, confidence_score, created_at, source

    backend/db/database.py
       - SQLAlchemy async engine setup
       - get_db() dependency

    backend/db/migrations/
       - Alembic env.py + initial migration script

    backend/api/schemas.py
       - Pydantic v2 schemas matching PRD API contracts exactly
       - ProblemResponse, ProblemListResponse, ClassifyRequest, ClassifyResponse

    backend/api/routes/problems.py
       - GET /api/problems with ALL filters from PRD:
         page, page_size, domain, status (unsolved|partial|adequate), min_confidence
       - Paginated response with total count
       - Redis caching: cache query results for 5 minutes

    backend/api/routes/ml.py
       - POST /ml/classify — proxies to ML inference service on port 8001

    backend/api/main.py
       - FastAPI app with CORS, Swagger docs, lifespan events
       - Mount all routers
       - Redis connection pool setup

    backend/api/dependencies.py
       - Redis dependency, DB session dependency

    docker-compose.yml (project root)
       - Services: api, ml_classifier, worker (Celery), postgres, redis
       - Correct port mappings and environment variables

    Output report to output/backend_api_report.txt
    """,
    agent=backend_api_agent,
    expected_output="All API files written to backend/ + docker-compose.yml + report",
    context=[task_backend_scrapers],
)

task_frontend = Task(
    description="""
    Build the complete Next.js 16 frontend for DiagnoSys.

    DELIVERABLES — write all files to frontend/src/:

    app/layout.tsx
       - Root layout with Tailwind, metadata, Inter font

    app/page.tsx
       - Landing/search page
       - Hero with search bar (keyword input)
       - Filter panel: domain (multi-select), solution status, min confidence slider, date range
       - Problem cards grid (responsive: 1 col mobile, 2 col tablet, 3 col desktop)
       - Pagination controls
       - Loading skeleton states
       - Error boundary with retry

    app/problems/[id]/page.tsx
       - Problem detail page
       - Full title + description
       - Domain tags with confidence badges
       - Solution status badge (color coded: red=unsolved, yellow=partial, green=adequate)
       - Source link button
       - Related problems sidebar (placeholder for semantic search)

    components/ProblemCard.tsx
       - Reusable card: title (truncated), description preview, domain chips,
         confidence score, solution status badge, source icon, time ago

    components/FilterPanel.tsx
       - All filters with URL search params sync (shareable URLs)

    components/SearchBar.tsx
       - Debounced input (300ms), loading spinner, clear button

    lib/api.ts
       - Type-safe API client using fetch
       - All types matching backend Pydantic schemas
       - React Query hooks: useProblems(), useProblem(id)

    package.json
       - Next.js 16, TypeScript, Tailwind CSS, React Query, lucide-react

    tailwind.config.ts + tsconfig.json

    Output report to output/frontend_report.txt
    """,
    agent=frontend_agent,
    expected_output="Complete Next.js app written to frontend/src/ + report in output/",
    context=[task_backend_api],
)
