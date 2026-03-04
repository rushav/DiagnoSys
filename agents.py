from crewai import Agent, LLM
from crewai_tools import FileWriterTool, DirectoryReadTool

# ── Models ──────────────────────────────────────────────────────────────────
sonnet  = LLM(model="anthropic/claude-sonnet-4-6",   temperature=0.3)
haiku   = LLM(model="anthropic/claude-haiku-4-5-20251001", temperature=0.3)

file_writer = FileWriterTool()
dir_reader  = DirectoryReadTool()

# ── PRD Context (injected into every agent backstory) ────────────────────────
PRD = """
DiagnoSys is an ML-powered web platform that discovers and categorizes unsolved 
engineering problems from Stack Exchange, GitHub Issues, and Reddit.

Tech Stack:
- Backend:  FastAPI, PostgreSQL, SQLAlchemy, Celery + Redis, Alembic
- ML:       PyTorch + Transformers, Llama 3.1 8B (fine-tuned), DeepSeek-V3 API
- Frontend: Next.js 16, TypeScript, Tailwind CSS, React Query
- Vector DB: Pinecone or Weaviate for semantic search

Key targets:
- 50K problems catalogued (MVP), 500K+ (production)
- 80%+ classification precision
- <100ms query response (P50), <500ms (P99)
- ML inference <200ms per problem

Folder layout (use RELATIVE paths, CWD is already /home/rushav/DiagnoSys/):
  ml/classification/        ← Llama 3.1 fine-tuning & inference
  ml/quality_assessment/    ← DeepSeek-V3 integration & CoT prompts
  ml/embeddings/            ← Embedding generation for vector search
  backend/scrapers/         ← Stack Exchange, GitHub, Reddit scrapers
  backend/api/              ← FastAPI routes, schemas, dependencies
  backend/jobs/             ← Celery tasks & scheduler
  backend/db/               ← SQLAlchemy models & Alembic migrations
  frontend/                 ← Next.js 16 app
  output/                   ← Agent reports & summaries

CRITICAL: Use ONLY relative directory paths like "ml/classification" (NOT "~/DiagnoSys/ml/classification").
The working directory is already /home/rushav/DiagnoSys/.
"""

# ── Agents ───────────────────────────────────────────────────────────────────

ml_classifier_agent = Agent(
    role="ML Classification Engineer",
    goal=(
        "Build and fine-tune the Llama 3.1 8B model for multi-label domain "
        "classification of engineering problems. Produce a FastAPI inference "
        "endpoint with <200ms latency and 80%+ precision."
    ),
    backstory=(
        f"You are an expert ML engineer specialising in LLM fine-tuning on "
        f"consumer GPUs (RTX 3080). You write production-quality PyTorch + "
        f"HuggingFace Transformers code with QLoRA/PEFT for memory efficiency.\n\n"
        f"IMPORTANT: When calling file_writer_tool, ALWAYS include the 'content' "
        f"field with the full file content in the same tool call. Never call the "
        f"tool without content.\n\nPROJECT CONTEXT:\n{PRD}"
    ),
    tools=[file_writer, dir_reader],
    llm=sonnet,
    verbose=True,
    allow_delegation=False,
    max_iter=20,
)

ml_quality_agent = Agent(
    role="ML Quality Assessment Engineer",
    goal=(
        "Integrate DeepSeek-V3 API to classify problems as unsolved / partial / "
        "adequate using Chain-of-Thought reasoning and structured XML-tagged prompts."
    ),
    backstory=(
        f"You are an expert in LLM prompt engineering and API integration. "
        f"You write robust Python clients with retry logic, structured output "
        f"parsing, and cost-efficient batching.\n\n"
        f"IMPORTANT: When calling file_writer_tool, ALWAYS include the 'content' "
        f"field with the full file content in the same tool call.\n\nPROJECT CONTEXT:\n{PRD}"
    ),
    tools=[file_writer, dir_reader],
    llm=sonnet,
    verbose=True,
    allow_delegation=False,
    max_iter=20,
)

backend_scraper_agent = Agent(
    role="Backend Data Collection Engineer",
    goal=(
        "Build production scrapers for Stack Exchange API, GitHub Issues API, "
        "and Reddit API. Implement rate limiting, retry logic, deduplication, "
        "and Celery background jobs."
    ),
    backstory=(
        f"You are a senior backend engineer specialising in data pipelines, "
        f"async Python, and API integrations. You write well-tested FastAPI + "
        f"Celery code that gracefully handles rate limits.\n\n"
        f"IMPORTANT: When calling file_writer_tool, ALWAYS include the 'content' "
        f"field with the full file content in the same tool call.\n\nPROJECT CONTEXT:\n{PRD}"
    ),
    tools=[file_writer, dir_reader],
    llm=sonnet,
    verbose=True,
    allow_delegation=False,
    max_iter=20,
)

backend_api_agent = Agent(
    role="Backend API Engineer",
    goal=(
        "Build the FastAPI REST API with PostgreSQL via SQLAlchemy, Redis caching, "
        "Alembic migrations, and all endpoints from the PRD. Achieve <100ms P50 "
        "query response times."
    ),
    backstory=(
        f"You are an expert in FastAPI, PostgreSQL query optimisation, and "
        f"Redis caching strategies. You write clean, well-documented APIs with "
        f"Pydantic schemas and OpenAPI specs.\n\n"
        f"IMPORTANT: When calling file_writer_tool, ALWAYS include the 'content' "
        f"field with the full file content in the same tool call.\n\nPROJECT CONTEXT:\n{PRD}"
    ),
    tools=[file_writer, dir_reader],
    llm=sonnet,
    verbose=True,
    allow_delegation=False,
    max_iter=20,
)

frontend_agent = Agent(
    role="Frontend Engineer",
    goal=(
        "Build the complete Next.js 16 + TypeScript + Tailwind CSS frontend: "
        "search interface, filter panel, problem cards, detail pages, "
        "responsive design, loading states and error handling."
    ),
    backstory=(
        f"You are a senior frontend engineer expert in Next.js App Router, "
        f"TypeScript, Tailwind CSS, and React Query. You write clean, "
        f"accessible, performant UIs.\n\n"
        f"IMPORTANT: When calling file_writer_tool, ALWAYS include the 'content' "
        f"field with the full file content in the same tool call.\n\nPROJECT CONTEXT:\n{PRD}"
    ),
    tools=[file_writer, dir_reader],
    llm=haiku,
    verbose=True,
    allow_delegation=False,
    max_iter=20,
)
