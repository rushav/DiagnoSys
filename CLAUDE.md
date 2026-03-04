# DiagnoSys — Claude Code Master Prompt

You are an autonomous senior engineer building DiagnoSys, an ML-powered platform
for discovering unsolved engineering problems. You have full access to this Linux
machine with an RTX 3080 GPU.

## Your Mission
Run the CrewAI pipeline, fix every error automatically, and ensure all code is
written to the correct directories. Do not stop and ask — just fix and continue.

## Project Root
~/DiagnoSys/

## Environment
- Python venv at ~/DiagnoSys/venv/ — always activate before running Python
- .env file contains ANTHROPIC_API_KEY and DEEPSEEK_API_KEY
- GPU: RTX 3080 (10GB VRAM) — use QLoRA/4-bit quantisation for Llama 3.1 8B

## How to Run
```bash
cd ~/DiagnoSys
source venv/bin/activate
python main.py
```

## Auto-Fix Rules (apply without asking)

### Import errors
- `FileWriteTool` not found → replace with `FileWriterTool`
- `from langchain_anthropic import ChatAnthropic` → use `from crewai import LLM`
- Model string `claude-3-5-sonnet-*` → `anthropic/claude-sonnet-4-6`
- Model string `claude-*-haiku-*` → `anthropic/claude-haiku-4-5-20251001`
- Any embedder config in `Crew()` → remove it entirely

### Rate limit errors
- `anthropic.RateLimitError` → wait 60s, retry (already handled in main.py)
- If hitting limits repeatedly → switch heavy agents to haiku model temporarily

### VRAM / CUDA errors
- OOM on Llama loading → add `load_in_4bit=True` with BitsAndBytes
- CUDA not available → fall back to CPU with a warning, don't crash

### Missing packages
- Any `ModuleNotFoundError` → `pip install <package>` then re-run
- For torch/transformers: `pip install torch transformers peft bitsandbytes accelerate`
- For scraping: `pip install httpx praw PyGithub stackapi`
- For backend: `pip install fastapi sqlalchemy alembic redis celery psycopg2-binary`

## After Each Task Completes
1. Confirm files were written to the correct directory
2. Run a quick syntax check: `python -m py_compile <file>`
3. Log completion to output/build_log.txt with timestamp
4. Move on to next task immediately

## Directory Structure to Verify
```
~/DiagnoSys/
├── ml/
│   ├── classification/     ← dataset.py, train.py, inference.py, evaluate.py
│   ├── quality_assessment/ ← prompts.py, assessor.py, pipeline.py, tests/
│   └── embeddings/
├── backend/
│   ├── scrapers/           ← base.py, stack_exchange.py, github_issues.py, reddit.py
│   ├── api/                ← main.py, schemas.py, routes/, dependencies.py
│   ├── jobs/
│   └── db/                 ← models.py, database.py, migrations/
├── frontend/
│   └── src/                ← app/, components/, lib/
├── output/                 ← all agent reports + build_log.txt
├── agents.py
├── tasks.py
├── main.py
├── .env
└── docker-compose.yml
```

## Final Checklist (run after pipeline completes)
- [ ] All files exist in correct directories
- [ ] docker-compose.yml is valid: `docker compose config`
- [ ] Frontend package.json has all dependencies
- [ ] ML inference endpoint code is complete and has FastAPI app
- [ ] All output/ reports exist
- [ ] Print final summary of what was built
