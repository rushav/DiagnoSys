"""
pipeline.py — DiagnoSys Quality Assessment
Celery tasks for async quality assessment pipeline.
"""

import asyncio
import logging
import os

from celery import Celery
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from assessor import DeepSeekAssessor

logger = logging.getLogger(__name__)

CELERY_BROKER = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
CELERY_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://diagnosys:diagnosys@localhost:5432/diagnosys")

app = Celery("quality_pipeline", broker=CELERY_BROKER, backend=CELERY_BACKEND)
app.conf.update(task_serializer="json", result_serializer="json", accept_content=["json"])

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
Session = sessionmaker(bind=engine)


def _run_async(coro):
    """Run async coroutine from sync Celery task."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@app.task(bind=True, max_retries=3, default_retry_delay=60)
def assess_problem_quality(self, problem_id: str):
    """Celery task: assess quality of a single problem."""
    try:
        with Session() as session:
            row = session.execute(
                text("SELECT title, description FROM problems WHERE id = :id"),
                {"id": problem_id},
            ).fetchone()
            if not row:
                logger.warning(f"Problem {problem_id} not found")
                return {"status": "not_found"}

            answers_rows = session.execute(
                text("SELECT body FROM answers WHERE problem_id = :id ORDER BY score DESC LIMIT 5"),
                {"id": problem_id},
            ).fetchall()
            answers = [r.body for r in answers_rows]

        assessor = DeepSeekAssessor()
        result = _run_async(
            assessor.assess_single(row.title, row.description or "", answers, problem_id)
        )

        with Session() as session:
            session.execute(
                text("""
                    UPDATE problems
                    SET solution_status = :verdict,
                        updated_at = NOW()
                    WHERE id = :id
                """),
                {"verdict": result["verdict"], "id": problem_id},
            )
            session.commit()

        logger.info(f"Problem {problem_id} assessed: {result['verdict']}")
        return result

    except Exception as exc:
        logger.error(f"Error assessing {problem_id}: {exc}")
        raise self.retry(exc=exc)


@app.task
def batch_assess_problems(limit: int = 100):
    """Celery task: assess a batch of unassessed problems."""
    with Session() as session:
        rows = session.execute(
            text("""
                SELECT p.id, p.title, p.description
                FROM problems p
                WHERE p.solution_status IS NULL
                ORDER BY p.created_at DESC
                LIMIT :limit
            """),
            {"limit": limit},
        ).fetchall()

        for row in rows:
            ans_rows = session.execute(
                text("SELECT body FROM answers WHERE problem_id = :id ORDER BY score DESC LIMIT 5"),
                {"id": row.id},
            ).fetchall()
            answers = [r.body for r in ans_rows]

    assessor = DeepSeekAssessor()
    problems = [
        {"id": r.id, "title": r.title, "description": r.description or "", "answers": answers}
        for r in rows
    ]

    results = _run_async(assessor.assess_batch(problems))
    assessor.log_cost_summary()

    with Session() as session:
        for result in results:
            if "error" not in result and result.get("problem_id"):
                session.execute(
                    text("UPDATE problems SET solution_status = :verdict, updated_at = NOW() WHERE id = :id"),
                    {"verdict": result["verdict"], "id": result["problem_id"]},
                )
        session.commit()

    return {"assessed": len(results)}
