"""
main.py — DiagnoSys FastAPI Application
Main app with CORS, Swagger, Redis pool, and all routers.
"""

import os
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.problems import router as problems_router
from routes.ml import router as ml_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    app.state.redis = await aioredis.from_url(redis_url, decode_responses=True)
    yield
    # Shutdown
    await app.state.redis.aclose()


app = FastAPI(
    title="DiagnoSys API",
    description="ML-powered platform for discovering unsolved engineering problems",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(problems_router)
app.include_router(ml_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
