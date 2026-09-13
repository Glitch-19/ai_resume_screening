"""
AI Resume Screening & Job Matching System — Backend Entry Point
FastAPI app with health-check route per ARCHITECTURE.md §8.
"""
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

_START_TIME = datetime.now(timezone.utc)

# Optional: lazy model loading (stub V1, full ST in M1)
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_MODEL_LOADED = False  # set True when SentenceTransformer is instantiated


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _MODEL_LOADED
    # TODO: Load SentenceTransformer here in M1
    # from sentence_transformers import SentenceTransformer
    # app.state.st_model = SentenceTransformer(_MODEL_NAME)
    # _MODEL_LOADED = True
    yield


app = FastAPI(
    title="AI Resume Screening & Job Matching API",
    version="1.0.0",
    description="Hybrid TF-IDF + Semantic (all-MiniLM-L6-v2) resume matcher",
    lifespan=lifespan,
)

# CORS for Vite frontend (5173) + production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["health"])
async def health_check():
    """Liveness & readiness probe. Used by Docker HEALTHCHECK and frontend."""
    uptime = (datetime.now(timezone.utc) - _START_TIME).total_seconds()
    return {
        "status": "ok",
        "model_loaded": _MODEL_LOADED,
        "model_name": _MODEL_NAME,
        "version": app.version,
        "uptime_seconds": round(uptime, 2),
    }


@app.get("/", tags=["root"])
async def root():
    return {
        "message": "AI Resume Screening API is running",
        "docs": "/docs",
        "health": "/health",
    }


# Placeholder for future routers (PRD §9):
# from app.api.v1.router import api_router
# app.include_router(api_router, prefix="/api/v1")
