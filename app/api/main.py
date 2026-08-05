"""FastAPI application entrypoint for the Automotive GenAI diagnostic assistant."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import diagnose, feedback, monitoring
from app.core.logging_config import app_logger

app = FastAPI(
    title="Automotive Vehicle Diagnostics & Service Recommendation Assistant",
    description=(
        "RAG-powered assistant that explains diagnostic codes/symptoms, retrieves "
        "relevant service-manual content, and recommends next service steps with citations."
    ),
    version="1.0.0",
)

# CORS: restrict to local UI origins by default; adjust for your deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(diagnose.router)
app.include_router(feedback.router)
app.include_router(monitoring.router)


@app.get("/health", tags=["health"])
def health() -> dict:
    return {"status": "ok"}


@app.on_event("startup")
def on_startup() -> None:
    app_logger.info("Automotive GenAI diagnostic assistant API starting up.")
