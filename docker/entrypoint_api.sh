#!/usr/bin/env sh
# Builds the vector index if missing, then starts the FastAPI server.
set -e

if [ ! -f "data/vectorstore/manifest.json" ]; then
  echo "[entrypoint] Vector index not found - building it now..."
  python -m app.rag.ingest
fi

echo "[entrypoint] Starting FastAPI server..."
exec uvicorn app.api.main:app --host 0.0.0.0 --port 8000
