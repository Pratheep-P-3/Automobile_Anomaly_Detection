"""Builds the vector index from synthetic diagnostic codes + service manuals.

Run as a script:
    python -m app.rag.ingest
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import settings
from app.core.logging_config import app_logger
from app.rag.chunking import chunk_manual_text
from app.rag.embeddings import get_embedding_backend
from app.rag.vectorstore import get_vectorstore


def load_diagnostic_codes() -> list[dict]:
    with settings.diagnostic_codes_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_manual_path(doc_file: str) -> Path:
    configured_path = Path(doc_file)
    pdf_name = configured_path.with_suffix(".pdf").name
    pdf_path = settings.manuals_pdf_path / pdf_name
    if pdf_path.exists():
        return pdf_path
    return settings.manuals_path / configured_path.name


def load_manual_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


def build_index() -> None:
    codes = load_diagnostic_codes()

    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []

    for entry in codes:
        doc_path = resolve_manual_path(entry["doc_file"])
        if not doc_path.exists():
            app_logger.warning("Manual file missing for code %s: %s", entry["code"], doc_path)
            continue
        raw_text = load_manual_text(doc_path)
        chunks = chunk_manual_text(raw_text, source_id=entry["code"])
        for chunk in chunks:
            ids.append(chunk.chunk_id)
            texts.append(chunk.text)
            metadatas.append(
                {
                    "code": entry["code"],
                    "title": entry["title"],
                    "system": entry["system"],
                    "severity": entry["severity"],
                    "doc_file": doc_path.name,
                    "source_type": doc_path.suffix.lower().lstrip("."),
                    "heading": chunk.heading,
                    "symptoms": entry.get("symptoms", []),
                }
            )

    if not texts:
        raise RuntimeError("No manual content found to index. Check data/manuals and diagnostic_codes.json.")

    embedding_backend = get_embedding_backend(settings.use_openai_embeddings, settings.openai_model)
    embedding_backend.fit(texts)
    embeddings = embedding_backend.transform(texts)

    vectorstore = get_vectorstore(settings.vector_db_backend)
    vectorstore.set_persist_dir(settings.vectorstore_path)
    vectorstore.add(ids=ids, texts=texts, metadatas=metadatas, embeddings=embeddings)
    vectorstore.save(settings.vectorstore_path)
    embedding_backend.save(settings.vectorstore_path)

    manifest = {
        "embedding_backend": embedding_backend.name,
        "vector_db_backend": settings.vector_db_backend,
        "num_chunks": len(texts),
        "num_codes": len(codes),
        "manual_source": "pdf_first",
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    with (settings.vectorstore_path / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    app_logger.info(
        "Indexed %d chunks from %d diagnostic codes using backend=%s vector_db=%s",
        len(texts),
        len(codes),
        embedding_backend.name,
        settings.vector_db_backend,
    )


if __name__ == "__main__":
    build_index()
