"""Retriever: turns a diagnostic code / symptom query into ranked, cited manual chunks."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from app.core.config import settings
from app.rag.embeddings import OpenAIEmbeddingBackend, TfidfEmbeddingBackend
from app.rag.keyword_search import KeywordSearchAgent
from app.rag.vectorstore import SearchResult, get_vectorstore

CODE_PATTERN = re.compile(r"\b([PBCU]0?\d{3,4})\b", re.IGNORECASE)


@dataclass
class RetrievedChunk:
    chunk_id: str
    score: float
    text: str
    code: str
    title: str
    system: str
    severity: str
    doc_file: str
    heading: str


def extract_code(query: str) -> str | None:
    match = CODE_PATTERN.search(query.strip())
    if not match:
        return None
    code = match.group(1).upper()
    # Normalize e.g. "P300" -> "P0300"
    letter, digits = code[0], code[1:]
    digits = digits.zfill(4)
    return f"{letter}{digits}"


class Retriever:
    """Loads the persisted vector index and serves top-k retrieval for a query."""

    def __init__(self) -> None:
        self._loaded = False
        self._embedding_backend = None
        self._vectorstore = None
        self._keyword_agent = KeywordSearchAgent()

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        manifest_path = settings.vectorstore_path / "manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(
                "Vector index not found. Run `python -m app.rag.ingest` to build it first."
            )
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        embedding_backend_name = manifest.get("embedding_backend", "tfidf")
        if embedding_backend_name == "openai":
            self._embedding_backend = OpenAIEmbeddingBackend()
        else:
            self._embedding_backend = TfidfEmbeddingBackend()
        self._embedding_backend.load(settings.vectorstore_path)

        self._vectorstore = get_vectorstore(manifest.get("vector_db_backend", "faiss"))
        self._vectorstore.load(settings.vectorstore_path)
        self._keyword_agent.load(self._vectorstore.documents())
        self._loaded = True

    def retrieve(self, query: str, top_k: int | None = None, code_filter: str | None = None) -> list[RetrievedChunk]:
        self._ensure_loaded()
        top_k = top_k or settings.top_k

        query_embedding = self._embedding_backend.transform([query])[0]
        # Fetch a wider pool so we can return all sections of a matched code
        # (manuals typically have 6-10 sections per code).
        pool_size = max(top_k * 5, 50)
        vector_results: list[SearchResult] = self._vectorstore.search(query_embedding, pool_size)
        keyword_results: list[SearchResult] = self._keyword_agent.search(query, pool_size)
        raw_results = self._merge_hybrid_results(vector_results, keyword_results)

        detected_code = code_filter or extract_code(query)

        def to_chunk(result: SearchResult) -> RetrievedChunk:
            meta = result.metadata
            return RetrievedChunk(
                chunk_id=result.chunk_id,
                score=result.score,
                text=result.text,
                code=meta.get("code", ""),
                title=meta.get("title", ""),
                system=meta.get("system", ""),
                severity=meta.get("severity", ""),
                doc_file=meta.get("doc_file", ""),
                heading=meta.get("heading", ""),
            )

        all_chunks = [to_chunk(r) for r in raw_results]

        if detected_code:
            code_matches = [c for c in all_chunks if c.code.upper() == detected_code.upper()]
            if code_matches:
                # Return ALL sections for an exact code match so the LLM gets full
                # manual context (Possible Causes, Diagnostic Steps, Recommended
                # Service Actions, etc.) not just the top-scored ones.
                return code_matches

        return all_chunks[:top_k]

    def _merge_hybrid_results(
        self,
        vector_results: list[SearchResult],
        keyword_results: list[SearchResult],
    ) -> list[SearchResult]:
        vector_scores = {result.chunk_id: max(result.score, 0.0) for result in vector_results}
        keyword_scores = self._bound_keyword_scores(keyword_results)
        by_id = {result.chunk_id: result for result in vector_results}
        by_id.update({result.chunk_id: result for result in keyword_results})

        merged: list[SearchResult] = []
        for chunk_id, result in by_id.items():
            score = (
                settings.vector_search_weight * vector_scores.get(chunk_id, 0.0)
                + settings.keyword_search_weight * keyword_scores.get(chunk_id, 0.0)
            )
            merged.append(
                SearchResult(
                    chunk_id=chunk_id,
                    score=score,
                    text=result.text,
                    metadata=result.metadata,
                )
            )

        merged.sort(key=lambda result: result.score, reverse=True)
        return merged

    def _bound_keyword_scores(self, results: list[SearchResult]) -> dict[str, float]:
        return {result.chunk_id: result.score / (result.score + 20.0) for result in results if result.score > 0}


# Module-level singleton so FastAPI/Streamlit share one loaded index per process.
retriever = Retriever()
