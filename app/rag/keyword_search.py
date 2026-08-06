"""Keyword-search agent used by the hybrid retriever."""
from __future__ import annotations

import math
import re
from collections import Counter

from app.rag.vectorstore import SearchResult

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text) if len(token) > 1]


class KeywordSearchAgent:
    """Small BM25-style agent for lexical retrieval over persisted chunks."""

    def __init__(self) -> None:
        self._documents: list[SearchResult] = []
        self._term_counts: list[Counter[str]] = []
        self._document_frequency: Counter[str] = Counter()
        self._avg_doc_len = 1.0

    def load(self, documents: list[SearchResult]) -> None:
        self._documents = documents
        self._term_counts = []
        self._document_frequency = Counter()

        total_len = 0
        for document in documents:
            tokens = _tokenize(self._searchable_text(document))
            counts = Counter(tokens)
            self._term_counts.append(counts)
            self._document_frequency.update(counts.keys())
            total_len += len(tokens)

        self._avg_doc_len = max(total_len / max(len(documents), 1), 1.0)

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        query_terms = _tokenize(query)
        if not query_terms or not self._documents:
            return []

        scores: list[tuple[float, SearchResult]] = []
        for document, counts in zip(self._documents, self._term_counts):
            score = self._bm25_score(query_terms, counts)
            if score > 0:
                scores.append((score, document))

        scores.sort(key=lambda item: item[0], reverse=True)
        return [
            SearchResult(
                chunk_id=document.chunk_id,
                score=score,
                text=document.text,
                metadata=document.metadata,
            )
            for score, document in scores[:top_k]
        ]

    def _bm25_score(self, query_terms: list[str], counts: Counter[str]) -> float:
        k1 = 1.5
        b = 0.75
        doc_len = sum(counts.values()) or 1
        total_docs = max(len(self._documents), 1)
        score = 0.0

        for term in query_terms:
            frequency = counts.get(term, 0)
            if frequency == 0:
                continue
            doc_frequency = self._document_frequency.get(term, 0)
            idf = math.log(1 + (total_docs - doc_frequency + 0.5) / (doc_frequency + 0.5))
            denominator = frequency + k1 * (1 - b + b * doc_len / self._avg_doc_len)
            score += idf * (frequency * (k1 + 1)) / denominator
        return score

    def _searchable_text(self, document: SearchResult) -> str:
        metadata = document.metadata
        symptoms = metadata.get("symptoms", [])
        symptom_text = " ".join(symptoms) if isinstance(symptoms, list) else str(symptoms)
        return " ".join(
            [
                metadata.get("code", ""),
                metadata.get("title", ""),
                metadata.get("system", ""),
                symptom_text,
                metadata.get("heading", ""),
                document.text,
            ]
        )