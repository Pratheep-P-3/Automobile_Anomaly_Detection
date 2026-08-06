"""Embedding backends for the RAG pipeline.

Default backend is a TF-IDF vectorizer (scikit-learn) so the whole assistant works fully
offline with no API key and no model downloads - useful for demos, grading, and CI.
An OpenAI embedding backend is available for higher-quality semantic retrieval when
OPENAI_API_KEY is configured and USE_OPENAI_EMBEDDINGS=true.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Protocol

import numpy as np


class EmbeddingBackend(Protocol):
    name: str

    def fit(self, texts: list[str]) -> None: ...

    def transform(self, texts: list[str]) -> np.ndarray: ...

    def save(self, path: Path) -> None: ...

    def load(self, path: Path) -> None: ...


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class TfidfEmbeddingBackend:
    """Offline embedding backend using scikit-learn TF-IDF + SVD-free cosine similarity."""

    name = "tfidf"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=4096,
        )
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        self.vectorizer.fit(texts)
        self._fitted = True

    def transform(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbeddingBackend must be fit() before transform().")
        matrix = self.vectorizer.transform(texts).toarray().astype("float32")
        return _l2_normalize(matrix)

    def save(self, path: Path) -> None:
        with (path / "tfidf_vectorizer.pkl").open("wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self, path: Path) -> None:
        with (path / "tfidf_vectorizer.pkl").open("rb") as f:
            self.vectorizer = pickle.load(f)
        self._fitted = True


class OpenAIEmbeddingBackend:
    """Embedding backend backed by the OpenAI embeddings API."""

    name = "openai"

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()
        return self._client

    def fit(self, texts: list[str]) -> None:
        # No fitting required for a hosted embedding model.
        return None

    def transform(self, texts: list[str]) -> np.ndarray:
        client = self._get_client()
        # Batch to stay well within request size limits.
        vectors: list[list[float]] = []
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = client.embeddings.create(model=self.model, input=batch)
            vectors.extend([item.embedding for item in response.data])
        matrix = np.array(vectors, dtype="float32")
        return _l2_normalize(matrix)

    def save(self, path: Path) -> None:
        with (path / "openai_embedding_config.pkl").open("wb") as f:
            pickle.dump({"model": self.model}, f)

    def load(self, path: Path) -> None:
        with (path / "openai_embedding_config.pkl").open("rb") as f:
            data = pickle.load(f)
        self.model = data["model"]


def get_embedding_backend(use_openai: bool, openai_model: str | None = None) -> EmbeddingBackend:
    if use_openai:
        return OpenAIEmbeddingBackend(model="text-embedding-3-small")
    return TfidfEmbeddingBackend()
