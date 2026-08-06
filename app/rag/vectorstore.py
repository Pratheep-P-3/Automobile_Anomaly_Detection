"""Vector store abstraction with FAISS (default) and Chroma backends.

Both backends implement the same minimal interface so the RAG pipeline can switch
between them via the VECTOR_DB_BACKEND setting without any other code changes.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SearchResult:
    chunk_id: str
    score: float
    text: str
    metadata: dict = field(default_factory=dict)


class BaseVectorStore:
    def add(self, ids: list[str], texts: list[str], metadatas: list[dict], embeddings: np.ndarray) -> None:
        raise NotImplementedError

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[SearchResult]:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        raise NotImplementedError

    def load(self, path: Path) -> None:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def documents(self) -> list[SearchResult]:
        raise NotImplementedError

    def set_persist_dir(self, path: Path) -> None:
        """Optional hook for backends that need a storage location before add()."""
        return None


class FaissVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        self._index = None
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metadatas: list[dict] = []

    def add(self, ids: list[str], texts: list[str], metadatas: list[dict], embeddings: np.ndarray) -> None:
        import faiss

        dim = embeddings.shape[1]
        if self._index is None:
            self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings.astype("float32"))
        self._ids.extend(ids)
        self._texts.extend(texts)
        self._metadatas.extend(metadatas)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[SearchResult]:
        if self._index is None or self._index.ntotal == 0:
            return []
        query = query_embedding.astype("float32").reshape(1, -1)
        scores, indices = self._index.search(query, min(top_k, self._index.ntotal))
        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(
                SearchResult(
                    chunk_id=self._ids[idx],
                    score=float(score),
                    text=self._texts[idx],
                    metadata=self._metadatas[idx],
                )
            )
        return results

    def save(self, path: Path) -> None:
        import faiss

        path.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            faiss.write_index(self._index, str(path / "faiss.index"))
        with (path / "faiss_store.pkl").open("wb") as f:
            pickle.dump({"ids": self._ids, "texts": self._texts, "metadatas": self._metadatas}, f)

    def load(self, path: Path) -> None:
        import faiss

        index_file = path / "faiss.index"
        if index_file.exists():
            self._index = faiss.read_index(str(index_file))
        with (path / "faiss_store.pkl").open("rb") as f:
            data = pickle.load(f)
        self._ids = data["ids"]
        self._texts = data["texts"]
        self._metadatas = data["metadatas"]

    def count(self) -> int:
        return len(self._ids)

    def documents(self) -> list[SearchResult]:
        return [
            SearchResult(chunk_id=chunk_id, score=0.0, text=text, metadata=metadata)
            for chunk_id, text, metadata in zip(self._ids, self._texts, self._metadatas)
        ]


class ChromaVectorStore(BaseVectorStore):
    """Local, persistent Chroma collection (no external server required)."""

    def __init__(self, collection_name: str = "service_manuals") -> None:
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._persist_dir: Path | None = None

    def set_persist_dir(self, path: Path) -> None:
        self._persist_dir = path

    def _ensure_client(self, persist_dir: Path):
        if self._client is None:
            import chromadb

            self._client = chromadb.PersistentClient(path=str(persist_dir))
            self._collection = self._client.get_or_create_collection(self.collection_name)
        return self._collection

    def add(self, ids: list[str], texts: list[str], metadatas: list[dict], embeddings: np.ndarray) -> None:
        if self._persist_dir is None:
            raise RuntimeError("ChromaVectorStore requires set_persist_dir() before add().")
        collection = self._ensure_client(self._persist_dir)
        collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings.tolist())

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[SearchResult]:
        collection = self._collection
        if collection is None:
            return []
        result = collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)
        results: list[SearchResult] = []
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        for chunk_id, distance, text, metadata in zip(ids, distances, documents, metadatas):
            # Chroma returns L2 distance by default; convert to a similarity-like score.
            score = 1.0 / (1.0 + distance)
            results.append(SearchResult(chunk_id=chunk_id, score=score, text=text, metadata=metadata or {}))
        return results

    def save(self, path: Path) -> None:
        # Chroma persists automatically to its PersistentClient path; nothing extra needed
        # beyond ensuring the client/collection has been created against this path.
        self._persist_dir = path
        self._ensure_client(path)

    def load(self, path: Path) -> None:
        self._persist_dir = path
        self._ensure_client(path)

    def count(self) -> int:
        if self._collection is None:
            return 0
        return self._collection.count()

    def documents(self) -> list[SearchResult]:
        if self._collection is None:
            return []
        result = self._collection.get(include=["documents", "metadatas"])
        ids = result.get("ids", [])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])
        return [
            SearchResult(chunk_id=chunk_id, score=0.0, text=text, metadata=metadata or {})
            for chunk_id, text, metadata in zip(ids, documents, metadatas)
        ]


def get_vectorstore(backend_name: str) -> BaseVectorStore:
    if backend_name == "chroma":
        return ChromaVectorStore()
    return FaissVectorStore()
