"""
vector_store.py
MSAI 631 Course Support RAG Chatbot
Contributor: Ashraf Mohammad

A vector index that stores text-chunk embeddings and retrieves the most
similar chunks for a query. Uses FAISS (Facebook AI Similarity Search)
when it is installed, which is the standard, free library for efficient
similarity search. If FAISS is unavailable, it falls back to a pure
NumPy/Python cosine-similarity search so the pipeline still works.

Each stored item keeps the chunk text and its source filename so answers
can cite where information came from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class StoredChunk:
    text: str
    source: str


class VectorStore:
    """Stores embeddings + chunk metadata and does top-k similarity search."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self._chunks: List[StoredChunk] = []
        self._faiss_index = None
        self._np = None
        self._vectors = None  # fallback storage: numpy array or list
        self._use_faiss = self._init_faiss()

    def _init_faiss(self) -> bool:
        try:
            import faiss  # type: ignore
            import numpy as np
            self._faiss = faiss
            self._np = np
            # Inner product on normalized vectors == cosine similarity.
            self._faiss_index = faiss.IndexFlatIP(self.dimension)
            return True
        except Exception:
            try:
                import numpy as np
                self._np = np
            except Exception:
                self._np = None
            self._vectors = []
            return False

    @property
    def size(self) -> int:
        return len(self._chunks)

    def add(self, embeddings: List[List[float]], chunks: List[StoredChunk]) -> None:
        """Add a batch of embeddings with their matching chunk metadata."""
        if len(embeddings) != len(chunks):
            raise ValueError("embeddings and chunks must be the same length")
        self._chunks.extend(chunks)

        if self._use_faiss:
            arr = self._np.array(embeddings, dtype="float32")
            self._faiss_index.add(arr)
        elif self._np is not None:
            arr = self._np.array(embeddings, dtype="float32")
            if self._vectors is None or len(self._vectors) == 0:
                self._vectors = arr
            else:
                self._vectors = self._np.vstack([self._vectors, arr])
        else:
            # Pure-Python fallback
            self._vectors.extend(embeddings)

    def search(self, query_embedding: List[float], k: int = 4) -> List[Tuple[StoredChunk, float]]:
        """Return up to k (chunk, score) pairs most similar to the query."""
        if self.size == 0:
            return []
        k = min(k, self.size)

        if self._use_faiss:
            q = self._np.array([query_embedding], dtype="float32")
            scores, idxs = self._faiss_index.search(q, k)
            return [(self._chunks[i], float(scores[0][rank]))
                    for rank, i in enumerate(idxs[0]) if i != -1]

        if self._np is not None:
            q = self._np.array(query_embedding, dtype="float32")
            sims = self._vectors @ q  # cosine since vectors are normalized
            top = self._np.argsort(sims)[::-1][:k]
            return [(self._chunks[i], float(sims[i])) for i in top]

        # Pure-Python cosine
        def dot(a, b):
            return sum(x * y for x, y in zip(a, b))
        scored = [(self._chunks[i], dot(v, query_embedding))
                  for i, v in enumerate(self._vectors)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
