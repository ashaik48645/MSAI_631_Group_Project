"""
MSAI 631 Course Support RAG Chatbot
Contributor area: Ashraf Mohammad

Reference FAISS vector index.
Ashraf should review, modify, test, and commit his own version.
"""

from dataclasses import dataclass
from typing import List, Sequence
import numpy as np

from text_processing import TextChunk


@dataclass
class SearchResult:
    chunk: TextChunk
    score: float


class FaissVectorStore:
    def __init__(self):
        self.index = None
        self.chunks: List[TextChunk] = []

    def build(self, embeddings: np.ndarray, chunks: Sequence[TextChunk]) -> None:
        import faiss

        embeddings = np.asarray(embeddings, dtype="float32")
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a two-dimensional array")
        if len(embeddings) != len(chunks):
            raise ValueError("Number of embeddings must equal number of chunks")
        if len(chunks) == 0:
            raise ValueError("Cannot build an index with zero chunks")

        # EmbeddingModel already normalizes, but normalize again defensively.
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.chunks = list(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> List[SearchResult]:
        if self.index is None:
            raise RuntimeError("Vector index has not been built")

        import faiss

        query_embedding = np.asarray(query_embedding, dtype="float32")
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        faiss.normalize_L2(query_embedding)

        k = min(max(int(top_k), 1), len(self.chunks))
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append(SearchResult(self.chunks[int(idx)], float(score)))
        return results
