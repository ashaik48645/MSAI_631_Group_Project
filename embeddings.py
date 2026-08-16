"""
embeddings.py
MSAI 631 Course Support RAG Chatbot
Contributor: Ashraf Mohammad

Wraps a sentence-transformers MiniLM model to turn text into dense
vector embeddings for semantic search. MiniLM (all-MiniLM-L6-v2) is a
small, fast, free model that runs on a typical laptop with no API key.

If sentence-transformers is not installed or the model cannot be
downloaded (for example, with no internet), this module falls back to a
deterministic hashing-based embedding so the rest of the pipeline still
runs and can be tested offline. The fallback is clearly not as accurate
as MiniLM but keeps the system functional.
"""

from __future__ import annotations

import hashlib
import math
from typing import List


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_FALLBACK_DIM = 384  # MiniLM-L6-v2 output dimension


class Embedder:
    """Encodes text into fixed-length vectors using MiniLM when available."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._model = None
        self._dim = _FALLBACK_DIM
        self._using_fallback = False
        self._load_model()

    def _load_model(self) -> None:
        """Try to load MiniLM; fall back to a hashing embedder on failure."""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._dim = self._model.get_sentence_embedding_dimension()
        except Exception:
            # No sentence-transformers or no network: use offline fallback.
            self._model = None
            self._using_fallback = True

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def using_fallback(self) -> bool:
        return self._using_fallback

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Return one embedding vector per input text."""
        if isinstance(texts, str):
            texts = [texts]
        if self._model is not None:
            vectors = self._model.encode(texts, normalize_embeddings=True)
            return [list(map(float, v)) for v in vectors]
        return [self._fallback_encode(t) for t in texts]

    def _fallback_encode(self, text: str) -> List[float]:
        """Deterministic bag-of-words hashing embedding, L2-normalized.

        Not semantically strong, but stable and offline. Each token is
        hashed into a bucket; the resulting vector is normalized so cosine
        similarity behaves sensibly.
        """
        vec = [0.0] * self._dim
        for token in text.lower().split():
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            vec[h % self._dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec
