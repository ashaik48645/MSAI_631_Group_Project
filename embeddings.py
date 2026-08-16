"""
MSAI 631 Course Support RAG Chatbot
Contributor area: Ashraf Mohammad

Reference implementation for sentence embedding generation.
Ashraf should review, modify, test, and commit his own version.
"""

from typing import Sequence
import numpy as np

from config import EMBEDDING_MODEL_NAME


class EmbeddingModel:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        if hasattr(self.model, "encode_document"):
            vectors = self.model.encode_document(
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            vectors = self.model.encode(
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return np.asarray(vectors, dtype="float32")

    def encode_query(self, query: str) -> np.ndarray:
        if hasattr(self.model, "encode_query"):
            vector = self.model.encode_query(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        else:
            vector = self.model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        return np.asarray(vector, dtype="float32").reshape(1, -1)
