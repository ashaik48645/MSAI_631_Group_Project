"""
MSAI 631 Course Support RAG Chatbot
Contributor area: Ashraf Mohammad
Final integration: Abdul Kareem Shaik

Reference implementation of the Retrieval-Augmented Generation pipeline.
Ashraf should review, modify, test, and commit his own RAG implementation.
Abdul can integrate the final reviewed modules.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from config import MIN_SIMILARITY, TOP_K
from document_loader import load_documents
from embeddings import EmbeddingModel
from llm_handler import LocalLLM
from text_processing import TextChunk, documents_to_chunks
from vector_store import FaissVectorStore, SearchResult


@dataclass
class RAGAnswer:
    answer: str
    sources: List[SearchResult]


class CourseSupportRAG:
    def __init__(self, enable_llm: bool = True):
        self.embedder = EmbeddingModel()
        self.store = FaissVectorStore()
        self.llm = LocalLLM() if enable_llm else None
        self.chunks: List[TextChunk] = []

    def index_files(self, file_paths: Iterable[str | Path]) -> int:
        documents = load_documents(file_paths)
        self.chunks = documents_to_chunks(documents)

        if not self.chunks:
            raise ValueError("No readable text was found in the uploaded documents")

        vectors = self.embedder.encode_documents([c.text for c in self.chunks])
        self.store.build(vectors, self.chunks)
        return len(self.chunks)

    def retrieve(self, question: str, top_k: int = TOP_K) -> List[SearchResult]:
        if not question.strip():
            return []
        query_vector = self.embedder.encode_query(question)
        return self.store.search(query_vector, top_k=top_k)

    @staticmethod
    def _format_context(results: List[SearchResult]) -> str:
        blocks = []
        for i, result in enumerate(results, start=1):
            blocks.append(
                f"[Source {i}: {result.chunk.source}; "
                f"chunk {result.chunk.chunk_id}; similarity {result.score:.3f}]\n"
                f"{result.chunk.text}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _source_footer(results: List[SearchResult]) -> str:
        lines = []
        seen = set()
        for result in results:
            key = (result.chunk.source, result.chunk.chunk_id)
            if key not in seen:
                lines.append(
                    f"- {result.chunk.source}, chunk {result.chunk.chunk_id} "
                    f"(similarity {result.score:.3f})"
                )
                seen.add(key)
        return "\n".join(lines)

    def ask(self, question: str) -> RAGAnswer:
        question = question.strip()
        if not question:
            return RAGAnswer(
                "Please enter a question about the uploaded course documents.",
                [],
            )

        if self.store.index is None:
            return RAGAnswer(
                "Please upload and index course documents before asking a question.",
                [],
            )

        results = self.retrieve(question)

        if not results or results[0].score < MIN_SIMILARITY:
            return RAGAnswer(
                "I could not find enough information in the uploaded course documents. "
                "Please review the assignment instructions or contact the instructor.",
                results,
            )

        context = self._format_context(results)

        if self.llm is not None:
            try:
                answer = self.llm.generate(question, context)
                if answer:
                    answer += "\n\nSources:\n" + self._source_footer(results)
                    return RAGAnswer(answer, results)
            except Exception:
                # Retrieval-only fallback keeps the prototype usable if the LLM
                # is too large for the current Colab session.
                pass

        best = results[0]
        answer = (
            "The most relevant passage I found is:\n\n"
            f"{best.chunk.text}\n\n"
            "Source:\n"
            f"- {best.chunk.source}, chunk {best.chunk.chunk_id} "
            f"(similarity {best.score:.3f})"
        )
        return RAGAnswer(answer, results)
