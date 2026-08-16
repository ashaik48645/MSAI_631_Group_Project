"""
rag_pipeline.py
MSAI 631 Course Support RAG Chatbot
Contributor: Ashraf Mohammad

Orchestrates the full Retrieval-Augmented Generation (RAG) pipeline:

    documents -> load -> clean/chunk -> embed -> index (vector store)
    question  -> embed -> retrieve top-k chunks -> LLM answer w/ sources

This module exposes the CourseSupportRAG class that the Gradio interface
(app.py) and the tests (test_rag_chatbot.py) depend on. It reuses:
  * document_loader.load_document  (Abdul: loading)
  * text_processing.clean_text / chunk_text  (Abdul: cleaning/chunking)
  * embeddings.Embedder            (Ashraf: MiniLM embeddings)
  * vector_store.VectorStore       (Ashraf: FAISS index)
  * llm_handler.LLMHandler         (Ashraf: answer generation)

Every component degrades gracefully if an optional dependency is missing,
so the pipeline runs end-to-end on a typical laptop without a GPU or any
API keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from document_loader import load_document
from text_processing import clean_text, chunk_text
from embeddings import Embedder
from vector_store import VectorStore, StoredChunk
from llm_handler import LLMHandler


@dataclass
class RAGAnswer:
    """Structured result returned by CourseSupportRAG.ask()."""
    answer: str
    sources: List[str] = field(default_factory=list)


class CourseSupportRAG:
    """End-to-end RAG pipeline for answering course-support questions."""

    def __init__(self, enable_llm: bool = True, top_k: int = 4):
        self.top_k = top_k
        self.embedder = Embedder()
        self.store = VectorStore(dimension=self.embedder.dimension)
        self.llm = LLMHandler(enable_llm=enable_llm)

    # --- Indexing side -------------------------------------------------
    def index_files(self, file_paths: List[str]) -> int:
        """Load, clean, chunk, embed, and index the given files.

        Returns the number of text chunks indexed.
        """
        total_chunks = 0
        for path in file_paths:
            doc = load_document(path)
            cleaned = clean_text(doc.text)
            chunks = chunk_text(cleaned, doc.source)
            if not chunks:
                continue
            texts = [c.text for c in chunks]
            embeddings = self.embedder.encode(texts)
            stored = [StoredChunk(text=c.text, source=c.source) for c in chunks]
            self.store.add(embeddings, stored)
            total_chunks += len(chunks)
        return total_chunks

    # --- Query side ----------------------------------------------------
    def ask(self, question: str) -> RAGAnswer:
        """Answer a question using retrieved context, with sources."""
        if not question or not question.strip():
            return RAGAnswer(answer="Please enter a question.", sources=[])

        if self.store.size == 0:
            return RAGAnswer(
                answer=("No documents have been indexed yet. Please upload and "
                        "index course documents, then ask again."),
                sources=[],
            )

        query_vec = self.embedder.encode([question])[0]
        results = self.store.search(query_vec, k=self.top_k)
        contexts = [chunk.text for chunk, _score in results]
        sources = []
        for chunk, _score in results:
            if chunk.source not in sources:
                sources.append(chunk.source)

        answer_text = self.llm.generate(question, contexts)
        if sources:
            answer_text = f"{answer_text}\n\nSources: {', '.join(sources)}"
        return RAGAnswer(answer=answer_text, sources=sources)
