"""
llm_handler.py
MSAI 631 Course Support RAG Chatbot
Contributor: Ashraf Mohammad

Generates a natural-language answer from a question and the retrieved
context passages. Uses a small, free Hugging Face text2text model
(google/flan-t5-base) when the transformers library and the model are
available; no API key is required.

If transformers is not installed or the model cannot be loaded, this
handler falls back to an extractive answer that returns the most relevant
retrieved passage. This guarantees the RAG pipeline always produces a
grounded response, satisfying the assignment's requirement that the code
run without errors even on a laptop with no GPU.
"""

from __future__ import annotations

from typing import List


DEFAULT_MODEL = "google/flan-t5-base"


class LLMHandler:
    """Wraps a small seq2seq LLM with a safe extractive fallback."""

    def __init__(self, model_name: str = DEFAULT_MODEL, enable_llm: bool = True):
        self.model_name = model_name
        self.enable_llm = enable_llm
        self._pipe = None
        self._using_fallback = True
        if enable_llm:
            self._load()

    def _load(self) -> None:
        try:
            from transformers import pipeline
            self._pipe = pipeline("text2text-generation", model=self.model_name)
            self._using_fallback = False
        except Exception:
            self._pipe = None
            self._using_fallback = True

    @property
    def using_fallback(self) -> bool:
        return self._using_fallback

    def build_prompt(self, question: str, contexts: List[str]) -> str:
        """Construct a grounded RAG prompt from the question and context."""
        context_block = "\n\n".join(f"- {c}" for c in contexts)
        return (
            "You are a helpful course-support assistant. Answer the question "
            "using ONLY the context below. If the answer is not in the context, "
            "say you do not have that information.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    def generate(self, question: str, contexts: List[str]) -> str:
        """Return an answer grounded in the provided context passages."""
        if not contexts:
            return ("I do not have any indexed course documents yet. "
                    "Please upload and index documents first.")

        if self._pipe is not None:
            prompt = self.build_prompt(question, contexts)
            out = self._pipe(prompt, max_length=256, truncation=True)
            return out[0]["generated_text"].strip()

        # Extractive fallback: return the most relevant passage, trimmed.
        best = contexts[0].strip()
        snippet = best if len(best) <= 500 else best[:500] + "..."
        return ("Based on the most relevant course material I found:\n\n"
                f"{snippet}")
