"""
MSAI 631 Course Support RAG Chatbot
Contributor: Abdul Kareem Shaik

Cleans extracted text and divides it into overlapping chunks for retrieval.
"""

import re
from dataclasses import dataclass
from typing import Iterable, List

from config import CHUNK_OVERLAP, CHUNK_SIZE
from document_loader import LoadedDocument


@dataclass
class TextChunk:
    source: str
    chunk_id: int
    text: str


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[TextChunk]:
    """
    Character-based overlapping chunker.

    It tries to end chunks on paragraph/sentence/space boundaries when possible.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    text = clean_text(text)
    if not text:
        return []

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        target_end = min(start + chunk_size, len(text))
        end = target_end

        if target_end < len(text):
            search_start = start + max(chunk_size // 2, 1)
            candidates = [
                text.rfind("\n\n", search_start, target_end),
                text.rfind(". ", search_start, target_end),
                text.rfind(" ", search_start, target_end),
            ]
            best = max(candidates)
            if best > start:
                end = best + (2 if text[best:best + 2] in {". ", "\n\n"} else 1)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(TextChunk(source=source, chunk_id=chunk_id, text=chunk))
            chunk_id += 1

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def documents_to_chunks(documents: Iterable[LoadedDocument]) -> List[TextChunk]:
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_text(doc.text, doc.source))
    return all_chunks
