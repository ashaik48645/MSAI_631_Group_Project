"""
MSAI 631 Course Support RAG Chatbot
Contributor area: Jyothirmayi Sunkara

Lightweight tests that run without downloading large Hugging Face models.
Jyothirmayi should add interface, fallback, and end-to-end tests.
"""

from pathlib import Path

from document_loader import load_document
from text_processing import chunk_text, clean_text


def test_clean_text_collapses_whitespace():
    text = "Hello     world.\n\n\n\nSecond paragraph."
    cleaned = clean_text(text)
    assert "Hello world." in cleaned
    assert "\n\n\n" not in cleaned


def test_chunk_text_produces_overlap():
    text = " ".join(["course"] * 400)
    chunks = chunk_text(text, "sample.txt", chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(chunk.source == "sample.txt" for chunk in chunks)


def test_load_txt(tmp_path: Path):
    path = tmp_path / "sample.txt"
    path.write_text("The design document should describe inputs and outputs.", encoding="utf-8")
    doc = load_document(path)
    assert doc.source == "sample.txt"
    assert "inputs and outputs" in doc.text


def test_rejects_unsupported_extension(tmp_path: Path):
    path = tmp_path / "sample.csv"
    path.write_text("a,b,c", encoding="utf-8")

    try:
        load_document(path)
        assert False, "Expected ValueError"
    except ValueError:
        pass
