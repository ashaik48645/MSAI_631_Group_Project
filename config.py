"""
MSAI 631 Course Support RAG Chatbot
Contributor area: Abdul Kareem Shaik

Central configuration used by all modules.
"""

from pathlib import Path

PROJECT_NAME = "MSAI 631 Course Support RAG Chatbot"

# Models proposed by the group.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PRIMARY_LLM_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
BACKUP_LLM_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Retrieval / chunking settings.
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
TOP_K = 4
MIN_SIMILARITY = 0.25

# Generation settings.
MAX_NEW_TOKENS = 220
TEMPERATURE = 0.2

# Supported document extensions.
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

# Local folders.
BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DOCS_DIR = BASE_DIR / "sample_docs"
