import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:

    # --- Paths ---
    data_dir: Path = field(default_factory=lambda: Path("data"))
    vectorstore_dir: Path = field(default_factory=lambda: Path("data/vectorstore"))

    # --- Embedding model ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- LLM ---
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = "gpt-3.5-turbo"
    hf_model_id: str = "google/flan-t5-base"

    # --- Chunking ---
    chunk_size: int = 800
    chunk_overlap: int = 100

    # --- Retrieval ---
    default_top_k: int = 4

    @property
    def use_openai(self) -> bool:
        return bool(self.openai_api_key)

    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()