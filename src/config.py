"""
config.py — Central configuration using Pydantic Settings
Reads from .env in the project root (one level above src/)
"""

import os
from pathlib import Path
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Always resolve .env relative to this file's parent (project root)
# Works no matter which directory you run python from
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── API Keys ───────────────────────────────────────────────
    openai_api_key: str     = Field(default="")
    anthropic_api_key: str  = Field(default="")
    google_api_key: str     = Field(default="")
    cohere_api_key: str     = Field(default="")

    # ── Vector DB ──────────────────────────────────────────────
    qdrant_host: str        = "localhost"
    qdrant_port: int        = 6333
    qdrant_collection: str  = "rag_collection"

    # ── Embeddings ─────────────────────────────────────────────
    embedding_model: str    = "BAAI/bge-small-en-v1.5"
    embedding_provider: str = "huggingface"        # huggingface | openai

    # ── LLM ───────────────────────────────────────────────────
    llm_provider: str       = "google"             # google | openai | anthropic
    llm_model: str          = "gemini-1.5-flash"
    llm_temperature: float  = 0.0
    llm_max_tokens: int     = 1024

    # ── Retrieval ─────────────────────────────────────────────
    retriever_k: int        = 5
    use_hybrid_search: bool = True
    use_reranker: bool      = False   # set True only if you have Cohere key
    reranker_model: str     = "rerank-english-v3.0"

    # ── Chunking ──────────────────────────────────────────────
    chunk_size: int         = 512
    chunk_overlap: int      = 64
    chunk_strategy: str     = "recursive"

    # ── App ───────────────────────────────────────────────────
    app_host: str           = "0.0.0.0"
    app_port: int           = 8000
    log_level: str          = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

# ── Inject into environment (required by google-genai SDK) ─────────────────
if settings.google_api_key:
    os.environ["GOOGLE_API_KEY"] = settings.google_api_key
else:
    import warnings
    warnings.warn(
        f"\n\n*** GOOGLE_API_KEY is empty! ***\n"
        f"    .env file looked for at: {ENV_FILE}\n"
        f"    Make sure {ENV_FILE} exists and contains: GOOGLE_API_KEY=AIza...\n"
    )