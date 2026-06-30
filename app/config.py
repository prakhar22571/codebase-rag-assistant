from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM (Groq)
    groq_api_key: str
    groq_model: str = "llama-3.1-70b-versatile"
    groq_max_tokens: int = 2048
    groq_temperature: float = 0.1

    # Embeddings
    voyage_api_key: Optional[str] = None
    voyage_model: str = "voyage-code-3"
    embedding_dim: int = 1024
    bge_model_name: str = "BAAI/bge-m3"

    # Vector DB (Qdrant)
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str = "codebase_rag"

    # GitHub
    github_token: Optional[str] = None

    # Chunking
    max_tokens_per_chunk: int = 512
    chunk_overlap_tokens: int = 64

    # Retrieval
    top_k: int = 8

    @property
    def use_voyage(self) -> bool:
        return self.voyage_api_key is not None


settings = Settings()
