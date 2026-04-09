"""
Central configuration loaded from environment variables.
Tune retrieval thresholds after inspecting similarity scores for your corpus.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    google_api_key: str = ""
    tavily_api_key: str = ""

    # --- Pinecone vector DB ---
    pinecone_api_key: str = ""
    pinecone_index_name: str = "smart-research-assistant"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # --- Chunking (see README for reasoning) ---
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # --- Retrieval ---
    top_k: int = 5
    # We convert Pinecone relevance (higher is better) to pseudo-distance: 1 - relevance.
    # If the best pseudo-distance is above this, we treat matches as weak and may use web.
    max_l2_distance: float = 0.55

    # Gemini models (Developer API / AI Studio)
    # 1.5-era IDs often 404 on v1beta; use a current Flash from:
    # https://ai.google.dev/gemini-api/docs/models/gemini
    # Override with CHAT_MODEL in .env if needed (e.g. gemini-flash-latest).
    chat_model: str = "gemini-2.5-flash"
    # Embeddings: use names from https://ai.google.dev/gemini-api/docs/embeddings
    # `models/text-embedding-004` often 404s on embedContent v1beta; LangChain defaults to:
    embedding_model: str = "gemini-embedding-001"

def get_settings() -> Settings:
    return Settings()
