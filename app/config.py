from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    google_api_key: str = ""
    tavily_api_key: str = ""

    pinecone_api_key: str = ""
    pinecone_index_name: str = "smart-research-assistant"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    chunk_size: int = 1000
    chunk_overlap: int = 200

    top_k: int = 5
    max_l2_distance: float = 0.55

    chat_model: str = "gemini-2.5-flash"
    embedding_model: str = "gemini-embedding-001"


def get_settings() -> Settings:
    return Settings()
