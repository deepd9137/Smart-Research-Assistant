from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import Settings


def make_embeddings(settings: Settings) -> GoogleGenerativeAIEmbeddings:
    if not settings.google_api_key:
        raise ValueError("GOOGLE_API_KEY is required for embeddings.")
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
    )
