"""
Pinecone vector store for production-style cloud retrieval.

Metadata on each vector (document_name, page) is preserved for citations.
"""

import time
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from app.config import Settings


def _ensure_pinecone_index(settings: Settings, embeddings: Embeddings) -> None:
    if not settings.pinecone_api_key:
        raise ValueError("PINECONE_API_KEY is required.")

    pc = Pinecone(api_key=settings.pinecone_api_key)
    listed = pc.list_indexes()
    if hasattr(listed, "names"):
        existing = set(listed.names())
    else:
        existing = {idx.get("name") for idx in listed}
    if settings.pinecone_index_name in existing:
        return

    # Match index dimension to current embedding model.
    probe = embeddings.embed_query("dimension probe")
    dim = len(probe)
    pc.create_index(
        name=settings.pinecone_index_name,
        dimension=dim,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
        ),
    )
    # Wait for readiness once on first create.
    while True:
        status = pc.describe_index(settings.pinecone_index_name).status
        if status.get("ready"):
            break
        time.sleep(1.5)


def build_vector_store(
    documents: List[Document],
    embeddings: Embeddings,
    settings: Settings,
) -> PineconeVectorStore:
    """Create/update a Pinecone index from chunked documents."""
    if not documents:
        raise ValueError("No documents to index.")
    _ensure_pinecone_index(settings, embeddings)
    store = PineconeVectorStore(
        index_name=settings.pinecone_index_name,
        embedding=embeddings,
        pinecone_api_key=settings.pinecone_api_key,
    )
    store.add_documents(documents)
    return store


def save_vector_store(store: PineconeVectorStore, settings: Settings) -> None:
    """No-op for Pinecone (data is already persisted remotely)."""
    return None


def load_vector_store(
    embeddings: Embeddings,
    settings: Settings,
) -> Optional[PineconeVectorStore]:
    """Load Pinecone vector store handle if credentials are configured."""
    if not settings.pinecone_api_key:
        return None
    return PineconeVectorStore(
        index_name=settings.pinecone_index_name,
        embedding=embeddings,
        pinecone_api_key=settings.pinecone_api_key,
    )


def similarity_search_with_scores(
    store: PineconeVectorStore,
    query: str,
    k: int,
) -> List[Tuple[Document, float]]:
    """
    Return (document, pseudo-distance) where lower is better.

    Pinecone returns relevance in [0, 1] (higher better), so we map:
    pseudo_distance = 1 - relevance
    """
    pairs = store.similarity_search_with_relevance_scores(query, k=k)
    out: List[Tuple[Document, float]] = []
    for doc, relevance in pairs:
        rel = max(0.0, min(1.0, float(relevance)))
        out.append((doc, 1.0 - rel))
    return out
