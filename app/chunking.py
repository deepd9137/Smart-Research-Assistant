"""
Text chunking for RAG.

RecursiveCharacterTextSplitter tries splits on paragraphs, newlines, then spaces — good
for keeping related sentences together vs fixed windows.
"""

from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config import Settings


def make_splitter(settings: Settings) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_documents(documents: List[Document], settings: Settings) -> List[Document]:
    """
    Split page-level documents into smaller chunks.
    Metadata (document_name, page, source) is copied onto each chunk by LangChain.
    """
    splitter = make_splitter(settings)
    return splitter.split_documents(documents)
