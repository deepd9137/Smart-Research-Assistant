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
    splitter = make_splitter(settings)
    return splitter.split_documents(documents)
