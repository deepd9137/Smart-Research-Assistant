from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf_file(path: str | Path, display_name: str | None = None) -> List[Document]:
    path = Path(path)
    name = display_name or path.name
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    for d in docs:
        d.metadata["document_name"] = name
        page_zero_based = d.metadata.get("page", 0)
        d.metadata["page"] = int(page_zero_based) + 1
        d.metadata["source"] = str(path)
    return docs


def load_many_pdfs(paths: List[Path], max_files: int = 5) -> List[Document]:
    if len(paths) > max_files:
        raise ValueError(f"At most {max_files} PDFs allowed; got {len(paths)}")
    all_docs: List[Document] = []
    for p in paths:
        all_docs.extend(load_pdf_file(p, display_name=p.name))
    return all_docs
