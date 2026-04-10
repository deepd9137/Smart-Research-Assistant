from dataclasses import dataclass
from typing import Dict, List, Tuple

from langchain_core.documents import Document


@dataclass
class CitationRef:
    index: int
    document_name: str
    page: int
    section: str | None


def document_label(doc: Document, fallback_index: int) -> str:
    name = doc.metadata.get("document_name") or doc.metadata.get("source", "document")
    if isinstance(name, str) and "/" in name:
        name = name.rsplit("/", 1)[-1]
    page = doc.metadata.get("page", "?")
    return f"{name}, p. {page}"


def build_numbered_context(docs: List[Document]) -> Tuple[str, List[CitationRef]]:
    lines: List[str] = []
    refs: List[CitationRef] = []
    for i, doc in enumerate(docs, start=1):
        label = document_label(doc, i)
        lines.append(f"[{i}] ({label})\n{doc.page_content.strip()}")
        refs.append(
            CitationRef(
                index=i,
                document_name=str(doc.metadata.get("document_name", "Unknown")),
                page=int(doc.metadata.get("page", 0) or 0),
                section=doc.metadata.get("section"),
            )
        )
    return "\n\n".join(lines), refs


def refs_to_display_map(refs: List[CitationRef]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for r in refs:
        out[r.index] = f"{r.document_name}, p. {r.page}"
    return out
