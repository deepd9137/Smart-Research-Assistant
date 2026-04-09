"""
Citation helpers: map retrieved chunks to human-readable references.

We label each retrieved chunk with [1], [2], ... in the LLM prompt and ask the model
to cite those labels in the answer. The UI maps labels back to document + page.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from langchain_core.documents import Document


@dataclass
class CitationRef:
    """One line in the 'References' section of the UI."""

    index: int
    document_name: str
    page: int
    section: str | None  # optional heading if we add it later; None for PDFs


def document_label(doc: Document, fallback_index: int) -> str:
    """Stable human-readable name for a chunk."""
    name = doc.metadata.get("document_name") or doc.metadata.get("source", "document")
    if isinstance(name, str) and "/" in name:
        # Show basename only in UI
        name = name.rsplit("/", 1)[-1]
    page = doc.metadata.get("page", "?")
    return f"{name}, p. {page}"


def build_numbered_context(docs: List[Document]) -> Tuple[str, List[CitationRef]]:
    """
    Build prompt context with [n] prefixes and parallel citation list.

    Returns:
        context_block: text for the LLM
        refs: metadata for the sidebar / references UI
    """
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
    """Map citation index -> 'filename.pdf, p. 3' for Streamlit."""
    out: Dict[int, str] = {}
    for r in refs:
        out[r.index] = f"{r.document_name}, p. {r.page}"
    return out
