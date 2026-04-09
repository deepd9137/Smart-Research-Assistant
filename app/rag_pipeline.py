"""
RAG generation: retrieve → optional rerank (placeholder) → cite-grounded answer + confidence.
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.citations import CitationRef, build_numbered_context
from app.config import Settings
from app.gemini_invoke import invoke_chat_with_retry
from app.vector_store import similarity_search_with_scores


# How far we assume L2 distance can get before "unrelated" (for confidence scaling).
_DISTANCE_CAP = 2.0

# Model emits this alone on the final line when PDF context cannot answer the question.
# Agent then runs web search and appends a web-based answer (see agent.py).
DOC_INSUFFICIENT_MARKER = "__DOC_INSUFFICIENT__"


def l2_distances_to_confidence(distances: Sequence[float]) -> float:
    """
    Map best (minimum) L2 distance to a rough confidence in [0, 1].

    Lower distance → higher confidence. This is a heuristic, not a calibrated probability.
    """
    if not distances:
        return 0.0
    best = min(distances)
    # Linear decay: d=0 -> 1.0, d>=cap -> 0.0
    return max(0.0, min(1.0, 1.0 - (best / _DISTANCE_CAP)))


@dataclass
class RAGResult:
    answer: str
    citations: List[CitationRef]
    source_excerpts: List[Tuple[str, str]]  # (label, short excerpt) for optional UI
    confidence: float
    retrieved_distances: List[float]
    used_documents: bool


SYSTEM_PROMPT = """You are a careful research assistant. Answer using ONLY the provided context when document context is given.
Rules:
- If the context clearly supports an answer, respond helpfully and cite sources using bracket numbers like [1], [2] that match the context blocks. Do NOT use the marker line below.
- If the context is insufficient to answer the question, you MUST:
  1) Clearly state that the uploaded documents do not contain enough information (and briefly say what is missing, without inventing facts).
  2) Put ONLY the exact text __DOC_INSUFFICIENT__ on the very last line of your reply (no punctuation or spaces after it). An automated agent will then run web search to try to help.
- For web results (when labeled as WEB in the prompt), cite them as [W1], [W2] etc.
- Keep answers concise unless the user asks for detail."""


def _format_history(messages: Sequence[BaseMessage], max_turns: int = 6) -> str:
    """Last N conversational turns as plain text for the prompt."""
    if not messages:
        return ""
    recent = list(messages)[-max_turns:]
    parts: List[str] = []
    for m in recent:
        if isinstance(m, HumanMessage):
            parts.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            parts.append(f"Assistant: {m.content}")
    return "\n".join(parts)


def run_rag(
    query: str,
    vectorstore,
    settings: Settings,
    chat_history: Sequence[BaseMessage] | None = None,
    precomputed_pairs: List[Tuple[Document, float]] | None = None,
) -> RAGResult:
    """
    Retrieve top-k chunks, build grounded prompt, call Gemini.

    Pass `precomputed_pairs` when the caller already ran similarity search (avoids duplicate work).
    """
    if not settings.google_api_key:
        raise ValueError("GOOGLE_API_KEY is required.")

    pairs = precomputed_pairs
    if pairs is None:
        pairs = similarity_search_with_scores(vectorstore, query, k=settings.top_k)
    docs: List[Document] = [d for d, _ in pairs]
    distances: List[float] = [float(s) for _, s in pairs]
    confidence = l2_distances_to_confidence(distances)

    context_block, refs = build_numbered_context(docs)
    history_block = _format_history(chat_history or [])

    user_content = f"""Conversation so far (most recent last):
{history_block or "(no prior turns)"}

User question: {query}

Context from uploaded documents:
{context_block}

Answer the user. Use citations [1], [2] where appropriate.
If documents are insufficient, end with the marker line __DOC_INSUFFICIENT__ as instructed in the system rules."""

    llm = ChatGoogleGenerativeAI(
        model=settings.chat_model,
        google_api_key=settings.google_api_key,
        temperature=0.2,
    )
    response = invoke_chat_with_retry(
        llm,
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ],
    )
    answer = response.content if hasattr(response, "content") else str(response)

    # Short excerpts for optional "source highlights" in UI (first 220 chars per chunk)
    excerpts: List[Tuple[str, str]] = []
    for r, d in zip(refs, docs):
        label = f"[{r.index}]"
        excerpt = (d.page_content or "").strip().replace("\n", " ")[:220]
        if len((d.page_content or "")) > 220:
            excerpt += "…"
        excerpts.append((label, excerpt))

    return RAGResult(
        answer=answer,
        citations=refs,
        source_excerpts=excerpts,
        confidence=confidence,
        retrieved_distances=distances,
        used_documents=True,
    )


def split_rag_answer_for_web_followup(answer: str) -> Tuple[str, bool]:
    """
    If the model followed instructions, the answer ends with DOC_INSUFFICIENT_MARKER.
    Returns (visible_document_only_text, needs_web_followup).
    """
    text = (answer or "").strip()
    if not text:
        return "", True
    lines = text.splitlines()
    if lines and lines[-1].strip() == DOC_INSUFFICIENT_MARKER:
        body = "\n".join(lines[:-1]).strip()
        return body, True
    if text.endswith(DOC_INSUFFICIENT_MARKER):
        body = text[: -len(DOC_INSUFFICIENT_MARKER)].strip()
        return body, True
    return text, False


WEB_AFTER_GAP_SYSTEM = """You are a helpful research assistant.
The user already sees a separate "From your documents" section explaining their PDFs did not answer the question.
Your job: write ONLY the web-based answer — do NOT repeat, summarize, or quote that document section.
Answer using ONLY the web search snippets. Cite [W1], [W2]. Do not claim snippets came from the user's PDFs.
Start with substantive facts (e.g. names, dates), not with "The documents do not..." or similar."""


def run_web_after_doc_gap(
    query: str,
    web_snippets: List[str],
    settings: Settings,
    chat_history: Sequence[BaseMessage] | None = None,
) -> str:
    """Synthesize the web-only answer. Caller already showed the doc gap above — no repetition here."""
    if not settings.google_api_key:
        raise ValueError("GOOGLE_API_KEY is required.")

    web_block = "\n\n".join(
        f"[W{i}] {snippet}" for i, snippet in enumerate(web_snippets, start=1)
    )
    history_block = _format_history(chat_history or [])

    user_content = f"""Conversation so far:
{history_block or "(no prior turns)"}

User question: {query}

Web search results:
{web_block}

Give a direct answer from these results only. Cite [W1], [W2]. One short clause at the end like "(Sources: web search)" is enough — do not restate document limitations."""

    llm = ChatGoogleGenerativeAI(
        model=settings.chat_model,
        google_api_key=settings.google_api_key,
        temperature=0.3,
    )
    response = invoke_chat_with_retry(
        llm,
        [
            SystemMessage(content=WEB_AFTER_GAP_SYSTEM),
            HumanMessage(content=user_content),
        ],
    )
    return response.content if hasattr(response, "content") else str(response)


def run_web_augmented_answer(
    query: str,
    settings: Settings,
    web_snippets: List[str],
    chat_history: Sequence[BaseMessage] | None = None,
) -> str:
    """Synthesize an answer from Tavily snippets (already fetched)."""
    if not settings.google_api_key:
        raise ValueError("GOOGLE_API_KEY is required.")

    web_block = "\n\n".join(
        f"[W{i}] {snippet}" for i, snippet in enumerate(web_snippets, start=1)
    )
    history_block = _format_history(chat_history or [])

    user_content = f"""Conversation so far:
{history_block or "(no prior turns)"}

User question: {query}

Web search results (may be incomplete; verify critical facts elsewhere):
{web_block}

Answer using web results. Cite as [W1], [W2]. If results are weak, say so."""

    web_only_system = """You are a helpful assistant. Answer using the web search snippets in the user message.
Cite sources as [W1], [W2]. Do not use __DOC_INSUFFICIENT__ — you already have web context."""

    llm = ChatGoogleGenerativeAI(
        model=settings.chat_model,
        google_api_key=settings.google_api_key,
        temperature=0.3,
    )
    response = invoke_chat_with_retry(
        llm,
        [
            SystemMessage(content=web_only_system),
            HumanMessage(content=user_content),
        ],
    )
    return response.content if hasattr(response, "content") else str(response)
