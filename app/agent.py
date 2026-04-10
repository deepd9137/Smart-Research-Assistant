from dataclasses import dataclass
from typing import List, Sequence

from langchain_core.messages import BaseMessage
from tavily import TavilyClient

from app.config import Settings
from app.rag_pipeline import (
    RAGResult,
    run_rag,
    run_web_after_doc_gap,
    run_web_augmented_answer,
    split_rag_answer_for_web_followup,
)
from app.vector_store import similarity_search_with_scores


@dataclass
class AgentAnswer:
    answer: str
    rag_result: RAGResult | None
    used_web: bool
    web_snippets: List[str]
    routing_reason: str
    show_document_provenance: bool = True


def _doc_gap_preamble_weak_retrieval() -> str:
    return (
        "The uploaded documents do not appear to contain enough relevant information "
        "to answer this question confidently (similarity to your PDFs was weak).\n\n"
    )


def _doc_gap_preamble_no_chunks() -> str:
    return (
        "The uploaded documents did not yield any usable passages for this question.\n\n"
    )


def _format_doc_plus_web(doc_section: str, web_section: str) -> str:
    return (
        "### From your documents\n"
        f"{doc_section.strip()}\n\n"
        "### From web search\n"
        f"{web_section.strip()}"
    )


def _fetch_tavily(query: str, settings: Settings, max_results: int = 4) -> List[str]:
    if not settings.tavily_api_key:
        return []
    if len((query or "").strip()) < 2:
        return []
    client = TavilyClient(api_key=settings.tavily_api_key)
    try:
        resp = client.search(query, max_results=max_results)
    except Exception:
        return []
    snippets: List[str] = []
    for item in resp.get("results", []) or []:
        title = item.get("title") or ""
        body = item.get("content") or ""
        url = item.get("url") or ""
        snippets.append(f"{title}\n{body}\nSource: {url}".strip())
    return snippets


def route_and_answer(
    query: str,
    vectorstore,
    settings: Settings,
    chat_history: Sequence[BaseMessage] | None = None,
) -> AgentAnswer:
    clean_query = (query or "").strip()
    if len(clean_query) < 2:
        return AgentAnswer(
            answer=(
                "I need a bit more detail to help. Try a full question, for example:\n\n"
                "- `Summarize the main findings in the uploaded PDFs`\n"
                "- `What does the document say about X?`\n"
                "- `Compare document A and B on Y`\n\n"
                "If you want a web-based answer, ask a specific question with at least a few words."
            ),
            rag_result=None,
            used_web=False,
            web_snippets=[],
            routing_reason="Query too short.",
            show_document_provenance=False,
        )

    if vectorstore is None:
        reason = "No document index loaded."
        web = _fetch_tavily(clean_query, settings)
        if not web:
            return AgentAnswer(
                answer="Please upload PDFs and click **Build knowledge base**, or set TAVILY_API_KEY for web-only mode.",
                rag_result=None,
                used_web=False,
                web_snippets=[],
                routing_reason=reason,
            )
        web_ans = run_web_augmented_answer(clean_query, settings, web, chat_history)
        preamble = (
            "No PDFs are indexed yet, so this answer is **not** from your uploads — "
            "only from web search:\n\n"
        )
        return AgentAnswer(
            answer=preamble + web_ans,
            rag_result=None,
            used_web=True,
            web_snippets=web,
            routing_reason=reason + " Web-only (no knowledge base).",
        )

    pairs = similarity_search_with_scores(vectorstore, clean_query, k=settings.top_k)
    if not pairs:
        reason = "Retriever returned no chunks."
        web = _fetch_tavily(clean_query, settings)
        if web:
            web_ans = run_web_augmented_answer(clean_query, settings, web, chat_history)
            ans = _doc_gap_preamble_no_chunks() + web_ans
        else:
            ans = (
                _doc_gap_preamble_no_chunks().strip()
                + "\n\nWeb search is not available (set **TAVILY_API_KEY**)."
            )
        return AgentAnswer(
            answer=ans,
            rag_result=None,
            used_web=bool(web),
            web_snippets=web,
            routing_reason=reason + (" Web follow-up." if web else ""),
        )

    distances = [float(s) for _, s in pairs]
    best = min(distances)

    if best > settings.max_l2_distance:
        reason = (
            f"Best retrieval L2 distance {best:.3f} exceeded threshold "
            f"{settings.max_l2_distance:.3f}; using web fallback."
        )
        web = _fetch_tavily(clean_query, settings)
        if web:
            web_ans = run_web_augmented_answer(clean_query, settings, web, chat_history)
            ans = _doc_gap_preamble_weak_retrieval() + web_ans
            return AgentAnswer(
                answer=ans,
                rag_result=None,
                used_web=True,
                web_snippets=web,
                routing_reason=reason + " Stated document gap; answered from web.",
            )
        rag = run_rag(
            clean_query, vectorstore, settings, chat_history, precomputed_pairs=pairs
        )
        doc_part, needs_follow = split_rag_answer_for_web_followup(rag.answer)
        rag.answer = doc_part
        suffix = (
            "\n\n*(Similarity to your PDFs was weak and web search is not configured, "
            "so only the document-grounded attempt above is shown.)*"
        )
        return AgentAnswer(
            answer=(doc_part or "") + suffix,
            rag_result=rag,
            used_web=False,
            web_snippets=[],
            routing_reason=reason + " Tavily not configured; best-effort RAG only.",
            show_document_provenance=False,
        )

    reason = f"Strong in-document match (best L2 distance={best:.3f})."
    rag = run_rag(clean_query, vectorstore, settings, chat_history, precomputed_pairs=pairs)
    doc_part, needs_web = split_rag_answer_for_web_followup(rag.answer)
    rag.answer = doc_part

    if needs_web:
        reason += " Model reported insufficient document context; attempting web follow-up."
        web = _fetch_tavily(clean_query, settings)
        if web:
            web_ans = run_web_after_doc_gap(clean_query, web, settings, chat_history)
            combined = _format_doc_plus_web(doc_part or "_No answer from PDFs._", web_ans)
            return AgentAnswer(
                answer=combined,
                rag_result=rag,
                used_web=True,
                web_snippets=web,
                routing_reason=reason,
                show_document_provenance=False,
            )
        note = (
            "\n\n---\n**Web search:** Not configured (`TAVILY_API_KEY`). "
            "Add a key to complete the agent fallback."
        )
        return AgentAnswer(
            answer=(doc_part or rag.answer) + note,
            rag_result=rag,
            used_web=False,
            web_snippets=[],
            routing_reason=reason + " Tavily missing.",
            show_document_provenance=False,
        )

    return AgentAnswer(
        answer=rag.answer,
        rag_result=rag,
        used_web=False,
        web_snippets=[],
        routing_reason=reason,
    )
