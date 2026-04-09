"""Map simple chat dicts (Streamlit) to LangChain message types for the RAG prompt."""

from typing import Any, Dict, List, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def transcript_to_messages(history: Sequence[Dict[str, Any]]) -> List[BaseMessage]:
    """history items: {'role': 'user'|'assistant', 'content': str}"""
    out: List[BaseMessage] = []
    for turn in history:
        role = turn.get("role")
        content = turn.get("content", "")
        if role == "user":
            out.append(HumanMessage(content=str(content)))
        elif role == "assistant":
            out.append(AIMessage(content=str(content)))
    return out
