"""
Retry Gemini chat calls on 429 / RESOURCE_EXHAUSTED with fast fail.

Keep waits short so Streamlit does not hang on "Thinking..." for too long.
"""

from __future__ import annotations

import re
import time
from typing import Any, List

from langchain_core.messages import BaseMessage


def _is_rate_limit_error(exc: BaseException) -> bool:
    s = str(exc).lower()
    return "429" in s or "resource_exhausted" in s or (
        "quota" in s and "exceeded" in s
    )


def _retry_after_seconds(exc: BaseException) -> float | None:
    m = re.search(r"retry in ([\d.]+)\s*s", str(exc), re.I)
    if m:
        # Honor provider hint but keep UX responsive.
        return min(12.0, float(m.group(1)) + 1.0)
    return None


def invoke_chat_with_retry(
    llm: Any,
    messages: List[BaseMessage],
    *,
    max_attempts: int = 3,
) -> Any:
    """
    Invoke LangChain chat model with short retries on rate limits.
    Raises last error if non-rate-limit or attempts exhausted.
    """
    last: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return llm.invoke(messages)
        except BaseException as e:
            last = e
            if not _is_rate_limit_error(e):
                raise
            if attempt >= max_attempts - 1:
                break
            wait = _retry_after_seconds(e)
            if wait is None:
                wait = min(8.0, 1.5 + (attempt * 2.0))
            time.sleep(wait)
    assert last is not None
    raise last
