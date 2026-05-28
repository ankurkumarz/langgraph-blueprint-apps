"""
Token usage aggregation for the orchestrator stream.

Emits a `usage` SSE event with decomposed token counts (fresh input,
cache read, cache creation, output) per model. Reads LangChain's
`usage_metadata` via UsageMetadataCallbackHandler, which captures every
LLM call in the graph — orchestrator, evaluator, classifier, sub-agents.

Independent of MLflow. Works whether MLFLOW_ENABLED is true or false.
The autolog in app.tracing populates the MLflow trace UI in parallel;
this module populates the API response.
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import UsageMetadataCallbackHandler
from pydantic import BaseModel


class TokenUsage(BaseModel):
    """Per-request token usage, decomposed for correct cost math.

    `cache_read` and `cache_creation` are SUBSETS of the provider's
    `input_tokens` — so `fresh_input = input_tokens - cache_read - cache_creation`.
    Keeping the buckets separate lets downstream billing apply the
    discounted cache-read rate without re-deriving it.

    Gemini reports `cache_read` (implicit caching, no creation cost).
    Bedrock Claude reports both `cache_read` and `cache_creation`.
    OpenAI reports `cache_read` only. The model handles all three shapes.
    """

    fresh_input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    output_tokens: int = 0
    per_model: dict[str, dict[str, Any]] = {}


def make_callback() -> UsageMetadataCallbackHandler:
    """Return a fresh callback handler scoped to one request.

    Pass to the graph as ``config={"callbacks": [cb]}`` — it propagates
    through ``astream`` into every nested LLM call, including sub-agents
    invoked via ``asyncio.gather`` or ToolNode parallel dispatch.
    """
    return UsageMetadataCallbackHandler()


def aggregate(cb: UsageMetadataCallbackHandler) -> TokenUsage:
    """Roll the callback's per-model usage into a TokenUsage."""
    total_in = out = cache_r = cache_c = 0
    for m in cb.usage_metadata.values():
        total_in += m.get("input_tokens", 0) or 0
        out += m.get("output_tokens", 0) or 0
        details = m.get("input_token_details") or {}
        cache_r += details.get("cache_read", 0) or 0
        cache_c += details.get("cache_creation", 0) or 0
    return TokenUsage(
        fresh_input_tokens=max(0, total_in - cache_r - cache_c),
        cache_read_tokens=cache_r,
        cache_creation_tokens=cache_c,
        output_tokens=out,
        per_model=dict(cb.usage_metadata),
    )


def usage_event(cb: UsageMetadataCallbackHandler) -> dict:
    """Build the SSE payload for the `usage` event.

    Emit immediately before the terminal ``done`` event so clients can
    record cost-per-request before tearing down the EventSource.
    """
    return {"type": "usage", "usage": aggregate(cb).model_dump()}
