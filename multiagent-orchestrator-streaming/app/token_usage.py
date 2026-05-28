"""
Token usage aggregation for the orchestrator stream.

Emits a `usage` SSE event with token counts aggregated across every LLM
call in the graph (orchestrator, evaluator, classifier, sub-agents) via
UsageMetadataCallbackHandler propagated through the LangChain config.

Independent of MLflow. Works whether MLFLOW_ENABLED is true or false.
The autolog in app.tracing populates the MLflow trace UI in parallel;
this module populates the API response.
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import UsageMetadataCallbackHandler
from pydantic import BaseModel


class TokenUsage(BaseModel):
    """Per-request token usage across all LLM calls in the graph.

    `total_tokens` = `input_tokens` + `output_tokens`.
    `per_model` breaks down usage by model ID so callers can attribute
    cost to individual sub-agents in a multi-model graph.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
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
    total_in = out = 0
    for m in cb.usage_metadata.values():
        total_in += m.get("input_tokens", 0) or 0
        out += m.get("output_tokens", 0) or 0
    return TokenUsage(
        input_tokens=total_in,
        output_tokens=out,
        total_tokens=total_in + out,
        per_model=dict(cb.usage_metadata),
    )


def usage_event(cb: UsageMetadataCallbackHandler) -> dict:
    """Build the SSE payload for the `usage` event.

    Emit immediately before the terminal ``done`` event so clients can
    record cost-per-request before tearing down the EventSource.
    """
    return {"type": "usage", "usage": aggregate(cb).model_dump()}
