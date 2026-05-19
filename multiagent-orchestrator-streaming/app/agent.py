"""
Backward-compatibility shim.

All implementation has moved to:
  app.subagents        — sub-agent definitions, shared utilities, agent registry
  app.orchestrator_agent — orchestrator graphs, graph cache, stream entrypoint
"""

from app.subagents import (
    AGENT_CONFIGS,
    MAX_SUBAGENT_MESSAGES,
    _MCP_CONNECTIVITY_MARKERS,
    _extract_final_answer,
    _make_tool_result_emitter,
    _partition_tools,
    _trim_subagent_messages,
    handle_mcp_tool_errors,
    set_debug_mode,
)
from app.orchestrator_agent import (
    begin_shutdown,
    get_inflight_count,
    invalidate_graph_cache,
    is_shutting_down,
    stream_agent,
    wait_for_inflight,
    warm_up,
)

__all__ = [
    "AGENT_CONFIGS",
    "MAX_SUBAGENT_MESSAGES",
    "_MCP_CONNECTIVITY_MARKERS",
    "_extract_final_answer",
    "_make_tool_result_emitter",
    "_partition_tools",
    "_trim_subagent_messages",
    "handle_mcp_tool_errors",
    "set_debug_mode",
    "begin_shutdown",
    "get_inflight_count",
    "invalidate_graph_cache",
    "is_shutting_down",
    "stream_agent",
    "wait_for_inflight",
    "warm_up",
]
