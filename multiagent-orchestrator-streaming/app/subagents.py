"""
Sub-agent definitions, shared utilities, and the agent registry.

Each entry in AGENT_CONFIGS becomes a compiled ReAct sub-graph that the
orchestrator can delegate to.  To add a new sub-agent:
  1. Add an entry to AGENT_CONFIGS below.
  2. Map the new MCP tools in _partition_tools().
  3. The orchestrator modes pick it up automatically.
"""

import contextvars
import json
import logging
import operator
from typing import Annotated

import httpx
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.config import get_stream_writer
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from mcp.client.streamable_http import StreamableHTTPError
from mcp.shared.exceptions import McpError
from typing_extensions import TypedDict

from app.settings import settings

logging.getLogger("langchain_google_genai._function_utils").setLevel(logging.ERROR)

# ══════════════════════════════════════════════════════════════════════════════
#  Debug Mode
# ══════════════════════════════════════════════════════════════════════════════

_DEBUG_MODE = False


def set_debug_mode(enabled: bool) -> None:
    """Called by the API layer to propagate the DEBUG_MODE flag."""
    global _DEBUG_MODE
    _DEBUG_MODE = enabled


def _debug_payload(base: dict, **extra) -> dict:
    """Merge extra debug fields into an SSE payload when debug mode is on."""
    if not _DEBUG_MODE:
        return base
    return {**base, "debug": extra}


# ══════════════════════════════════════════════════════════════════════════════
#  Think-Mode Delegation Log (per-request via ContextVar)
# ══════════════════════════════════════════════════════════════════════════════

# Tracks which agents were delegated to and their results within a single
# request.  Used by think_tool to return a real state summary.  ContextVar
# provides per-request isolation so the cached singleton graph stays safe.
_think_delegation_log_var: contextvars.ContextVar[list[dict] | None] = (
    contextvars.ContextVar("think_delegation_log", default=None)
)

# ══════════════════════════════════════════════════════════════════════════════
#  MCP Connectivity Error Handler for ToolNode
# ══════════════════════════════════════════════════════════════════════════════

# Exception types that signal an MCP server connectivity problem.
_MCP_CONNECTIVITY_EXCEPTIONS: tuple[type[Exception], ...] = (
    httpx.NetworkError,
    httpx.TimeoutException,
    httpx.HTTPStatusError,
    StreamableHTTPError,
    McpError,
    ConnectionError,
    OSError,
)

# Strings injected into ToolMessage content by handle_mcp_tool_errors when
# a connectivity failure is detected.  Used by tool_result_emitter nodes to
# emit "mcp_server disconnected" SSE events so the frontend can show status.
_MCP_CONNECTIVITY_MARKERS: tuple[str, ...] = (
    "MCP server connection failed",
    "MCP server timed out",
    "MCP server returned HTTP",
    "MCP StreamableHTTP transport error",
    "MCP protocol error",
    "Network connectivity error",
)


def handle_mcp_tool_errors(e: Exception) -> str:
    """Custom ToolNode error handler that provides actionable messages for
    MCP server connectivity failures.

    When a tool call fails due to a network/transport issue with an MCP server
    this handler returns a clear diagnosis so the LLM can retry with a
    different tool, skip the failing server, or inform the user.

    For non-connectivity errors the default repr-based message is returned.
    """
    if isinstance(e, httpx.ConnectError):
        return (
            f"MCP server connection failed: {e}. "
            "The server may be down or unreachable. "
            "Try a different approach or inform the user that this data source "
            "is temporarily unavailable."
        )
    if isinstance(e, httpx.TimeoutException):
        timeout_type = type(e).__name__
        return (
            f"MCP server timed out ({timeout_type}): {e}. "
            "The server is responding too slowly. "
            "Consider retrying with a simpler query or using an alternative tool."
        )
    if isinstance(e, httpx.HTTPStatusError):
        status = e.response.status_code
        return (
            f"MCP server returned HTTP {status}: {e}. "
            "The server encountered an error. "
            "Retry the request or use a different tool."
        )
    if isinstance(e, StreamableHTTPError):
        return (
            f"MCP StreamableHTTP transport error: {e}. "
            "The streaming connection to the MCP server was interrupted. "
            "Consider retrying or using an alternative tool."
        )
    if isinstance(e, McpError):
        return (
            f"MCP protocol error: {e}. "
            "The MCP server session may have expired or the server rejected "
            "the request. Consider retrying."
        )
    if isinstance(e, (ConnectionError, OSError)):
        return (
            f"Network connectivity error: {e}. "
            "A low-level network failure occurred while communicating with the "
            "MCP server. The server may be unreachable."
        )
    return f"Error: {repr(e)}\n Please fix your mistakes."


# ══════════════════════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════════════════════

GOOGLE_API_KEY = settings.google_api_key.get_secret_value()
GEMINI_MODEL = settings.gemini_model
MAX_AGENT_STEPS = settings.max_agent_steps
MAX_SUBAGENT_STEPS = settings.max_subagent_steps
MAX_RESOLUTION_ROUNDS = settings.max_resolution_rounds
ORCHESTRATOR_MODE = settings.orchestrator_mode
MAX_SUBAGENT_MESSAGES = settings.max_subagent_messages
MAX_ORCHESTRATOR_MESSAGES = settings.max_orchestrator_messages
MCP_INIT_TIMEOUT = settings.mcp_init_timeout
MCP_CONNECTIONS = settings.mcp_connections

# ══════════════════════════════════════════════════════════════════════════════
#  Agent Registry
# ══════════════════════════════════════════════════════════════════════════════
#
# Single source of truth for all sub-agents.  Both orchestrator modes
# read from this.  To add a new agent:
#   1. Add an entry to AGENT_CONFIGS below.
#   2. Map the new MCP tools in _partition_tools().
#   3. That's it — evaluator mode auto-generates nodes/edges,
#      think mode auto-generates delegate options.

AGENT_CONFIGS = [
    {
        "name": "doc_and_knowledge_search_agent",
        "role": "Doc and Knowledge Search specialist",
        "description": (
            "Doc and Knowledge Search specialist — use for questions about "
            "LangChain, LangGraph, LangSmith documentation, API "
            "references, how-to guides, tutorials, and concepts."
        ),
        "system_prompt": (
            "You are **Doc and Knowledge Search Agent**, a specialist in LangChain ecosystem "
            "documentation.\nYou have access to the LangChain documentation "
            "MCP server.\n\nInstructions:\n"
            "- Use the available tools to search and retrieve relevant documentation.\n"
            "- Provide clear, accurate answers grounded in the official docs.\n"
            "- Include code examples when appropriate.\n"
            "- Cite the documentation source when possible.\n"
            "- If the docs don't cover the topic, say so clearly."
        ),
        "mcp_server": "langchain_docs",
        "tool_prefix": "langchain_docs",
        "evaluator_intent": "doc_and_knowledge_search",
    },
    {
        "name": "web_agent",
        "role": "Web research specialist",
        "description": (
            "Web research specialist — use for general web search, "
            "scraping URLs, fetching live data, or anything not "
            "covered by the documentation."
        ),
        "system_prompt": (
            "You are **WebAgent**, a web research specialist powered by "
            "Firecrawl.\nYou have access to Firecrawl tools for web scraping "
            "and search.\n\nInstructions:\n"
            "- Use the available tools to search the web or scrape specific URLs.\n"
            "- Synthesize information from multiple sources when needed.\n"
            "- Provide well-structured, factual answers.\n"
            "- Always attribute information to its source.\n"
            "- Respect rate limits and be efficient with tool calls."
        ),
        "mcp_server": "firecrawl",
        "tool_prefix": "firecrawl",
        "evaluator_intent": "web",
    },
]


def _partition_tools(all_tools: list) -> dict[str, list]:
    """Split MCP tools by agent config tool_prefix.  Unmatched → last agent."""
    buckets: dict[str, list] = {cfg["name"]: [] for cfg in AGENT_CONFIGS}
    for t in all_tools:
        placed = False
        for cfg in AGENT_CONFIGS:
            if t.name.startswith(cfg["tool_prefix"]):
                buckets[cfg["name"]].append(t)
                placed = True
                break
        if not placed:
            buckets[AGENT_CONFIGS[-1]["name"]].append(t)
    return buckets


# ══════════════════════════════════════════════════════════════════════════════
#  Message Trimming Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _trim_subagent_messages(msgs: list) -> list:
    """Trim a sub-agent message list to at most MAX_SUBAGENT_MESSAGES.

    Uses ``trim_messages`` with strategy="last" so we keep the most recent
    exchanges.  ``include_system=True`` ensures the system prompt survives.
    ``start_on=("human", "ai")`` prevents orphaned ToolMessages at the
    start while allowing both human-initiated and mid-loop AI starts.
    """
    if len(msgs) <= MAX_SUBAGENT_MESSAGES:
        return msgs
    return trim_messages(
        msgs,
        max_tokens=MAX_SUBAGENT_MESSAGES,
        token_counter=len,
        strategy="last",
        include_system=True,
        start_on=("human", "ai"),
    )


def _trim_orchestrator_messages(msgs: list) -> list:
    """Trim the think-mode orchestrator message list.

    The orchestrator accumulates its own LLM replies plus tool results from
    delegate / think_tool calls.  We keep a larger window than sub-agents
    because the orchestrator needs more conversational context.
    """
    if len(msgs) <= MAX_ORCHESTRATOR_MESSAGES:
        return msgs
    return trim_messages(
        msgs,
        max_tokens=MAX_ORCHESTRATOR_MESSAGES,
        token_counter=len,
        strategy="last",
        include_system=True,
        start_on=("human", "ai"),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  State Schema
# ══════════════════════════════════════════════════════════════════════════════


class SubAgentState(TypedDict):
    """Shared state for sub-agent ReAct loops and think-mode orchestrator."""
    messages: Annotated[list, operator.add]


# ══════════════════════════════════════════════════════════════════════════════
#  Sub-Agent Graph Builder
# ══════════════════════════════════════════════════════════════════════════════


def _build_subagent_graph(
    tools: list,
    agent_name: str,
    system_prompt: str,
    mcp_server_key: str,
):
    """
    Build a ReAct sub-graph for a specific MCP server.

      agent_node ──► tools_condition ──► tool_node ──► agent_node (loop)
                 └──► END
    """
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
    ).bind_tools(tools)

    async def agent_node(state: SubAgentState) -> dict:
        writer = get_stream_writer()
        writer({"type": "llm_start", "agent": agent_name})

        msgs = list(state["messages"])
        if not any(isinstance(m, SystemMessage) for m in msgs):
            msgs.insert(0, SystemMessage(content=system_prompt))

        msgs = _trim_subagent_messages(msgs)

        response: AIMessage = await llm.ainvoke(msgs)

        if _DEBUG_MODE:
            raw_content = response.content
            if isinstance(raw_content, list):
                raw_content = "".join(
                    p["text"] if isinstance(p, dict) else str(p) for p in raw_content
                )
            writer(_debug_payload(
                {"type": "node_response", "node": f"{agent_name}/agent", "agent": agent_name},
                raw_content=raw_content,
                tool_calls=[
                    {"name": tc["name"], "args": tc["args"], "id": tc["id"]}
                    for tc in response.tool_calls
                ],
                message_type=response.__class__.__name__,
            ))

        for tc in response.tool_calls:
            writer(_debug_payload(
                {
                    "type": "tool_call",
                    "name": tc["name"],
                    "args": tc["args"],
                    "id": tc["id"],
                    "server": mcp_server_key,
                    "agent": agent_name,
                },
                full_args=tc["args"],
            ))

        return {"messages": [response]}

    builder = StateGraph(SubAgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(
        tools, handle_tool_errors=handle_mcp_tool_errors))
    builder.add_node("tool_result_emitter",
                     _make_tool_result_emitter(agent_name, mcp_server_key))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "tool_result_emitter")
    builder.add_edge("tool_result_emitter", "agent")

    return builder.compile()


def _build_agent_registry(tool_buckets: dict[str, list]) -> dict[str, dict]:
    """Build compiled sub-graphs for every agent in AGENT_CONFIGS."""
    registry = {}
    for cfg in AGENT_CONFIGS:
        registry[cfg["name"]] = {
            "graph": _build_subagent_graph(
                tools=tool_buckets.get(cfg["name"], []),
                agent_name=cfg["name"],
                system_prompt=cfg["system_prompt"],
                mcp_server_key=cfg["mcp_server"],
            ),
            **cfg,
        }
    return registry


# ══════════════════════════════════════════════════════════════════════════════
#  Shared Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _extract_final_answer(messages: list) -> str:
    """Walk messages in reverse to find the last AI text response."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            content = msg.content
            if isinstance(content, list):
                content = "".join(
                    p["text"] if isinstance(p, dict) else str(p) for p in content
                )
            return content
    return "I completed the research but couldn't formulate a final answer."


async def _invoke_agent(
    registry: dict[str, dict],
    agent_name: str,
    messages: list,
    writer=None,
    plan_step_id: str | None = None,
) -> str:
    """Run a sub-agent from the registry, emitting SSE events."""
    entry = registry[agent_name]
    if writer is None:
        writer = get_stream_writer()
    writer({"type": "agent_start", "agent": agent_name, "role": entry["role"]})
    if plan_step_id:
        writer({"type": "plan_step", "step_id": plan_step_id, "status": "running"})
    writer({"type": "mcp_server", "server": entry["mcp_server"], "status": "connected"})

    final_messages = []
    try:
        async for mode, chunk in entry["graph"].astream(
            {"messages": messages},
            stream_mode=["custom", "updates"],
            config={"recursion_limit": MAX_SUBAGENT_STEPS},
        ):
            if mode == "custom":
                writer(chunk)
            elif mode == "updates":
                for _node_name, state_delta in chunk.items():
                    for msg in state_delta.get("messages", []):
                        final_messages.append(msg)
    except GraphRecursionError:
        logging.warning(
            "Sub-agent '%s' hit recursion limit (%s steps)",
            agent_name, MAX_SUBAGENT_STEPS,
        )
        writer({"type": "error",
                "detail": f"The {agent_name} needed more research steps than allowed. "
                           "Returning the best partial answer so far.",
                "agent": agent_name})
    answer = _extract_final_answer(final_messages)

    writer({"type": "agent_end", "agent": agent_name,
            "summary": f"{agent_name} complete"})
    if plan_step_id:
        writer({"type": "plan_step", "step_id": plan_step_id, "status": "done"})
    return answer


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _make_tool_result_emitter(
    agent_name: str,
    mcp_server_key: str | list[str] | None = None,
):
    """Factory that returns a tool_result_emitter graph node.

    Shared by sub-agent graphs and the think-mode orchestrator graph to avoid
    duplicating the ToolMessage scanning and SSE-emission logic.

    Args:
        agent_name: Label attached to emitted SSE events (e.g. "web_agent").
        mcp_server_key: MCP server name(s) for connectivity events.
            - str: single server (current single-MCP subagent case)
            - list[str]: multiple servers — server is derived from tool name prefix
            - None: no MCP (think-mode orchestrator); server field is omitted
    """
    # Normalise to list for uniform handling inside the closure.
    _server_keys: list[str] = (
        [mcp_server_key] if isinstance(mcp_server_key, str)
        else (mcp_server_key or [])
    )

    def _resolve_server(tool_name: str | None) -> str | None:
        """Return the server key whose prefix matches the tool name."""
        if not tool_name or not _server_keys:
            return None
        for key in _server_keys:
            if tool_name.startswith(key):
                return key
        return _server_keys[0] if len(_server_keys) == 1 else None

    async def tool_result_emitter(state: SubAgentState) -> dict:
        writer = get_stream_writer()
        for msg in reversed(state["messages"]):
            if not isinstance(msg, ToolMessage):
                break
            content_str = str(msg.content)
            server = _resolve_server(msg.name)
            if any(marker in content_str for marker in _MCP_CONNECTIVITY_MARKERS):
                writer({
                    "type": "mcp_server",
                    "server": server or "unknown",
                    "status": "disconnected",
                    "error": content_str[:500],
                })
            payload: dict = {
                "type": "tool_result",
                "name": msg.name,
                "content": content_str[:1000],
                "agent": agent_name,
            }
            if server:
                payload["server"] = server
            writer(_debug_payload(
                payload,
                full_content=content_str,
                tool_call_id=getattr(msg, "tool_call_id", None),
            ))
        return {"messages": []}

    return tool_result_emitter
