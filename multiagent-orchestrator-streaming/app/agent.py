"""
LangGraph multi-agent orchestrator.

Architecture
────────────
  OrchestratorAgent  ─►  Agent Registry  ─┬─►  DocumentAgent  (LangChain Docs MCP)
                                           ├─►  WebAgent       (Firecrawl MCP)
                                           └─►  ... (add more agents here)

Two orchestrator modes  (ORCHESTRATOR_MODE env var)
───────────────────────────────────────────────────
  "evaluator" (default)
      Structured multi-node graph with a separate evaluator LLM that
      judges research completeness after each agent run.  Supports
      parallel fan-out and sequential follow-up with automatic
      cross-agent escalation.
        classify → agent(s) → evaluate → follow-up? → synthesize

  "think"
      Single ReAct loop where the orchestrator LLM has think_tool and
      a generic delegate(agent_name, query) tool.  The LLM self-decides
      what to call, reflects via think_tool, and stops when satisfied.
      Parallel fan-out happens when the LLM issues multiple delegate
      calls in a single response — ToolNode runs them concurrently.
        orchestrator → tools (think/delegate) → orchestrator (loop) → END

Both modes share the same agent registry.  To add a new sub-agent,
add one entry to AGENT_CONFIGS and it becomes available in both modes.

SSE event protocol (v2 — extensible)
─────────────────────────────────────
Core events:
  {"type": "llm_start"}
  {"type": "tool_call",    "name": "...", "args": {...}, "id": "..."}
  {"type": "tool_result",  "name": "...", "content": "..."}
  {"type": "text",         "content": "..."}
  {"type": "done"}
  {"type": "error",        "detail": "..."}

Orchestration events:
  {"type": "agent_start",  "agent": "...", "role": "..."}
  {"type": "agent_end",    "agent": "...", "summary": "..."}
  {"type": "handoff",      "from": "...", "to": "...", "reason": "..."}
  {"type": "plan",         "steps": [...], "agent": "orchestrator"}
  {"type": "plan_step",    "step_id": "...", "status": "running|done"}
  {"type": "status",       "message": "...", "agent": "..."}
  {"type": "mcp_server",   "server": "...", "status": "connected|disconnected", "error?": "..."}
"""

import asyncio
import contextvars
import json
import logging
import operator
from typing import Annotated, Literal

import httpx
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.config import get_stream_writer
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
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
# httpx.NetworkError covers ConnectError, DNS failures, socket resets, etc.
# httpx.TimeoutException covers ConnectTimeout, ReadTimeout, WriteTimeout, PoolTimeout.
# httpx.HTTPStatusError covers 5xx responses from MCP server infrastructure.
# StreamableHTTPError covers transport-layer failures in the MCP streamable HTTP client.
# McpError covers MCP protocol-level errors (e.g. session terminated).
# ConnectionError / OSError cover low-level socket / network issues.
_MCP_CONNECTIVITY_EXCEPTIONS: tuple[type[Exception], ...] = (
    httpx.NetworkError,
    httpx.TimeoutException,
    httpx.HTTPStatusError,
    StreamableHTTPError,
    McpError,
    ConnectionError,
    OSError,
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
        timeout_type = type(e).__name__  # ConnectTimeout, ReadTimeout, etc.
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
    # Fallback for all other errors — matches LangGraph default behaviour.
    return f"Error: {repr(e)}\n Please fix your mistakes."


# ══════════════════════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════════════════════

# All env-var config is validated centrally in settings.py.
# Local aliases keep the rest of this file concise.
GOOGLE_API_KEY = settings.google_api_key.get_secret_value()
GEMINI_MODEL = settings.gemini_model
MAX_AGENT_STEPS = settings.max_agent_steps
MAX_SUBAGENT_STEPS = settings.max_subagent_steps
MAX_RESOLUTION_ROUNDS = settings.max_resolution_rounds
ORCHESTRATOR_MODE = settings.orchestrator_mode

# Message trimming configuration.
# Sub-agents run tool-heavy ReAct loops that can easily exceed the model's
# context window.  We keep the most recent N messages (by count) and always
# preserve the SystemMessage.  Using message-count (token_counter=len) is
# cheap and predictable; for token-precise trimming, swap to
# token_counter="approximate" and adjust the max_tokens value.
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
        "name": "document_agent",
        "role": "LangChain docs specialist",
        "description": (
            "LangChain docs specialist — use for questions about "
            "LangChain, LangGraph, LangSmith documentation, API "
            "references, how-to guides, tutorials, and concepts."
        ),
        "system_prompt": (
            "You are **DocumentAgent**, a specialist in LangChain ecosystem "
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
        "evaluator_intent": "document",
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
            # fallback: last agent gets unmatched tools
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
#  State Schemas
# ══════════════════════════════════════════════════════════════════════════════


class OrchestratorState(TypedDict):
    """Top-level orchestrator state (evaluator mode)."""
    messages: Annotated[list, operator.add]
    intent: str
    active_agent: str
    final_answer: str
    agent_answers: dict          # {agent_name: answer_str}
    resolution_status: str       # "resolved" | "needs_<agent_name>"
    resolution_round: int
    followup_query: str


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

        # Trim to prevent context window overflow in long ReAct loops
        msgs = _trim_subagent_messages(msgs)

        response: AIMessage = await llm.ainvoke(msgs)

        # Debug: emit the full LLM response
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

    async def tool_result_emitter(state: SubAgentState) -> dict:
        writer = get_stream_writer()
        for msg in reversed(state["messages"]):
            if not isinstance(msg, ToolMessage):
                break
            content_str = str(msg.content)
            # Detect MCP connectivity failures in ToolMessage content and
            # emit a disconnected event so the frontend can show status.
            _MCP_CONNECTIVITY_MARKERS = (
                "MCP server connection failed",
                "MCP server timed out",
                "MCP server returned HTTP",
                "MCP StreamableHTTP transport error",
                "MCP protocol error",
                "Network connectivity error",
            )
            if any(marker in content_str for marker in _MCP_CONNECTIVITY_MARKERS):
                writer({
                    "type": "mcp_server",
                    "server": mcp_server_key,
                    "status": "disconnected",
                    "error": content_str[:500],
                })
            writer(_debug_payload(
                {
                    "type": "tool_result",
                    "name": msg.name,
                    "content": content_str[:1000],
                    "server": mcp_server_key,
                    "agent": agent_name,
                },
                full_content=content_str,
                tool_call_id=getattr(msg, "tool_call_id", None),
            ))
        return {"messages": []}

    builder = StateGraph(SubAgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(
        tools, handle_tool_errors=handle_mcp_tool_errors))
    builder.add_node("tool_result_emitter", tool_result_emitter)
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

    # Use astream with custom mode so sub-agent tool_call / tool_result
    # events propagate to the parent SSE stream.
    final_messages = []
    try:
        async for mode, chunk in entry["graph"].astream(
            {"messages": messages},
            stream_mode=["custom", "updates"],
            config={"recursion_limit": MAX_SUBAGENT_STEPS},
        ):
            if mode == "custom":
                # Forward sub-agent custom events (tool_call, tool_result, etc.)
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


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluator-Mode Orchestrator  (ORCHESTRATOR_MODE=evaluator)
# ══════════════════════════════════════════════════════════════════════════════

# --- Prompts ---------------------------------------------------------------

_agent_names = [c["name"] for c in AGENT_CONFIGS]
_intent_lines = "\n".join(
    f'  - "{c["evaluator_intent"]}"  → {c["description"]}'
    for c in AGENT_CONFIGS
)

ORCHESTRATOR_SYSTEM_PROMPT = f"""\
You are a routing orchestrator. Your ONLY job is to classify the user's
intent and pick which specialist agent should handle the request.

Reply with EXACTLY one of these intents (nothing else):
{_intent_lines}
  - "both"      → complex questions that need multiple specialists
  - "general"   → simple greetings, chitchat, or questions you can answer
                   directly without any tools

Respond with ONLY the intent word, nothing else."""

# Build evaluator prompt dynamically from registry
_needs_options = "\n\n".join(
    f'  {{"status": "needs_{c["name"]}", "followup_query": "..."}}\n'
    f'    → we need research from {c["name"]}; provide a focused follow-up query'
    for c in AGENT_CONFIGS
)

EVALUATOR_SYSTEM_PROMPT = f"""\
You are a quality evaluator. Given the user's original question and the
research collected so far, decide whether the answer is complete.

You MUST reply with EXACTLY one JSON object (no markdown, no explanation):

  {{"status": "resolved"}}
    → the research fully answers the question

{_needs_options}

IMPORTANT:
- If only ONE source has been consulted so far and it clearly cannot
  answer the question, request a DIFFERENT source.
- Be pragmatic: if the existing research is 80%+ sufficient, return
  "resolved" and let the synthesizer fill minor gaps.  Only request
  a follow-up when there is a clear, significant gap."""


def build_evaluator_orchestrator(registry: dict[str, dict]):
    """
    Build the evaluator-mode orchestrator graph.

    All agent paths converge on the evaluate node, which can escalate
    to any other agent.  Supports N agents, parallel fan-out, and
    sequential follow-up with automatic cross-agent escalation.

    document / web / ...:
      classify → agent → evaluate ──┬─► synthesize → END
                                    └─► followup_agent → evaluate (loop)
    both:
      classify → parallel_agents → evaluate ──┬─► synthesize → END
                                              └─► followup_agent → evaluate (loop)
    general:
      classify → direct_answer → synthesize → END
    """
    agent_names = list(registry.keys())
    intent_to_agent = {cfg["evaluator_intent"]: cfg["name"] for cfg in AGENT_CONFIGS}

    classifier_llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL, google_api_key=GOOGLE_API_KEY)
    evaluator_llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL, google_api_key=GOOGLE_API_KEY)
    direct_llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL, google_api_key=GOOGLE_API_KEY)

    # ── Nodes ─────────────────────────────────────────────────────────────

    async def classify_intent(state: OrchestratorState) -> dict:
        writer = get_stream_writer()
        writer({"type": "agent_start", "agent": "orchestrator", "role": "coordinator"})
        writer({"type": "status", "message": "Analyzing your request...",
                "agent": "orchestrator"})

        user_query = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break

        response = await classifier_llm.ainvoke([
            SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
            HumanMessage(content=user_query),
        ])
        raw = response.content.strip().lower().strip('"').strip("'")

        # Match intent
        if "both" in raw:
            intent = "both"
        else:
            intent = "general"
            for key in intent_to_agent:
                if key in raw:
                    intent = key
                    break

        # Build plan
        plan_steps = [{"id": "classify", "label": "Classify intent"}]
        if intent == "both":
            for name in agent_names:
                plan_steps.append({"id": f"execute_{name}",
                                   "label": f"Run {name}"})
        elif intent != "general":
            agent = intent_to_agent[intent]
            plan_steps.append({"id": "execute", "label": f"Run {agent}"})
        else:
            plan_steps.append({"id": "execute", "label": "Direct answer"})
        if intent != "general":
            plan_steps.append({"id": "evaluate",
                               "label": "Evaluate completeness"})
        plan_steps.append({"id": "respond", "label": "Generate response"})

        writer({"type": "plan", "steps": plan_steps, "agent": "orchestrator"})
        writer({"type": "plan_step", "step_id": "classify", "status": "done"})

        if _DEBUG_MODE:
            writer(_debug_payload(
                {"type": "node_response", "node": "classify_intent", "agent": "orchestrator"},
                raw_llm_output=raw,
                resolved_intent=intent,
                plan_steps=plan_steps,
            ))

        return {"intent": intent, "active_agent": intent, "resolution_round": 0}

    def route_after_classify(state: OrchestratorState) -> str:
        intent = state["intent"]
        if intent == "both":
            return "parallel_agents"
        elif intent in intent_to_agent:
            return intent_to_agent[intent]
        return "direct_answer"

    # -- Single-agent nodes (one per registered agent) ---------------------

    async def _make_agent_node(agent_name: str, state: OrchestratorState) -> dict:
        writer = get_stream_writer()
        writer({"type": "handoff", "from": "orchestrator", "to": agent_name,
                "reason": f"{agent_name} lookup requested"})
        answer = await _invoke_agent(registry, agent_name,
                                     state["messages"], writer, "execute")
        answers = dict(state.get("agent_answers", {}))
        answers[agent_name] = answer
        return {"agent_answers": answers}

    # -- Parallel fan-out --------------------------------------------------

    async def parallel_agents_node(state: OrchestratorState) -> dict:
        writer = get_stream_writer()
        writer({"type": "handoff", "from": "orchestrator",
                "to": "parallel_agents",
                "reason": "Complex query — running all agents in parallel"})

        async def _run(name: str) -> tuple[str, str]:
            answer = await _invoke_agent(
                registry, name, state["messages"], writer,
                f"execute_{name}")
            return name, answer

        results = await asyncio.gather(*[_run(n) for n in agent_names])
        return {"agent_answers": dict(results)}

    # -- Direct answer (no tools) ------------------------------------------

    async def direct_answer_node(state: OrchestratorState) -> dict:
        writer = get_stream_writer()
        writer({"type": "agent_start", "agent": "orchestrator",
                "role": "direct responder"})
        writer({"type": "plan_step", "step_id": "execute", "status": "running"})

        response = await direct_llm.ainvoke(state["messages"])
        content = response.content
        if isinstance(content, list):
            content = "".join(
                p["text"] if isinstance(p, dict) else str(p) for p in content)

        writer({"type": "plan_step", "step_id": "execute", "status": "done"})
        return {"final_answer": content}

    # -- Evaluate completeness ---------------------------------------------

    async def evaluate_node(state: OrchestratorState) -> dict:
        writer = get_stream_writer()
        writer({"type": "plan_step", "step_id": "evaluate", "status": "running"})
        writer({"type": "status",
                "message": "Evaluating research completeness…",
                "agent": "orchestrator"})

        user_query = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break

        current_round = state.get("resolution_round", 0)
        answers = state.get("agent_answers", {})

        # Build evaluation input showing what each agent found
        research_sections = []
        for name in agent_names:
            answer = answers.get(name, "")
            label = "(not consulted yet)" if not answer else answer
            research_sections.append(f"## {name}\n{label}")

        eval_input = (
            f"## Original Question\n{user_query}\n\n"
            + "\n\n".join(research_sections)
            + f"\n\n## Resolution Round\n{current_round + 1} of "
            f"{MAX_RESOLUTION_ROUNDS}"
        )

        # Max rounds → force resolve
        if current_round >= MAX_RESOLUTION_ROUNDS:
            writer({"type": "status",
                    "message": "Max rounds reached — synthesizing",
                    "agent": "orchestrator"})
            writer({"type": "plan_step", "step_id": "evaluate",
                    "status": "done"})
            return {"resolution_status": "resolved",
                    "resolution_round": current_round}

        response = await evaluator_llm.ainvoke([
            SystemMessage(content=EVALUATOR_SYSTEM_PROMPT),
            HumanMessage(content=eval_input),
        ])

        raw = response.content.strip()
        try:
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            verdict = json.loads(raw)
        except (json.JSONDecodeError, IndexError):
            logging.warning("Evaluator unparseable: %s", raw)
            verdict = {"status": "resolved"}

        status = verdict.get("status", "resolved")
        followup = verdict.get("followup_query", "")

        # Validate status
        valid = {"resolved"} | {f"needs_{n}" for n in agent_names}
        if status not in valid:
            status = "resolved"

        writer({"type": "status",
                "message": f"Evaluation: {status}"
                + (f" — {followup}" if followup else ""),
                "agent": "orchestrator"})
        writer({"type": "plan_step", "step_id": "evaluate", "status": "done"})

        if status != "resolved":
            target = status.replace("needs_", "")
            writer({"type": "plan", "steps": [
                {"id": f"followup_r{current_round + 1}",
                 "label": f"Follow-up: {target} (round {current_round + 1})"},
                {"id": "evaluate", "label": "Re-evaluate"},
                {"id": "respond", "label": "Generate response"},
            ], "agent": "orchestrator"})

        if _DEBUG_MODE:
            writer(_debug_payload(
                {"type": "node_response", "node": "evaluate", "agent": "orchestrator"},
                raw_evaluator_output=raw,
                parsed_verdict=verdict,
                resolution_status=status,
                followup_query=followup,
                resolution_round=current_round + 1,
                eval_input_preview=eval_input[:2000],
            ))

        return {
            "resolution_status": status,
            "resolution_round": current_round + 1,
            "followup_query": followup,
        }

    def route_after_evaluate(state: OrchestratorState) -> str:
        status = state.get("resolution_status", "resolved")
        for name in agent_names:
            if status == f"needs_{name}":
                return f"followup_{name}"
        return "synthesize"

    # -- Follow-up agent nodes (one per registered agent) ------------------

    async def _make_followup_node(
        agent_name: str, state: OrchestratorState,
    ) -> dict:
        writer = get_stream_writer()
        rnd = state.get("resolution_round", 1)
        step_id = f"followup_r{rnd}"

        writer({"type": "handoff", "from": "orchestrator", "to": agent_name,
                "reason": f"Follow-up round {rnd}"})

        followup_q = state.get("followup_query", "")
        answers = state.get("agent_answers", {})

        # Build context from other agents' prior research
        context_parts = []
        for other_name, other_answer in answers.items():
            if other_name != agent_name and other_answer:
                context_parts.append(
                    f"Previous {other_name} research:\n{other_answer}")
        context_parts.append(f"Focus on:\n{followup_q}")
        messages = [HumanMessage(content="\n\n".join(context_parts))]

        answer = await _invoke_agent(registry, agent_name,
                                     messages, writer, step_id)

        # Append to existing answer
        prev = answers.get(agent_name, "")
        combined = (f"{prev}\n\n---\n### Follow-up (round {rnd})\n{answer}"
                    if prev else answer)
        new_answers = dict(answers)
        new_answers[agent_name] = combined
        return {"agent_answers": new_answers}

    # -- Synthesize --------------------------------------------------------

    async def synthesize(state: OrchestratorState) -> dict:
        writer = get_stream_writer()
        writer({"type": "plan_step", "step_id": "respond", "status": "running"})

        answers = state.get("agent_answers", {})
        filled = {k: v for k, v in answers.items() if v}

        if len(filled) > 1:
            # Merge multiple sources
            user_query = ""
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    user_query = msg.content
                    break

            sections = "\n\n".join(
                f"## {name}\n{ans}" for name, ans in filled.items())
            merge_prompt = (
                f"## Original Question\n{user_query}\n\n"
                "Combine the following research into one comprehensive, "
                "well-structured answer. Cite which source each fact comes "
                f"from.\n\n{sections}"
            )
            merged = await direct_llm.ainvoke([
                SystemMessage(content="You synthesise multi-source research "
                              "into a single authoritative answer."),
                HumanMessage(content=merge_prompt),
            ])
            answer = merged.content
            if isinstance(answer, list):
                answer = "".join(
                    p["text"] if isinstance(p, dict) else str(p)
                    for p in answer)
        elif filled:
            answer = next(iter(filled.values()))
        else:
            answer = state.get("final_answer",
                               "I wasn't able to generate a response.")

        writer({"type": "text", "content": answer})
        writer({"type": "agent_end", "agent": "orchestrator",
                "summary": "Response complete"})
        writer({"type": "plan_step", "step_id": "respond", "status": "done"})

        if _DEBUG_MODE:
            writer(_debug_payload(
                {"type": "node_response", "node": "synthesize", "agent": "orchestrator"},
                source_count=len(filled),
                sources=list(filled.keys()),
                answer_length=len(answer),
            ))

        return {"messages": [AIMessage(content=answer)]}

    # ── Build Graph ───────────────────────────────────────────────────────

    builder = StateGraph(OrchestratorState)

    # Core nodes
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("direct_answer", direct_answer_node)
    builder.add_node("parallel_agents", parallel_agents_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("synthesize", synthesize)

    # Dynamic: one agent node + one followup node per registered agent
    route_classify_targets = ["parallel_agents", "direct_answer"]
    route_evaluate_targets = ["synthesize"]

    for name in agent_names:
        # Agent node (initial execution)
        async def _agent_node(state, _name=name):
            return await _make_agent_node(_name, state)
        builder.add_node(name, _agent_node)
        route_classify_targets.append(name)
        builder.add_edge(name, "evaluate")

        # Follow-up node (sequential escalation)
        followup_name = f"followup_{name}"
        async def _followup_node(state, _name=name):
            return await _make_followup_node(_name, state)
        builder.add_node(followup_name, _followup_node)
        route_evaluate_targets.append(followup_name)
        builder.add_edge(followup_name, "evaluate")

    # Entry
    builder.add_edge(START, "classify_intent")

    # classify → agent selection
    builder.add_conditional_edges(
        "classify_intent", route_after_classify, route_classify_targets)

    # parallel → evaluate
    builder.add_edge("parallel_agents", "evaluate")

    # direct_answer → synthesize (skip evaluate for chitchat)
    builder.add_edge("direct_answer", "synthesize")

    # evaluate → (resolved → synthesize) | (needs_X → followup_X → evaluate)
    builder.add_conditional_edges(
        "evaluate", route_after_evaluate, route_evaluate_targets)

    # Terminal
    builder.add_edge("synthesize", END)

    return builder.compile()


# ══════════════════════════════════════════════════════════════════════════════
#  Think-Mode Orchestrator  (ORCHESTRATOR_MODE=think)
# ══════════════════════════════════════════════════════════════════════════════

THINK_ORCHESTRATOR_SYSTEM_PROMPT = """\
You are an orchestrator agent with access to specialist sub-agents.
Your job is to answer the user's question by delegating research to
the right sub-agents, reflecting on results, and synthesizing a final
answer.

Available tools:
  • think_tool  — reflect on what you know, what's missing, and what
                  to do next.  MUST be used after every delegation.
  • delegate    — send a focused query to one sub-agent by name.

Available sub-agents:
{agent_descriptions}

Parallel execution:
  To run multiple agents in parallel, call `delegate` multiple times
  in the SAME response (one call per agent, each with its own query).
  They will execute concurrently.

Workflow:
  1. Read the user's question.
  2. Decide which agent(s) to call (or answer directly for greetings).
  3. Call delegate(agent_name=..., query=...) for each agent you need.
     For parallel execution, include multiple delegate calls in one response.
  4. Use think_tool to assess: Is the answer complete?  What's missing?
  5. If gaps remain, delegate again with a refined query.
  6. When satisfied, respond directly to the user with the final answer.
     Do NOT call any more tools — just write the answer.

Rules:
  • You MUST use think_tool after each round of delegation to reflect.
  • Do NOT delegate more than {max_rounds} times total.
  • If one agent can't answer, try a different one.
  • For complex questions, delegate to multiple agents in parallel.
  • When you have enough information, stop calling tools and respond.
"""


def build_think_orchestrator(registry: dict[str, dict]):
    """
    Build the think-tool orchestrator: a single ReAct loop.

      START → orchestrator → tools_condition ─┬─► tools → orchestrator (loop)
                                              └─► END
    """
    agent_names = sorted(registry.keys())
    agent_names_str = ", ".join(f'"{n}"' for n in agent_names)

    # Build description block for system prompt
    agent_desc_lines = [
        f'  • "{name}" — {registry[name]["description"]}'
        for name in agent_names
    ]
    agent_descriptions_block = "\n".join(agent_desc_lines)

    # ── Delegation tracking ────────────────────────────────────────────────
    #
    # Per-request log shared between delegate() and think_tool() via
    # ContextVar.  Each delegate call appends a record so think_tool
    # can return a real state summary instead of a static string —
    # forcing the LLM to process actual research results rather than
    # hallucinating a reflection.
    #
    # The ContextVar is module-level (_think_delegation_log_var) so
    # stream_agent() can reset it per request.  Each async request
    # handler runs in its own context copy, so concurrent requests
    # are isolated.

    def _get_delegation_log() -> list[dict]:
        """Get the per-request delegation log, creating it if needed."""
        log = _think_delegation_log_var.get(None)
        if log is None:
            log = []
            _think_delegation_log_var.set(log)
        return log

    # ── Tools ─────────────────────────────────────────────────────────────

    @tool
    def think_tool(reflection: str) -> str:
        """Reflect on research progress and decide next steps.

        Use this after every round of delegation to assess:
        - What key information did I get?
        - Is it sufficient to answer the user?
        - What specific gaps remain?
        - Should I delegate again or respond now?

        Args:
            reflection: Your analysis of findings, gaps, and next action.

        Returns:
            A structured summary of all delegations so far, including which
            agents were consulted, their queries, and answer previews.
        """
        delegation_log = _get_delegation_log()

        if not delegation_log:
            return (
                "Reflection noted. No delegations have been made yet.\n"
                f"Available agents: {agent_names_str}\n"
                "Decide which agent(s) to consult, or respond directly."
            )

        # Build a structured state summary from actual delegation results
        total = len(delegation_log)
        succeeded = sum(1 for d in delegation_log if d["status"] == "ok")
        failed = total - succeeded
        agents_consulted = sorted(set(d["agent"] for d in delegation_log))
        agents_not_consulted = sorted(
            set(agent_names) - set(d["agent"] for d in delegation_log)
        )

        lines = [
            "═══ Research State Summary ═══",
            f"Delegations: {total} total, {succeeded} succeeded, {failed} failed",
            f"Agents consulted: {', '.join(agents_consulted)}",
        ]
        if agents_not_consulted:
            lines.append(
                f"Agents NOT yet consulted: {', '.join(agents_not_consulted)}"
            )

        lines.append("")
        for i, entry in enumerate(delegation_log, 1):
            status_icon = "✓" if entry["status"] == "ok" else "✗"
            lines.append(f"── Delegation {i} [{status_icon}] ──")
            lines.append(f"  Agent: {entry['agent']}")
            lines.append(f"  Query: {entry['query']}")
            # Show a meaningful preview — enough for the LLM to judge
            # completeness without repeating the full answer
            preview = entry["answer_preview"]
            lines.append(f"  Result preview ({entry['answer_len']} chars total):")
            lines.append(f"    {preview}")
            lines.append("")

        lines.append(
            "Based on the above, decide: is the research sufficient to "
            "answer the user's question, or do you need to delegate again "
            "with a more focused query?"
        )
        return "\n".join(lines)

    @tool
    async def delegate(agent_name: str, query: str) -> str:
        """Delegate a research query to a specialist sub-agent.

        To run agents IN PARALLEL, call `delegate` multiple times in the
        SAME response — each call runs concurrently.

        Args:
            agent_name: The name of the agent to delegate to.
            query: A focused research question for this agent.

        Returns:
            The agent's research findings.
        """
        if agent_name not in registry:
            error_msg = (
                f"Unknown agent '{agent_name}'. "
                f"Available agents: {agent_names_str}"
            )
            _get_delegation_log().append({
                "agent": agent_name,
                "query": query,
                "answer_preview": error_msg,
                "answer_len": 0,
                "status": "error",
            })
            return error_msg

        entry = registry[agent_name]
        writer = get_stream_writer()
        writer({"type": "handoff", "from": "orchestrator", "to": agent_name,
                "reason": query[:200]})
        writer({"type": "agent_start", "agent": agent_name,
                "role": entry["role"]})
        writer({"type": "mcp_server", "server": entry["mcp_server"],
                "status": "connected"})

        # Use astream with custom mode so sub-agent tool_call / tool_result
        # events propagate to the parent SSE stream.
        final_messages = []
        hit_limit = False
        try:
            async for mode, chunk in entry["graph"].astream(
                {"messages": [HumanMessage(content=query)]},
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
            hit_limit = True
            logging.warning(
                "Sub-agent '%s' hit recursion limit (%s steps) for query: %s",
                agent_name, MAX_SUBAGENT_STEPS, query[:200],
            )
            writer({"type": "error",
                    "detail": f"The {agent_name} needed more research steps than allowed. "
                               "Returning the best partial answer so far.",
                    "agent": agent_name})
        answer = _extract_final_answer(final_messages)

        # Track delegation for think_tool state summary
        _get_delegation_log().append({
            "agent": agent_name,
            "query": query,
            "answer_preview": answer[:500] + ("…" if len(answer) > 500 else ""),
            "answer_len": len(answer),
            "status": "truncated" if hit_limit else "ok",
        })

        writer({"type": "agent_end", "agent": agent_name,
                "summary": f"{agent_name} complete"})
        return answer

    # ── Build ReAct graph ─────────────────────────────────────────────────

    orchestrator_tools = [think_tool, delegate]

    orch_llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
    ).bind_tools(orchestrator_tools)

    system_prompt = (
        THINK_ORCHESTRATOR_SYSTEM_PROMPT
        .replace("{agent_descriptions}", agent_descriptions_block)
        .replace("{max_rounds}", str(MAX_RESOLUTION_ROUNDS + 2))
    )

    async def orchestrator_agent_node(state: SubAgentState) -> dict:
        writer = get_stream_writer()
        writer({"type": "llm_start", "agent": "orchestrator"})

        msgs = list(state["messages"])
        if not any(isinstance(m, SystemMessage) for m in msgs):
            msgs.insert(0, SystemMessage(content=system_prompt))

        # Trim to prevent context window overflow in long research sessions
        msgs = _trim_orchestrator_messages(msgs)

        response: AIMessage = await orch_llm.ainvoke(msgs)

        # Debug: emit the full orchestrator LLM response
        if _DEBUG_MODE:
            raw_content = response.content
            if isinstance(raw_content, list):
                raw_content = "".join(
                    p["text"] if isinstance(p, dict) else str(p) for p in raw_content
                )
            writer(_debug_payload(
                {"type": "node_response", "node": "orchestrator", "agent": "orchestrator"},
                raw_content=raw_content,
                tool_calls=[
                    {"name": tc["name"], "args": tc["args"], "id": tc["id"]}
                    for tc in response.tool_calls
                ],
                message_type=response.__class__.__name__,
                has_final_answer=not bool(response.tool_calls),
            ))

        for tc in response.tool_calls:
            writer(_debug_payload(
                {
                    "type": "tool_call",
                    "name": tc["name"],
                    "args": tc["args"],
                    "id": tc["id"],
                    "agent": "orchestrator",
                },
                full_args=tc["args"],
            ))

        if not response.tool_calls:
            content = response.content
            if isinstance(content, list):
                content = "".join(
                    p["text"] if isinstance(p, dict) else str(p)
                    for p in content)
            writer({"type": "text", "content": content})
            writer({"type": "agent_end", "agent": "orchestrator",
                    "summary": "Response complete"})

        return {"messages": [response]}

    async def tool_result_emitter(state: SubAgentState) -> dict:
        writer = get_stream_writer()
        for msg in reversed(state["messages"]):
            if not isinstance(msg, ToolMessage):
                break
            content_str = str(msg.content)
            # Detect MCP connectivity failures surfaced through delegate
            # tool results and emit a disconnected status event.
            _MCP_CONNECTIVITY_MARKERS = (
                "MCP server connection failed",
                "MCP server timed out",
                "MCP server returned HTTP",
                "MCP StreamableHTTP transport error",
                "MCP protocol error",
                "Network connectivity error",
            )
            if any(marker in content_str for marker in _MCP_CONNECTIVITY_MARKERS):
                writer({
                    "type": "mcp_server",
                    "server": "unknown",
                    "status": "disconnected",
                    "error": content_str[:500],
                })
            writer(_debug_payload(
                {
                    "type": "tool_result",
                    "name": msg.name,
                    "content": content_str[:500],
                    "agent": "orchestrator",
                },
                full_content=content_str,
                tool_call_id=getattr(msg, "tool_call_id", None),
            ))
        return {"messages": []}

    builder = StateGraph(SubAgentState)
    builder.add_node("orchestrator", orchestrator_agent_node)
    builder.add_node("tools", ToolNode(
        orchestrator_tools, handle_tool_errors=handle_mcp_tool_errors))
    builder.add_node("tool_result_emitter", tool_result_emitter)
    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges("orchestrator", tools_condition)
    builder.add_edge("tools", "tool_result_emitter")
    builder.add_edge("tool_result_emitter", "orchestrator")

    return builder.compile()


# ══════════════════════════════════════════════════════════════════════════════
#  Singleton Graph Cache
# ══════════════════════════════════════════════════════════════════════════════
#
# Optimization: build the MCP client, discover tools, and compile the
# LangGraph orchestrator graph ONCE on first request, then reuse on
# every subsequent request.  This avoids:
#   - Reconnecting to every MCP server to list tools on each request
#   - Rebuilding the full agent registry and LangGraph graph per request
#
# The compiled graph is stateless (state is passed in per invocation),
# so sharing it across requests is safe.  Individual tool calls still
# open their own MCP sessions (connection-per-call, built into the
# langchain-mcp-adapters library).
#
# To force a rebuild (e.g. after adding a new MCP server), call
# invalidate_graph_cache().

_cached_graph = None
_graph_lock = asyncio.Lock()

# In-flight request tracking — used for graceful shutdown and safe cache
# invalidation.  Each active stream_agent() generator increments on entry
# and decrements on exit.
_inflight_count: int = 0
_inflight_lock = asyncio.Lock()
_inflight_zero = asyncio.Event()    # set when count reaches 0
_inflight_zero.set()                # initially zero → set
_shutting_down: bool = False


async def _get_or_build_graph():
    """Return the cached orchestrator graph, building it on first call.

    Uses asyncio.Lock to ensure only one coroutine builds the graph
    even if multiple requests arrive concurrently at startup.
    """
    global _cached_graph
    if _cached_graph is not None:
        return _cached_graph

    async with _graph_lock:
        # Double-check after acquiring lock
        if _cached_graph is not None:
            return _cached_graph

        logging.info("Building orchestrator graph (first request)...")
        client = MultiServerMCPClient(MCP_CONNECTIONS)
        try:
            all_tools = await asyncio.wait_for(
                client.get_tools(), timeout=MCP_INIT_TIMEOUT
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"MCP tool discovery timed out after {MCP_INIT_TIMEOUT}s. "
                f"One or more MCP servers in MCP_CONNECTIONS may be "
                f"unreachable. Increase MCP_INIT_TIMEOUT or check server "
                f"health."
            )

        tool_buckets = _partition_tools(all_tools)
        registry = _build_agent_registry(tool_buckets)

        if ORCHESTRATOR_MODE == "think":
            graph = build_think_orchestrator(registry)
        else:
            graph = build_evaluator_orchestrator(registry)

        _cached_graph = graph
        logging.info("Orchestrator graph ready (mode=%s, tools=%d)",
                     ORCHESTRATOR_MODE, len(all_tools))
        return _cached_graph


def invalidate_graph_cache() -> None:
    """Force the orchestrator graph to be rebuilt on the next request.

    Call this after changing MCP_CONNECTIONS or AGENT_CONFIGS at runtime.
    In-flight requests continue using the old graph; the next request
    triggers a fresh build.
    """
    global _cached_graph
    _cached_graph = None
    logging.info("Graph cache invalidated — will rebuild on next request")


async def wait_for_inflight(timeout: float = 30.0) -> bool:
    """Wait until all in-flight requests have completed.

    Returns True if all requests finished within the timeout, False if
    the timeout expired with requests still in progress.
    """
    try:
        await asyncio.wait_for(_inflight_zero.wait(), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        logging.warning(
            "Timed out waiting for in-flight requests (%d still active)",
            _inflight_count,
        )
        return False


def get_inflight_count() -> int:
    """Return the current number of in-flight requests (for health checks)."""
    return _inflight_count


def is_shutting_down() -> bool:
    """Return True if a graceful shutdown has been initiated."""
    return _shutting_down


def begin_shutdown() -> None:
    """Mark the agent module as shutting down.

    New requests via stream_agent() will be rejected with a shutdown
    notice.  Existing in-flight requests are allowed to complete.
    """
    global _shutting_down
    _shutting_down = True
    logging.info("Shutdown initiated — rejecting new requests")


# ══════════════════════════════════════════════════════════════════════════════
#  Stream Entrypoint
# ══════════════════════════════════════════════════════════════════════════════


async def stream_agent(query: str):
    """Async generator that yields SSE strings from the orchestrator."""
    global _inflight_count

    # Reject new requests during graceful shutdown
    if _shutting_down:
        yield sse({"type": "error",
                   "detail": "Server is shutting down. Please retry shortly."})
        return

    # Track in-flight requests for graceful shutdown / cache invalidation
    async with _inflight_lock:
        _inflight_count += 1
        _inflight_zero.clear()

    try:
        graph = await _get_or_build_graph()

        # Reset per-request delegation log for think-mode state tracking
        _think_delegation_log_var.set(None)

        # Build per-request inputs
        if ORCHESTRATOR_MODE == "think":
            inputs = {"messages": [HumanMessage(content=query)]}
        else:
            inputs = {
                "messages": [HumanMessage(content=query)],
                "intent": "",
                "active_agent": "",
                "final_answer": "",
                "agent_answers": {},
                "resolution_status": "",
                "resolution_round": 0,
                "followup_query": "",
            }

        logging.info("Orchestrator mode: %s", ORCHESTRATOR_MODE)

        async for mode, chunk in graph.astream(
            inputs,
            stream_mode=["updates", "custom"],
            config={"recursion_limit": MAX_AGENT_STEPS},
        ):
            if mode == "custom":
                yield sse(chunk)

            elif mode == "updates":
                for node_name, state_delta in chunk.items():
                    # Debug: emit raw graph node state updates
                    if _DEBUG_MODE:
                        debug_state = {}
                        for k, v in state_delta.items():
                            if k == "messages":
                                debug_state[k] = [
                                    {
                                        "type": m.__class__.__name__,
                                        "content": str(m.content)[:500] if hasattr(m, "content") else None,
                                        **({"tool_calls": [
                                            {"name": tc["name"], "args": tc["args"], "id": tc["id"]}
                                            for tc in m.tool_calls
                                        ]} if hasattr(m, "tool_calls") and m.tool_calls else {}),
                                        **({"name": m.name} if hasattr(m, "name") and m.name else {}),
                                    }
                                    for m in v
                                ]
                            else:
                                try:
                                    json.dumps(v)
                                    debug_state[k] = v
                                except (TypeError, ValueError):
                                    debug_state[k] = str(v)[:500]
                        yield sse(_debug_payload(
                            {"type": "graph_state_update", "node": node_name},
                            state_delta=debug_state,
                        ))

                    for msg in state_delta.get("messages", []):
                        if isinstance(msg, ToolMessage):
                            server = None
                            if msg.name:
                                for srv_key in MCP_CONNECTIONS:
                                    if msg.name.startswith(srv_key):
                                        server = srv_key
                                        break
                            yield sse(_debug_payload(
                                {
                                    "type": "tool_result",
                                    "name": msg.name,
                                    "content": str(msg.content)[:1000],
                                    **({"server": server} if server else {}),
                                    "agent": node_name,
                                },
                                full_content=str(msg.content),
                                tool_call_id=getattr(msg, "tool_call_id", None),
                            ))

        yield sse({"type": "done"})

    except GraphRecursionError:
        logging.error(
            "Orchestrator hit recursion limit (%s steps)", MAX_AGENT_STEPS
        )
        yield sse({"type": "error",
                   "detail": "This question required more processing than the "
                              "system allows. Try breaking it into smaller, "
                              "more specific questions."})
    except Exception:
        logging.exception("stream_agent error")
        yield sse({"type": "error",
                   "detail": "An internal error occurred. Please try again."})
    finally:
        # Decrement in-flight counter so shutdown / cache-invalidation
        # can know when all active streams have completed.
        async with _inflight_lock:
            _inflight_count -= 1
            if _inflight_count <= 0:
                _inflight_count = 0
                _inflight_zero.set()
