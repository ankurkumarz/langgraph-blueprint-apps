# Reference Architecture: Registry-Driven Multi-Agent Orchestrator with LangGraph

> A comprehensive guide for adopting this architecture — whether migrating an existing
> LangGraph agent or building a new multi-agent system from scratch.

---

## Table of Contents

- [1. Architecture Overview](#1-architecture-overview)
  - [1.1 Core Principles](#11-core-principles)
  - [1.2 System Diagram](#12-system-diagram)
  - [1.3 Component Map](#13-component-map)
- [2. Architecture Deep Dive](#2-architecture-deep-dive)
  - [2.1 Agent Registry Pattern](#21-agent-registry-pattern)
  - [2.2 Sub-Agent ReAct Graph](#22-sub-agent-react-graph)
  - [2.3 Dual Orchestration Strategies](#23-dual-orchestration-strategies)
  - [2.4 SSE Streaming Protocol](#24-sse-streaming-protocol)
  - [2.5 MCP Tool Integration](#25-mcp-tool-integration)
  - [2.6 State Management](#26-state-management)
  - [2.7 Message Trimming](#27-message-trimming)
  - [2.8 Error Handling & Resilience](#28-error-handling--resilience)
  - [2.9 Singleton Graph Cache](#29-singleton-graph-cache)
  - [2.10 Graceful Shutdown & Lifecycle](#210-graceful-shutdown--lifecycle)
  - [2.11 Observability (MLflow Tracing)](#211-observability-mlflow-tracing)
  - [2.12 Configuration Management](#212-configuration-management)
- [3. Path A — Migrate an Existing LangGraph Agent](#3-path-a--migrate-an-existing-langgraph-agent)
  - [3.1 Assessment Checklist](#31-assessment-checklist)
  - [3.2 Step-by-Step Migration](#32-step-by-step-migration)
  - [3.3 Wrapping Your Existing Agent as a Sub-Agent](#33-wrapping-your-existing-agent-as-a-sub-agent)
  - [3.4 Adding SSE Streaming to Your Existing Graph](#34-adding-sse-streaming-to-your-existing-graph)
  - [3.5 Migration Pitfalls](#35-migration-pitfalls)
- [4. Path B — Build a New Multi-Agent System](#4-path-b--build-a-new-multi-agent-system)
  - [4.1 Project Scaffolding](#41-project-scaffolding)
  - [4.2 Define Your Agent Registry](#42-define-your-agent-registry)
  - [4.3 Implement Sub-Agent Graphs](#43-implement-sub-agent-graphs)
  - [4.4 Choose and Build Your Orchestrator](#44-choose-and-build-your-orchestrator)
  - [4.5 Wire Up the API Layer](#45-wire-up-the-api-layer)
  - [4.6 Add the Chat UI](#46-add-the-chat-ui)
- [5. Design Decisions & Trade-offs](#5-design-decisions--trade-offs)
  - [5.1 Evaluator vs Think Mode](#51-evaluator-vs-think-mode)
  - [5.2 When to Use Which Mode](#52-when-to-use-which-mode)
  - [5.3 Extending Beyond Two Modes](#53-extending-beyond-two-modes)
- [6. Production Readiness Checklist](#6-production-readiness-checklist)
- [7. Common Patterns & Recipes](#7-common-patterns--recipes)
  - [7.1 Adding a New Agent (Zero-Code-Change Pattern)](#71-adding-a-new-agent-zero-code-change-pattern)
  - [7.2 Non-MCP Agents (REST APIs, Local Tools)](#72-non-mcp-agents-rest-apis-local-tools)
  - [7.3 Conversation Memory / Multi-Turn](#73-conversation-memory--multi-turn)
  - [7.4 Human-in-the-Loop Approval](#74-human-in-the-loop-approval)
  - [7.5 Custom Orchestrator Strategies](#75-custom-orchestrator-strategies)
- [8. Anti-Patterns to Avoid](#8-anti-patterns-to-avoid)
- [9. Appendix](#9-appendix)
  - [A. Full SSE Event Reference](#a-full-sse-event-reference)
  - [B. Environment Variables Reference](#b-environment-variables-reference)
  - [C. Dependency Matrix](#c-dependency-matrix)
  - [D. File-by-File Architecture Map](#d-file-by-file-architecture-map)

---

## 1. Architecture Overview

### 1.1 Core Principles

| Principle | How This Architecture Implements It |
|---|---|
| **Registry-driven** | All agents are declared in a single config list (`AGENT_CONFIGS`). Adding an agent = adding one dict entry. No graph wiring needed. |
| **Strategy-switchable** | Two orchestration modes (evaluator, think) share the same registry. Switch with one env var. |
| **Streaming-first** | Every node emits structured SSE events via LangGraph's `get_stream_writer()`. The UI is fully decoupled from the backend. |
| **Tool-agnostic** | Sub-agents get tools via MCP servers. Swap, add, or remove tools without touching agent code. |
| **Stateless graph, stateful request** | The compiled graph is cached and shared. Per-request state is passed through `astream()` inputs. |
| **Fail-gracefully** | MCP connectivity errors, recursion limits, and unknown events are all handled with actionable messages. |

### 1.2 System Diagram

```
┌───────────────────────────────────────────────────────────────────────┐
│                        FastAPI Application                            │
│                                                                       │
│   POST /agent/conversation ──► stream_agent(query)                    │
│         │                          │                                  │
│         │  SSE Stream              ▼                                  │
│         │              ┌──────────────────────┐                       │
│         │              │  Orchestrator Graph   │                      │
│         │              │  (evaluator | think)  │                      │
│         │              └──────────┬───────────┘                       │
│         │                         │                                   │
│         │              ┌──────────▼───────────┐                       │
│         │              │   Agent Registry      │                      │
│         │              │   AGENT_CONFIGS[]     │                      │
│         │              └──┬──────────────┬────┘                       │
│         │                 │              │                             │
│         │          ┌──────▼──┐    ┌──────▼──┐                         │
│         │          │ Agent A │    │ Agent B │    ...                   │
│         │          │ (ReAct) │    │ (ReAct) │                         │
│         │          └────┬────┘    └────┬────┘                         │
│         │               │              │                              │
│         │          ┌────▼────┐    ┌────▼────┐                         │
│         │          │ MCP Srv │    │ MCP Srv │                         │
│         │          │  Tools  │    │  Tools  │                         │
│         │          └─────────┘    └─────────┘                         │
│         ▼                                                             │
│   StreamingResponse(SSE)                                              │
└───────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌───────────────────┐
│   Browser / UI    │
│   (SSE Consumer)  │
└───────────────────┘
```

### 1.3 Component Map

| Component | File | Responsibility |
|---|---|---|
| **Agent Registry** | `app/agent.py` → `AGENT_CONFIGS` | Single source of truth for all sub-agents |
| **Sub-Agent Builder** | `app/agent.py` → `_build_subagent_graph()` | Builds identical ReAct graphs per agent |
| **Evaluator Orchestrator** | `app/agent.py` → `build_evaluator_orchestrator()` | Multi-node graph: classify → route → evaluate → synthesize |
| **Think Orchestrator** | `app/agent.py` → `build_think_orchestrator()` | Single ReAct loop with `think_tool` + `delegate` tools |
| **Stream Entrypoint** | `app/agent.py` → `stream_agent()` | Builds graph (cached), runs it, yields SSE |
| **Graph Cache** | `app/agent.py` → `_get_or_build_graph()` | Lazy singleton with invalidation support |
| **API Layer** | `app/app.py` | FastAPI routes, CORS, middleware, static files, lifecycle |
| **Configuration** | `app/settings.py` | Pydantic-settings validation of all env vars |
| **Tracing** | `app/tracing.py` | Conditional MLflow auto-instrumentation |
| **Chat UI** | `static/` | Drop-in SSE consumer with thinking steps, activity panel |

---

## 2. Architecture Deep Dive

### 2.1 Agent Registry Pattern

The registry is the architectural keystone. Every agent is a dict with a fixed schema:

```python
AGENT_CONFIGS = [
    {
        "name": "doc_and_knowledge_search_agent",  # Unique identifier
        "role": "LangChain docs specialist", # Human-readable role (shown in UI)
        "description": "...",               # Used by think-mode LLM to decide delegation
        "system_prompt": "...",             # Injected into the sub-agent's ReAct loop
        "mcp_server": "langchain_docs",     # Key in MCP_CONNECTIONS (settings.py)
        "tool_prefix": "langchain_docs",    # How tools are partitioned to this agent
        "evaluator_intent": "document",     # Intent label for evaluator-mode classifier
    },
]
```

**Why this works:**

1. **Both orchestrators read from it.** Evaluator mode auto-generates graph nodes and edges. Think mode auto-generates the `delegate` tool's agent list.
2. **Tool partitioning is automatic.** `_partition_tools()` assigns MCP tools to agents by prefix matching. No manual tool assignment.
3. **Adding an agent = adding one dict.** No graph rewiring, no conditional edges, no new node functions.

**Key function — tool partitioning:**

```python
def _partition_tools(all_tools: list) -> dict[str, list]:
    """Split MCP tools by agent config tool_prefix. Unmatched → last agent."""
    buckets = {cfg["name"]: [] for cfg in AGENT_CONFIGS}
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
```

### 2.2 Sub-Agent ReAct Graph

Every sub-agent is an identical ReAct graph built by `_build_subagent_graph()`:

```
START → agent_node → tools_condition ──┬──► tool_node → tool_result_emitter → agent_node (loop)
                                       └──► END
```

**Key design decisions:**

| Decision | Implementation | Why |
|---|---|---|
| Same graph shape for all agents | `_build_subagent_graph()` is parameterized by tools, name, prompt, MCP key | Consistency, testability, less code |
| SSE emitted from inside nodes | `get_stream_writer()` in `agent_node` and `tool_result_emitter` | Real-time UI feedback during execution |
| `tool_result_emitter` as separate node | Sits between `tool_node` and looping back to `agent_node` | Gives a hook to inspect and emit tool results as SSE events |
| Message trimming before LLM calls | `_trim_subagent_messages()` keeps last N messages | Prevents context window overflow in long ReAct loops |
| Custom error handler on ToolNode | `handle_mcp_tool_errors` classifies MCP failures | Actionable messages so LLM can retry or inform user |

**The sub-agent node emits these SSE events:**

```python
async def agent_node(state):
    writer = get_stream_writer()
    writer({"type": "llm_start", "agent": agent_name})
    # ... invoke LLM ...
    for tc in response.tool_calls:
        writer({"type": "tool_call", "name": tc["name"], "args": tc["args"],
                "id": tc["id"], "server": mcp_server_key, "agent": agent_name})
    return {"messages": [response]}
```

### 2.3 Dual Orchestration Strategies

#### Evaluator Mode (`ORCHESTRATOR_MODE=evaluator`)

A structured, multi-node graph with explicit control flow:

```
classify_intent
    ├── agent_X ──────────► evaluate ──┬──► synthesize → END
    ├── agent_Y ──────────► evaluate   │
    ├── parallel_agents ──► evaluate   └──► followup_agent_Z → evaluate (loop)
    └── direct_answer ───────────────────► synthesize → END
```

**Nodes and their roles:**

| Node | Purpose |
|---|---|
| `classify_intent` | LLM classifies user intent into one of: agent-specific intents, `"both"` (parallel), or `"general"` (direct answer) |
| `agent_X` / `agent_Y` | Runs one sub-agent from the registry |
| `parallel_agents` | `asyncio.gather()` runs ALL agents concurrently |
| `evaluate` | Evaluator LLM judges if research is complete. Returns `"resolved"` or `"needs_<agent_name>"` |
| `followup_<agent>` | Re-runs a specific agent with a focused follow-up query |
| `synthesize` | Merges multi-source answers or passes through single-source |
| `direct_answer` | For simple queries — answers without tools |

**The evaluate loop:**

```python
EVALUATOR_SYSTEM_PROMPT = """
Reply with EXACTLY one JSON object:
  {"status": "resolved"}
  {"status": "needs_doc_and_knowledge_search_agent", "followup_query": "..."}
  {"status": "needs_web_agent", "followup_query": "..."}
"""
```

Bounded by `MAX_RESOLUTION_ROUNDS` (default: 2) to prevent infinite loops.

#### Think Mode (`ORCHESTRATOR_MODE=think`)

A single ReAct loop — the orchestrator LLM decides everything:

```
START → orchestrator_node → tools_condition ──┬──► tool_node → tool_result_emitter → orchestrator_node
                                              └──► END
```

**The orchestrator has two tools:**

| Tool | Purpose |
|---|---|
| `think_tool(reflection)` | Forces the LLM to reflect on research progress. Returns a structured state summary of all delegations so far. |
| `delegate(agent_name, query)` | Runs a sub-agent and returns its answer. Multiple calls in one response execute in parallel via `ToolNode`. |

**Delegation tracking via ContextVar:**

```python
_think_delegation_log_var: contextvars.ContextVar[list[dict] | None] = (
    contextvars.ContextVar("think_delegation_log", default=None)
)
```

Each `delegate()` call appends to a per-request log. `think_tool()` reads this log and returns a structured summary — forcing the LLM to process actual results rather than hallucinating.

### 2.4 SSE Streaming Protocol

The protocol is tiered and additive:

| Tier | Events | Purpose |
|---|---|---|
| **1 — Core** | `llm_start`, `text`, `done`, `error` | Minimum viable chat |
| **2 — Tools** | `tool_call`, `tool_result` | ReAct loop visibility |
| **3 — Orchestrator** | `agent_start`, `agent_end`, `handoff`, `plan`, `plan_step`, `status`, `mcp_server` | Multi-agent lifecycle |
| **4 — Debug** | `node_response`, `graph_state_update`, `debug` field on any event | Development diagnostics |

**SSE format:**

```python
def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"
```

**Key pattern — `get_stream_writer()` inside graph nodes:**

```python
from langgraph.config import get_stream_writer

async def my_node(state):
    writer = get_stream_writer()
    writer({"type": "status", "message": "Working..."})
    # ... do work ...
    writer({"type": "text", "content": result})
    return {"messages": [response]}
```

These events flow through LangGraph's `astream(stream_mode=["updates", "custom"])` and are forwarded as SSE by `stream_agent()`.

### 2.5 MCP Tool Integration

Tools are discovered from MCP servers at graph build time:

```python
# settings.py — MCP connection configuration
@property
def mcp_connections(self) -> dict[str, dict]:
    return {
        "firecrawl": {
            "transport": "streamable_http",
            "url": self.firecrawl_mcp_url,
        },
        "langchain_docs": {
            "transport": "streamable_http",
            "url": self.langchain_docs_mcp_url,
        },
    }

# agent.py — tool discovery
client = MultiServerMCPClient(MCP_CONNECTIONS)
all_tools = await asyncio.wait_for(
    client.get_tools(), timeout=MCP_INIT_TIMEOUT
)
tool_buckets = _partition_tools(all_tools)
```

**MCP connectivity error handling:**

The architecture defines a comprehensive error handler for the `ToolNode`:

```python
_MCP_CONNECTIVITY_EXCEPTIONS = (
    httpx.NetworkError,       # Connection failures, DNS, socket resets
    httpx.TimeoutException,   # Connect/Read/Write/Pool timeouts
    httpx.HTTPStatusError,    # 5xx responses
    StreamableHTTPError,      # MCP transport failures
    McpError,                 # MCP protocol errors
    ConnectionError,          # Low-level socket issues
    OSError,                  # OS-level network errors
)
```

Each exception type gets a tailored, actionable message that the LLM can use to decide whether to retry, use a different tool, or inform the user.

### 2.6 State Management

Two state schemas serve different roles:

```python
class OrchestratorState(TypedDict):
    """Evaluator-mode state — tracks routing, evaluation, and synthesis."""
    messages: Annotated[list, operator.add]
    intent: str                   # Classified intent
    active_agent: str             # Currently active agent
    final_answer: str             # Direct answer content
    agent_answers: dict           # {agent_name: answer_str}
    resolution_status: str        # "resolved" | "needs_<agent_name>"
    resolution_round: int         # Current evaluate loop iteration
    followup_query: str           # Focused query for follow-up agent

class SubAgentState(TypedDict):
    """Sub-agent and think-mode state — just messages."""
    messages: Annotated[list, operator.add]
```

**Key insight:** `OrchestratorState` carries rich metadata for the evaluator's multi-step flow. `SubAgentState` is deliberately minimal — sub-agents and the think-mode orchestrator only need message history.

### 2.7 Message Trimming

Long ReAct loops can exceed context windows. The architecture uses `trim_messages` with two configurations:

```python
# Sub-agents: smaller window (default 30 messages)
trim_messages(msgs, max_tokens=MAX_SUBAGENT_MESSAGES, token_counter=len,
              strategy="last", include_system=True, start_on=("human", "ai"))

# Think-mode orchestrator: larger window (default 50 messages)
trim_messages(msgs, max_tokens=MAX_ORCHESTRATOR_MESSAGES, token_counter=len,
              strategy="last", include_system=True, start_on=("human", "ai"))
```

**Design choices:**
- `token_counter=len` counts by message count (cheap and predictable). For token-precise trimming, switch to `"approximate"`.
- `include_system=True` ensures the system prompt always survives trimming.
- `start_on=("human", "ai")` prevents orphaned `ToolMessage` at the start of the trimmed window.

### 2.8 Error Handling & Resilience

| Concern | Implementation |
|---|---|
| **MCP server down** | `handle_mcp_tool_errors()` returns actionable messages; LLM can retry with a different tool or inform user |
| **MCP connectivity in results** | `tool_result_emitter` scans for connectivity markers and emits `mcp_server` disconnected events for the UI |
| **Sub-agent recursion limit** | `GraphRecursionError` caught in `_invoke_agent()` and `delegate()`; best partial answer returned |
| **Orchestrator recursion limit** | `GraphRecursionError` caught in `stream_agent()`; user gets actionable error message |
| **Evaluator infinite loop** | `MAX_RESOLUTION_ROUNDS` forces `"resolved"` after N evaluate cycles |
| **Evaluator parse failure** | Unparseable evaluator output defaults to `{"status": "resolved"}` |
| **Unknown SSE events** | UI renders any unknown event as informational step — never breaks |
| **General exceptions** | `stream_agent()` catches all exceptions and yields an SSE error event |

### 2.9 Singleton Graph Cache

```python
_cached_graph = None
_graph_lock = asyncio.Lock()

async def _get_or_build_graph():
    global _cached_graph
    if _cached_graph is not None:
        return _cached_graph
    async with _graph_lock:
        if _cached_graph is not None:  # Double-check after lock
            return _cached_graph
        # ... build graph (MCP discovery + compilation) ...
        _cached_graph = graph
        return _cached_graph
```

**Why cache the graph?**
- MCP tool discovery requires connecting to each server — expensive per-request.
- LangGraph graph compilation is deterministic for a given config — no reason to repeat.
- The compiled graph is stateless — state is passed per invocation.

**Invalidation:**
- `invalidate_graph_cache()` sets `_cached_graph = None`
- Next request rebuilds with potentially updated config
- Available via admin API: `POST /api/admin/invalidate-cache`
- Also triggered by `SIGHUP` signal for hot-reload

### 2.10 Graceful Shutdown & Lifecycle

```python
# In-flight request tracking
_inflight_count: int = 0
_inflight_lock = asyncio.Lock()
_inflight_zero = asyncio.Event()
_shutting_down: bool = False
```

**Shutdown flow:**

1. `begin_shutdown()` sets `_shutting_down = True` — new requests rejected
2. `wait_for_inflight(timeout)` waits for active streams to complete
3. `invalidate_graph_cache()` releases graph references
4. Process exits cleanly

**Health probes:**

| Endpoint | Purpose | During Shutdown |
|---|---|---|
| `GET /healthz` | Liveness — process alive? | Always 200 |
| `GET /readyz` | Readiness — accepting traffic? | Returns 503 |

### 2.11 Observability (MLflow Tracing)

```python
def setup_mlflow_tracing() -> bool:
    if not enabled:
        return False
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    autolog(log_traces=True)  # Auto-traces every LangChain/LangGraph call
    return True
```

**Key design:** Tracing is activated *before* any LangChain imports (in `app.py`'s top-level imports). This ensures the auto-instrumentation hooks are in place before any LLM or graph objects are created.

### 2.12 Configuration Management

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Required (startup fails if missing)
    firecrawl_api_key: SecretStr
    google_api_key: SecretStr
    
    # Optional with validated defaults
    orchestrator_mode: Literal["evaluator", "think"] = "evaluator"
    max_agent_steps: int = Field(default=25, ge=1)
    # ...
    
    # Computed/derived
    @computed_field
    @property
    def firecrawl_mcp_url(self) -> str:
        return f"https://mcp.firecrawl.dev/{self.firecrawl_api_key.get_secret_value()}/v2/mcp"

settings = Settings()  # Singleton — crashes immediately on validation error
```

**Benefits:**
- **Fail-fast**: Missing required vars crash at import time, not at first request
- **Type safety**: `int`, `bool`, `Literal` constraints validated automatically
- **Secret handling**: `SecretStr` prevents accidental logging of API keys
- **Single source of truth**: One `settings` object imported everywhere

---

## 3. Path A — Migrate an Existing LangGraph Agent

### 3.1 Assessment Checklist

Before migrating, evaluate your existing agent:

| Question | If Yes | If No |
|---|---|---|
| Does your agent have a `StateGraph`? | You can wrap it as a sub-agent directly | Build a new graph or use `create_react_agent` |
| Do you use `astream()` for streaming? | Migration is straightforward | You'll need to switch from `ainvoke()` to `astream()` |
| Do you have multiple tool groups? | Natural fit for multi-agent split | Consider whether multi-agent is needed |
| Do you use MCP for tools? | Tools integrate directly | You'll need to either add MCP or use local tools (see §7.2) |
| Do you have custom state beyond messages? | Map to `OrchestratorState` or extend it | `SubAgentState` works as-is |

### 3.2 Step-by-Step Migration

#### Step 1: Extract Your Agent as a Sub-Agent

Take your existing graph and wrap it as a sub-agent entry:

```python
# BEFORE: Your existing monolithic agent
llm = ChatOpenAI(model="gpt-4o").bind_tools(all_tools)

builder = StateGraph(MyState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(all_tools))
# ... edges ...
graph = builder.compile()
```

```python
# AFTER: Your agent becomes one entry in the registry
AGENT_CONFIGS = [
    {
        "name": "my_existing_agent",
        "role": "Your agent's role description",
        "description": "When to use this agent — used by orchestrator",
        "system_prompt": "Your existing system prompt",
        "mcp_server": "your_mcp_server",
        "tool_prefix": "your_tool_prefix",
        "evaluator_intent": "your_intent",
    },
]
```

#### Step 2: Add SSE Emission to Your Nodes

If you have custom nodes, add `get_stream_writer()` calls:

```python
# BEFORE: Silent node
async def my_node(state):
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}

# AFTER: Node with SSE events
async def my_node(state):
    writer = get_stream_writer()
    writer({"type": "llm_start", "agent": "my_agent"})
    
    response = await llm.ainvoke(state["messages"])
    
    for tc in response.tool_calls:
        writer({"type": "tool_call", "name": tc["name"],
                "args": tc["args"], "agent": "my_agent"})
    
    return {"messages": [response]}
```

#### Step 3: Switch from `ainvoke` to `astream` in Your Endpoint

```python
# BEFORE
result = await graph.ainvoke({"messages": [HumanMessage(content=query)]})
return {"response": result["messages"][-1].content}

# AFTER
async def stream_agent(query: str):
    async for mode, chunk in graph.astream(
        {"messages": [HumanMessage(content=query)]},
        stream_mode=["updates", "custom"],
    ):
        if mode == "custom":
            yield sse(chunk)
    yield sse({"type": "done"})
```

#### Step 4: Add the Registry and Orchestrator Layer

Copy the registry pattern from this architecture and register your agent(s). If you only have one agent, the orchestrator simply delegates everything to it — you still get the SSE streaming, UI, error handling, and caching infrastructure.

### 3.3 Wrapping Your Existing Agent as a Sub-Agent

If your agent has custom state or a non-standard graph shape, you can wrap it:

```python
def _build_subagent_graph(tools, agent_name, system_prompt, mcp_server_key):
    """Replace this function to use your custom graph shape."""
    
    # Option A: Use the standard ReAct pattern (recommended)
    # ... (use the existing _build_subagent_graph implementation) ...
    
    # Option B: Wrap your custom graph
    your_graph = build_your_custom_graph(tools, system_prompt)
    return your_graph  # Must accept SubAgentState and return SubAgentState
```

**Requirements for wrapped sub-agent graphs:**
1. Accept `SubAgentState` (or compatible) as input
2. Support `astream(stream_mode=["custom", "updates"])`
3. Use `get_stream_writer()` if you want SSE events from inside nodes

### 3.4 Adding SSE Streaming to Your Existing Graph

If you want to keep your existing graph but add SSE streaming, here's the minimal change:

```python
# Your existing graph builder — add one line per node
from langgraph.config import get_stream_writer

async def your_existing_node(state):
    writer = get_stream_writer()
    writer({"type": "llm_start"})         # ← Add this
    
    # ... your existing logic unchanged ...
    
    if not response.tool_calls:
        writer({"type": "text", "content": response.content})  # ← Add this
    
    return {"messages": [response]}

# Your existing endpoint — change the streaming call
async def stream_endpoint(query: str):
    async for mode, chunk in your_graph.astream(
        inputs,
        stream_mode=["updates", "custom"],  # ← Key change
    ):
        if mode == "custom":
            yield sse(chunk)
        elif mode == "updates":
            for node_name, delta in chunk.items():
                for msg in delta.get("messages", []):
                    if isinstance(msg, ToolMessage):
                        yield sse({"type": "tool_result", "name": msg.name,
                                   "content": str(msg.content)[:1000]})
    yield sse({"type": "done"})
```

### 3.5 Migration Pitfalls

| Pitfall | Cause | Solution |
|---|---|---|
| **Missing SSE events** | `stream_mode` doesn't include `"custom"` | Always use `stream_mode=["updates", "custom"]` |
| **UI shows nothing** | No `text` event emitted before `done` | Ensure your final answer node emits `{"type": "text", "content": "..."}` |
| **Context window overflow** | Long ReAct loops accumulate messages | Add `_trim_subagent_messages()` before every LLM call |
| **Graph rebuilds every request** | No caching | Implement `_get_or_build_graph()` singleton pattern |
| **Tool errors crash the stream** | No error handler on `ToolNode` | Add `handle_tool_errors=your_error_handler` to `ToolNode()` |
| **Concurrent request state bleed** | Mutable state shared across requests | Use `ContextVar` for per-request state (like `_think_delegation_log_var`) |

---

## 4. Path B — Build a New Multi-Agent System

### 4.1 Project Scaffolding

```
my-multi-agent/
├── main.py              # uvicorn entrypoint
├── app/
│   ├── __init__.py
│   ├── agent.py         # Orchestrator + sub-agents + registry + SSE
│   ├── app.py           # FastAPI routes, CORS, middleware, lifecycle
│   ├── settings.py      # Centralised config (pydantic-settings)
│   └── tracing.py       # Optional observability (MLflow / LangSmith)
├── static/              # Drop-in chat UI
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── pyproject.toml
└── .env
```

**Core dependencies:**

```toml
[project]
requires-python = ">=3.12"
dependencies = [
    "langgraph>=1.0.4",
    "langgraph-prebuilt>=1.0.4",
    "langchain>=1.2.6",
    "langchain-google-genai>=4.0.0",   # or langchain-openai, langchain-anthropic
    "langchain-mcp-adapters>=0.2.1",
    "fastapi>=0.124.0",
    "uvicorn[standard]>=0.30.0",
    "pydantic-settings>=2.14.0",
]
```

### 4.2 Define Your Agent Registry

Start by identifying your domain's specialist agents:

```python
# Step 1: List your agents
AGENT_CONFIGS = [
    {
        "name": "db_agent",
        "role": "Database specialist",
        "description": "SQL queries, schema analysis, performance tuning, migrations",
        "system_prompt": (
            "You are a database specialist with access to PostgreSQL tools.\n"
            "- Write and execute SQL queries to answer data questions.\n"
            "- Analyze query plans for performance issues.\n"
            "- Suggest schema improvements when relevant."
        ),
        "mcp_server": "postgres_mcp",
        "tool_prefix": "postgres",
        "evaluator_intent": "database",
    },
    {
        "name": "k8s_agent",
        "role": "Kubernetes specialist",
        "description": "Pod troubleshooting, deployments, services, networking",
        "system_prompt": "You are a Kubernetes expert...",
        "mcp_server": "k8s_mcp",
        "tool_prefix": "k8s",
        "evaluator_intent": "kubernetes",
    },
    {
        "name": "docs_agent",
        "role": "Documentation specialist",
        "description": "Internal docs, runbooks, architecture decisions",
        "system_prompt": "You are a documentation specialist...",
        "mcp_server": "docs_mcp",
        "tool_prefix": "docs",
        "evaluator_intent": "documentation",
    },
]
```

```python
# Step 2: Configure MCP connections in settings.py
@property
def mcp_connections(self) -> dict[str, dict]:
    return {
        "postgres_mcp": {
            "transport": "streamable_http",
            "url": "http://localhost:3001/mcp",
        },
        "k8s_mcp": {
            "transport": "streamable_http",
            "url": "http://localhost:3002/mcp",
        },
        "docs_mcp": {
            "transport": "streamable_http",
            "url": "http://localhost:3003/mcp",
        },
    }
```

### 4.3 Implement Sub-Agent Graphs

Use the standard ReAct builder (copy `_build_subagent_graph` from this project). The function creates identical graphs parameterized by:

- `tools` — the MCP tools assigned to this agent
- `agent_name` — for SSE event attribution
- `system_prompt` — injected as `SystemMessage`
- `mcp_server_key` — for SSE `server` field attribution

No per-agent customization needed for most cases. If you need a specialized agent (e.g., one that uses a different LLM or has custom logic), override just that agent's graph:

```python
def _build_agent_registry(tool_buckets):
    registry = {}
    for cfg in AGENT_CONFIGS:
        if cfg["name"] == "special_agent":
            registry[cfg["name"]] = {
                "graph": _build_custom_graph(tool_buckets[cfg["name"]], cfg),
                **cfg,
            }
        else:
            registry[cfg["name"]] = {
                "graph": _build_subagent_graph(
                    tools=tool_buckets[cfg["name"]],
                    agent_name=cfg["name"],
                    system_prompt=cfg["system_prompt"],
                    mcp_server_key=cfg["mcp_server"],
                ),
                **cfg,
            }
    return registry
```

### 4.4 Choose and Build Your Orchestrator

**Decision matrix:**

| Factor | Choose Evaluator | Choose Think |
|---|---|---|
| You want explicit control flow | ✅ | |
| You want the LLM to decide strategy | | ✅ |
| You need deterministic routing | ✅ | |
| You want minimal graph nodes | | ✅ (4 nodes fixed) |
| You need cross-agent evaluation | ✅ | |
| You want parallel execution | Both support it | Both support it |
| You want to start simple | | ✅ |

**Recommendation:** Start with think mode (simpler, fewer nodes). Switch to evaluator mode when you need explicit control over routing and evaluation.

### 4.5 Wire Up the API Layer

```python
# app.py — minimal version
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    query: str

@app.post("/agent/conversation")
async def conversation(request: ChatRequest):
    return StreamingResponse(
        stream_agent(request.query),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

Then add progressively:
1. **CORS middleware** — for cross-origin UI access
2. **Static file serving** — mount the chat UI
3. **Health endpoints** — `/healthz`, `/readyz`
4. **Admin endpoints** — cache invalidation, shutdown
5. **Security middleware** — headers, API key auth
6. **Lifespan handler** — graceful startup/shutdown

### 4.6 Add the Chat UI

1. Copy the `static/` folder from this project
2. Mount it: `app.mount("/agent/copilot", StaticFiles(directory="static", html=True))`
3. Update branding in `index.html`
4. Update sample questions
5. Remove/replace the Insights and Monitoring dashboard tabs if not applicable

The UI works with just Tier 1 events (`llm_start`, `text`, `done`, `error`). Add higher tiers incrementally for richer visualisation.

---

## 5. Design Decisions & Trade-offs

### 5.1 Evaluator vs Think Mode

| Dimension | Evaluator Mode | Think Mode |
|---|---|---|
| **Decision maker** | Separate classifier + evaluator LLMs | Single orchestrator LLM |
| **Graph complexity** | ~10+ nodes (scales with agent count) | 4 nodes (fixed) |
| **Routing control** | Deterministic (LLM classifies → code routes) | Emergent (LLM decides via tool calls) |
| **Parallel execution** | Explicit `asyncio.gather` in `parallel_agents` node | Implicit — multiple `delegate()` calls in one LLM response |
| **Follow-up strategy** | Evaluator returns `needs_X` → code routes to `followup_X` → re-evaluate | LLM calls `think_tool` → decides to call `delegate` again |
| **Context handling** | Follow-up includes other agents' research as context | LLM accumulates context in its own message history |
| **Token usage** | More LLM calls (classifier + evaluator + synthesizer) | Fewer LLM calls but longer context per call |
| **Debuggability** | Easier — each node has a clear purpose | Harder — LLM reasoning is opaque |
| **Extensibility** | Add routing rules when adding agents | No routing changes needed |

### 5.2 When to Use Which Mode

**Use Evaluator when:**
- You have well-defined agent specializations (clear intent boundaries)
- You need auditable, deterministic routing decisions
- You want explicit quality gates (the evaluate step)
- You need to synthesize across multiple sources reliably
- You're building for production with compliance requirements

**Use Think when:**
- Your agents have overlapping capabilities
- You want the LLM to discover the best strategy dynamically
- You're prototyping or iterating quickly
- You have complex queries that don't map cleanly to one agent
- You want minimal boilerplate and maximal flexibility

### 5.3 Extending Beyond Two Modes

The architecture supports adding new orchestration strategies:

```python
# In _get_or_build_graph():
if ORCHESTRATOR_MODE == "think":
    graph = build_think_orchestrator(registry)
elif ORCHESTRATOR_MODE == "evaluator":
    graph = build_evaluator_orchestrator(registry)
elif ORCHESTRATOR_MODE == "planner":
    graph = build_planner_orchestrator(registry)  # Your custom mode
else:
    raise ValueError(f"Unknown mode: {ORCHESTRATOR_MODE}")
```

All modes share the same registry, sub-agent graphs, SSE protocol, and caching infrastructure.

---

## 6. Production Readiness Checklist

| Category | Item | Status in This Architecture |
|---|---|---|
| **Configuration** | Env vars validated at startup | ✅ pydantic-settings with fail-fast |
| **Configuration** | Secrets not logged | ✅ `SecretStr` type |
| **Configuration** | Hot-reload without restart | ✅ `SIGHUP` → cache invalidation |
| **Security** | CORS configured | ✅ Configurable origins |
| **Security** | Security headers | ✅ Middleware (X-Content-Type-Options, X-Frame-Options, etc.) |
| **Security** | Admin endpoints authenticated | ✅ API key with timing-safe comparison |
| **Security** | Input validation | ✅ Pydantic model with max length |
| **Resilience** | Graceful shutdown | ✅ In-flight tracking + drain |
| **Resilience** | MCP server failures | ✅ Per-exception-type error messages |
| **Resilience** | Recursion limits | ✅ Configurable per-level (orchestrator + sub-agent) |
| **Resilience** | Message trimming | ✅ Prevents context overflow |
| **Observability** | Structured logging | ✅ Python logging module |
| **Observability** | Distributed tracing | ✅ MLflow autolog (optional) |
| **Observability** | Health probes | ✅ `/healthz` + `/readyz` |
| **Observability** | Debug mode | ✅ Enriched SSE events when enabled |
| **Performance** | Graph caching | ✅ Singleton with double-check lock |
| **Performance** | Async throughout | ✅ All I/O is async |
| **Deployment** | Container-ready | ✅ uvicorn entrypoint, env-var config |
| **Deployment** | Kubernetes-ready | ✅ Health probes, graceful shutdown, SIGTERM handling |

**What to add for your deployment:**

- [ ] Rate limiting (per-IP or per-user)
- [ ] Authentication / authorization (JWT, OAuth)
- [ ] Conversation memory persistence (database-backed checkpointer)
- [ ] Request/response logging (structured JSON logs)
- [ ] Metrics export (Prometheus, OpenTelemetry)
- [ ] Load testing (establish baseline latency and throughput)
- [ ] Circuit breakers around MCP servers
- [ ] Input sanitization beyond length limits

---

## 7. Common Patterns & Recipes

### 7.1 Adding a New Agent (Zero-Code-Change Pattern)

This is the simplest extension point. Add one entry to `AGENT_CONFIGS`:

```python
{
    "name": "monitoring_agent",
    "role": "Infrastructure monitoring specialist",
    "description": (
        "Monitoring specialist — use for Prometheus queries, alert analysis, "
        "dashboard interpretation, and SLO tracking."
    ),
    "system_prompt": (
        "You are **MonitoringAgent**, a specialist in infrastructure monitoring.\n"
        "You have access to Prometheus and Grafana tools.\n\n"
        "Instructions:\n"
        "- Query Prometheus for metrics using PromQL.\n"
        "- Analyze alert firing patterns.\n"
        "- Provide SLO reports when asked.\n"
        "- Suggest alert tuning when you see noisy alerts."
    ),
    "mcp_server": "monitoring_mcp",
    "tool_prefix": "monitoring",
    "evaluator_intent": "monitoring",
}
```

Then add the MCP connection in `settings.py`:

```python
@property
def mcp_connections(self) -> dict[str, dict]:
    return {
        # ... existing connections ...
        "monitoring_mcp": {
            "transport": "streamable_http",
            "url": self.monitoring_mcp_url,
        },
    }
```

**That's it.** Both orchestrator modes auto-discover the new agent:
- Evaluator mode generates a new node, follow-up node, and routing edges
- Think mode adds the agent to the `delegate` tool's available agents list

### 7.2 Non-MCP Agents (REST APIs, Local Tools)

If your tools aren't served via MCP, define them as regular LangChain tools:

```python
from langchain_core.tools import tool

@tool
async def query_database(sql: str) -> str:
    """Execute a SQL query against the production database."""
    async with get_db_connection() as conn:
        result = await conn.fetch(sql)
        return json.dumps([dict(r) for r in result])

@tool
async def check_pod_status(namespace: str, pod_name: str) -> str:
    """Check the status of a Kubernetes pod."""
    # Call kubectl or K8s API directly
    result = await k8s_client.read_namespaced_pod_status(pod_name, namespace)
    return json.dumps(result.to_dict())
```

**Modify `_partition_tools` to handle both MCP and local tools:**

```python
# In your agent config, add a "local_tools" field:
{
    "name": "db_agent",
    "local_tools": [query_database],  # Non-MCP tools
    "mcp_server": None,               # No MCP server
    "tool_prefix": "db",
    # ...
}

# Adjust _build_agent_registry:
def _build_agent_registry(tool_buckets):
    registry = {}
    for cfg in AGENT_CONFIGS:
        mcp_tools = tool_buckets.get(cfg["name"], [])
        local_tools = cfg.get("local_tools", [])
        all_tools = mcp_tools + local_tools
        registry[cfg["name"]] = {
            "graph": _build_subagent_graph(
                tools=all_tools,
                agent_name=cfg["name"],
                system_prompt=cfg["system_prompt"],
                mcp_server_key=cfg.get("mcp_server", "local"),
            ),
            **cfg,
        }
    return registry
```

### 7.3 Conversation Memory / Multi-Turn

This architecture processes single-turn queries. To add multi-turn:

```python
from langgraph.checkpoint.memory import MemorySaver

# Option 1: In-memory (development)
checkpointer = MemorySaver()

# Option 2: PostgreSQL (production)
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
checkpointer = AsyncPostgresSaver.from_conn_string(DATABASE_URL)

# Apply to graph compilation:
graph = builder.compile(checkpointer=checkpointer)

# Pass thread_id per conversation:
async for mode, chunk in graph.astream(
    inputs,
    stream_mode=["updates", "custom"],
    config={
        "configurable": {"thread_id": conversation_id},
        "recursion_limit": MAX_AGENT_STEPS,
    },
):
    ...
```

**API change — accept conversation ID:**

```python
class ChatRequest(BaseModel):
    query: str
    conversation_id: str | None = None

@app.post("/agent/conversation")
async def conversation(request: ChatRequest):
    conv_id = request.conversation_id or str(uuid.uuid4())
    return StreamingResponse(
        stream_agent(request.query, conversation_id=conv_id),
        media_type="text/event-stream",
    )
```

### 7.4 Human-in-the-Loop Approval

Add approval gates for dangerous operations:

```python
from langgraph.types import interrupt

async def agent_node_with_approval(state):
    writer = get_stream_writer()
    response = await llm.ainvoke(state["messages"])
    
    # Check if any tool call needs approval
    dangerous_tools = {"delete_resource", "scale_deployment", "run_migration"}
    for tc in response.tool_calls:
        if tc["name"] in dangerous_tools:
            writer({"type": "approval_request",
                     "action": tc["name"],
                     "detail": json.dumps(tc["args"])})
            # Pause execution until human approves
            approval = interrupt({"tool": tc["name"], "args": tc["args"]})
            writer({"type": "approval_result", "approved": approval.get("approved", False)})
            if not approval.get("approved"):
                return {"messages": [AIMessage(content="Action was rejected by operator.")]}
    
    return {"messages": [response]}
```

### 7.5 Custom Orchestrator Strategies

**Planner-Executor pattern:**

```python
def build_planner_orchestrator(registry):
    """
    A planner LLM creates a step-by-step plan, then executes each step
    sequentially by delegating to the appropriate agent.
    """
    async def plan_node(state):
        writer = get_stream_writer()
        # LLM generates a JSON plan: [{agent, query}, ...]
        plan = await planner_llm.ainvoke(...)
        writer({"type": "plan", "steps": [
            {"id": f"step_{i}", "label": step["query"][:60]}
            for i, step in enumerate(plan)
        ]})
        return {"plan": plan}
    
    async def execute_node(state):
        writer = get_stream_writer()
        for i, step in enumerate(state["plan"]):
            writer({"type": "plan_step", "step_id": f"step_{i}", "status": "running"})
            answer = await _invoke_agent(registry, step["agent"],
                                         [HumanMessage(content=step["query"])], writer)
            writer({"type": "plan_step", "step_id": f"step_{i}", "status": "done"})
        return {"agent_answers": {...}}
    
    # plan → execute → synthesize → END
```

**Router-only pattern (no evaluation):**

```python
def build_router_orchestrator(registry):
    """Simplified: classify → single agent → done. No evaluation loop."""
    # classify → agent_X | agent_Y | direct_answer → END
```

---

## 8. Anti-Patterns to Avoid

| Anti-Pattern | Why It's Bad | What to Do Instead |
|---|---|---|
| **Rebuilding the graph per request** | MCP tool discovery is expensive; graph compilation is redundant | Use the singleton cache pattern (§2.9) |
| **Shared mutable state across requests** | Concurrent requests corrupt each other | Use `ContextVar` for per-request state |
| **No message trimming** | Long ReAct loops overflow context windows | Apply `trim_messages` before every LLM call |
| **Catching all exceptions silently** | Failures are invisible | Always emit SSE error events so the UI can display them |
| **Hardcoded tool-to-agent mapping** | Breaks when MCP tools change | Use prefix-based partitioning |
| **Synchronous LLM calls** | Blocks the event loop, kills concurrency | Use `ainvoke()` exclusively |
| **Monolithic agent with all tools** | LLM confused by too many tools, poor tool selection | Split into specialist sub-agents with focused tool sets |
| **No recursion limits** | Runaway ReAct loops consume budget | Set limits at both orchestrator and sub-agent levels |
| **Logging secrets** | API keys exposed in logs | Use `SecretStr` and never `.get_secret_value()` in log statements |
| **Blocking startup on MCP** | If any MCP server is slow/down, app never starts | Use timeout on tool discovery with graceful degradation |

---

## 9. Appendix

### A. Full SSE Event Reference

| Event | Tier | Fields | Purpose |
|---|---|---|---|
| `llm_start` | 1 | `agent?` | LLM invocation started |
| `text` | 1 | `content` | Final answer text (accumulates) |
| `done` | 1 | — | Stream complete |
| `error` | 1 | `detail` | Error message |
| `tool_call` | 2 | `name`, `args`, `id?`, `agent?`, `server?` | Tool invocation |
| `tool_result` | 2 | `name`, `content`, `agent?`, `server?` | Tool response |
| `agent_start` | 3 | `agent`, `role?` | Sub-agent begins |
| `agent_end` | 3 | `agent`, `summary?` | Sub-agent completes |
| `handoff` | 3 | `from`, `to`, `reason?` | Control transfer |
| `plan` | 3 | `steps: [{id, label}]`, `agent?` | Execution plan |
| `plan_step` | 3 | `step_id`, `status` | Plan step update |
| `status` | 3 | `message`, `agent?`, `progress?` | Progress info |
| `mcp_server` | 3 | `server`, `status`, `error?` | MCP connectivity |
| `node_response` | 4 | `node`, `agent?`, `debug` | Raw LLM output |
| `graph_state_update` | 4 | `node`, `debug` | Graph state delta |

Any event can carry `"debug": {...}` for enriched diagnostics when debug mode is on.

### B. Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `GOOGLE_API_KEY` | ✅ | — | Google AI API key |
| `FIRECRAWL_API_KEY` | ✅ | — | Firecrawl API key |
| `ORCHESTRATOR_MODE` | | `evaluator` | `evaluator` or `think` |
| `GEMINI_MODEL` | | `gemini-2.5-flash` | Gemini model name |
| `MAX_AGENT_STEPS` | | `25` | Orchestrator recursion limit |
| `MAX_SUBAGENT_STEPS` | | `50` | Sub-agent recursion limit |
| `MAX_RESOLUTION_ROUNDS` | | `2` | Max evaluate → follow-up loops |
| `MAX_SUBAGENT_MESSAGES` | | `30` | Sub-agent message window |
| `MAX_ORCHESTRATOR_MESSAGES` | | `50` | Orchestrator message window |
| `MCP_INIT_TIMEOUT` | | `30.0` | MCP tool discovery timeout (seconds) |
| `ALLOWED_ORIGINS` | | `""` | Comma-separated CORS origins |
| `MAX_QUERY_LENGTH` | | `4000` | Max input query length |
| `DEBUG_MODE` | | `false` | Enable debug SSE events |
| `ADMIN_API_KEY` | | *(auto-generated)* | API key for admin endpoints |
| `SHUTDOWN_TIMEOUT` | | `30.0` | Max wait for in-flight drain |
| `LANGCHAIN_DOCS_MCP_URL` | | `https://docs.langchain.com/mcp` | LangChain docs MCP URL |
| `MLFLOW_ENABLED` | | `false` | Enable MLflow tracing |
| `MLFLOW_TRACKING_URI` | | `http://localhost:5050` | MLflow server URL |
| `MLFLOW_EXPERIMENT_NAME` | | `langgraph-fastapi-streaming` | MLflow experiment |

### C. Dependency Matrix

| Package | Version | Purpose |
|---|---|---|
| `langgraph` | ≥1.0.4 | Graph compilation, `StateGraph`, `astream` |
| `langgraph-prebuilt` | ≥1.0.4 | `ToolNode`, `tools_condition` |
| `langchain` | ≥1.2.6 | Core: messages, tools, `trim_messages` |
| `langchain-google-genai` | ≥4.0.0 | Gemini LLM integration |
| `langchain-mcp-adapters` | ≥0.2.1 | `MultiServerMCPClient`, MCP tool discovery |
| `fastapi` | ≥0.124.0 | API framework, SSE streaming |
| `uvicorn` | ≥0.30.0 | ASGI server |
| `pydantic-settings` | ≥2.14.0 | Environment validation |
| `mlflow` | ≥3.7.0 | Optional tracing |

**Swapping the LLM provider:**

Replace `langchain-google-genai` with your provider:

```python
# OpenAI
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")

# Anthropic
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-sonnet-4-20250514")

# Azure OpenAI
from langchain_openai import AzureChatOpenAI
llm = AzureChatOpenAI(deployment_name="gpt-4o")
```

### D. File-by-File Architecture Map

```
main.py
 └── Entrypoint: imports app, runs uvicorn with dev reload

app/__init__.py
 └── Package marker

app/settings.py
 └── Settings (BaseSettings)
      ├── Required: firecrawl_api_key, google_api_key
      ├── Optional: orchestrator_mode, max_agent_steps, ...
      ├── Computed: firecrawl_mcp_url, mcp_connections
      ├── Validators: _normalise_mode, _parse_bool, _generate_admin_key
      └── Singleton: settings = Settings()

app/tracing.py
 └── setup_mlflow_tracing()
      ├── Conditional on MLFLOW_ENABLED=true
      ├── Sets tracking URI + experiment
      └── Enables autolog(log_traces=True)

app/app.py
 └── FastAPI application
      ├── Lifespan: startup logging, SIGHUP handler, graceful shutdown drain
      ├── Middleware: CORS, SecurityHeaders
      ├── POST /agent/conversation     → SSE from stream_agent()
      ├── GET  /api/debug              → debug mode state
      ├── POST /api/admin/invalidate   → force graph rebuild
      ├── POST /api/admin/shutdown     → graceful shutdown
      ├── GET  /healthz                → liveness probe
      ├── GET  /readyz                 → readiness probe
      └── Static: /agent/copilot       → chat UI

app/agent.py
 └── Core agent logic
      ├── AGENT_CONFIGS[]              — agent registry
      ├── _partition_tools()           — assign tools by prefix
      ├── State schemas                — OrchestratorState, SubAgentState
      ├── _build_subagent_graph()      — ReAct graph per agent
      ├── _build_agent_registry()      — compiled sub-graphs
      ├── _invoke_agent()              — run a sub-agent with SSE
      ├── build_evaluator_orchestrator()
      │    ├── classify_intent         — LLM classifies user intent
      │    ├── route_after_classify    — conditional edges
      │    ├── agent nodes (dynamic)   — one per AGENT_CONFIGS entry
      │    ├── parallel_agents         — asyncio.gather all agents
      │    ├── direct_answer           — no-tool response
      │    ├── evaluate                — LLM judges completeness
      │    ├── followup nodes (dynamic)— one per agent
      │    └── synthesize              — merge multi-source answers
      ├── build_think_orchestrator()
      │    ├── think_tool              — reflection with state summary
      │    ├── delegate                — run sub-agent, track in ContextVar
      │    └── orchestrator_agent_node — single ReAct loop
      ├── Graph cache                  — _get_or_build_graph(), invalidate
      ├── In-flight tracking           — count, lock, drain
      └── stream_agent()              — async generator → SSE strings

static/
 ├── index.html   — page structure, sidebar, tabs
 ├── styles.css   — all visual styling
 └── app.js       — SSE handling, rendering, event handlers
      ├── handleEvent()        — extensible SSE event switch
      ├── friendlyArgs()       — auto-pretty tool args
      ├── friendlyResult()     — auto-summarise tool results
      ├── renderMarkdown()     — lightweight MD → HTML
      └── addDebugBlock()      — syntax-highlighted debug JSON
```

---

## Contributing

This reference architecture is designed to be forked and adapted. When extending:

1. **Keep the registry pattern.** It's the foundation for zero-code agent additions.
2. **Emit SSE events from nodes.** The UI consumes them. Unknown events render gracefully.
3. **Test both orchestrator modes.** They share the registry, so changes should work in both.
4. **Validate config at startup.** Use pydantic-settings for any new environment variables.
5. **Handle errors explicitly.** Every failure path should emit an SSE error event.

---

*This guide is part of the [langgraph-blueprint-apps](https://github.com/langgraph-blueprint-apps) project.
Reference implementation: `multiagent-orchestrator-streaming/`*
