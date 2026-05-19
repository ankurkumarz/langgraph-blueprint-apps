# SSE Streaming Event Protocol — Integration Reference

A self-contained reference for developers adopting the streaming multi-agent
UX pattern from this reference application into their own LangGraph-based
system.

This document covers:
- What events to emit from your backend graph nodes
- What fields each event requires
- How a frontend (or any SSE consumer) should handle each event
- The recommended adoption order (tiered)

---

## Overview

The protocol is a thin layer of structured JSON events streamed over
Server-Sent Events (SSE). Backend graph nodes emit events using LangGraph's
`get_stream_writer()`. The stream entrypoint forwards them to the HTTP
response. A frontend renders each event type differently.

```
LangGraph node
  writer({"type": "tool_call", "name": "search", ...})
    ↓  LangGraph custom stream mode
  stream entrypoint yields  data: {"type":"tool_call","name":"search",...}
    ↓  HTTP SSE
  frontend switch(evt.type) → case 'tool_call': render tool step
```

**Backend pattern (Python / LangGraph):**

```python
from langgraph.config import get_stream_writer

async def my_node(state):
    writer = get_stream_writer()
    writer({"type": "llm_start", "agent": "my_agent"})
    # ... invoke LLM, call tools, etc. ...
    writer({"type": "text", "content": final_answer})
    return {"messages": [...]}
```

**Stream entrypoint pattern:**

```python
async def stream_agent(query: str):
    async for mode, chunk in graph.astream(
        {"messages": [HumanMessage(content=query)]},
        stream_mode=["updates", "custom"],
    ):
        if mode == "custom":
            yield f"data: {json.dumps(chunk)}\n\n"
        elif mode == "updates":
            for node_name, delta in chunk.items():
                for msg in delta.get("messages", []):
                    if isinstance(msg, ToolMessage):
                        yield f"data: {json.dumps({'type':'tool_result','name':msg.name,'content':str(msg.content)[:1000],'agent':node_name})}\n\n"
    yield 'data: {"type": "done"}\n\n'
```

---

## Event Tiers

Events are **additive** — implement only the tiers you need. Higher tiers
enrich the UX but are never required.

| Tier | Events | Use case |
|---|---|---|
| **1 — Core** | `llm_start`, `text`, `done`, `error` | Any single-agent chat |
| **2 — Tools** | `tool_call`, `tool_result` | ReAct / tool-using agents |
| **3 — Orchestrator** | `agent_start`, `agent_end`, `handoff`, `plan`, `plan_step`, `status`, `mcp_server` | Multi-agent systems |
| **4 — Debug** | `node_response`, `graph_state_update`, `debug` field | Development / diagnostics |

---

## Tier 1 — Core Events

These four events produce a complete chat experience: thinking indicator,
streamed answer, and error handling.

### `llm_start`

Signal that an LLM call is in progress. Renders a "Reasoning…" spinner.

```json
{"type": "llm_start"}
{"type": "llm_start", "agent": "my_agent"}
```

| Field | Required | Description |
|---|---|---|
| `type` | ✅ | `"llm_start"` |
| `agent` | | Which agent is reasoning (used for sub-agent nesting) |

---

### `text`

Append content to the answer bubble. Send once or many times for
token-by-token streaming. Content is accumulated and rendered as Markdown.

```json
{"type": "text", "content": "The root cause is "}
{"type": "text", "content": "a misconfigured timeout in the pool."}
```

| Field | Required | Description |
|---|---|---|
| `type` | ✅ | `"text"` |
| `content` | ✅ | Answer text (plain or Markdown; accumulated across multiple events) |

---

### `done`

Signal that the stream is complete. Stops spinners, shows elapsed time.
Always the last event in a successful stream.

```json
{"type": "done"}
```

| Field | Required | Description |
|---|---|---|
| `type` | ✅ | `"done"` |

---

### `error`

Signal a failure. Renders the detail in the answer bubble (red). Should be
the last event — do not emit `done` after `error`.

```json
{"type": "error", "detail": "An internal error occurred. Please try again."}
```

| Field | Required | Description |
|---|---|---|
| `type` | ✅ | `"error"` |
| `detail` | ✅ | Human-readable error message |

---

## Tier 2 — Tool Visibility Events

Emit these from your agent node and tool result emitter so users can see the
ReAct loop in action.

### `tool_call`

Emitted when the LLM decides to call a tool, before the tool executes.
Renders an expandable step with the tool name and a friendly args summary.

```json
{
  "type": "tool_call",
  "name": "search_docs",
  "args": {"query": "connection pool tuning postgres"},
  "id": "call_abc123",
  "agent": "db_agent",
  "server": "postgres_mcp"
}
```

| Field | Required | Description |
|---|---|---|
| `type` | ✅ | `"tool_call"` |
| `name` | ✅ | Tool name (snake_case; auto-prettified for display) |
| `args` | ✅ | Tool arguments as a JSON object |
| `id` | | Tool call ID — used to correlate with the matching `tool_result` |
| `agent` | | Which agent called the tool (for sub-agent nesting) |
| `server` | | Which MCP server owns the tool (rendered as a context badge) |

**Emit from your node:**

```python
for tc in response.tool_calls:
    writer({
        "type": "tool_call",
        "name": tc["name"],
        "args": tc["args"],
        "id": tc["id"],
        "agent": agent_name,
    })
```

---

### `tool_result`

Emitted after a tool has executed and its result is available. Renders a
result step with an auto-generated summary of the content.

```json
{
  "type": "tool_result",
  "name": "search_docs",
  "content": "[{\"title\": \"Pool Config Guide\", ...}]",
  "agent": "db_agent"
}
```

| Field | Required | Description |
|---|---|---|
| `type` | ✅ | `"tool_result"` |
| `name` | ✅ | Tool name (should match the corresponding `tool_call`) |
| `content` | ✅ | Tool output as a string (truncate to ~1000 chars) |
| `agent` | | Which agent received the result |
| `server` | | Which MCP server produced the result |

**Emit from the `updates` stream (ToolMessages):**

```python
elif mode == "updates":
    for node_name, delta in chunk.items():
        for msg in delta.get("messages", []):
            if isinstance(msg, ToolMessage):
                writer({
                    "type": "tool_result",
                    "name": msg.name,
                    "content": str(msg.content)[:1000],
                    "agent": node_name,
                })
```

---

## Tier 3 — Orchestrator Events

Use these in multi-agent systems to show agent lifecycle, control flow, and
execution plans in the UI.

### `agent_start`

Emitted when a sub-agent begins work. Creates a collapsible nested group in
the UI — all subsequent events with `"agent": "<same_name>"` render inside it.

```json
{"type": "agent_start", "agent": "db_agent", "role": "Database specialist"}
```

| Field | Required | Description |
|---|---|---|
| `type` | ✅ | `"agent_start"` |
| `agent` | ✅ | Agent identifier (must match the `agent` field on later events) |
| `role` | | Human-readable role description shown in the UI header |

---

### `agent_end`

Emitted when a sub-agent completes. Marks the nested group as done (✓ icon).

```json
{"type": "agent_end", "agent": "db_agent", "summary": "Found 42 matching rows"}
```

| Field | Required | Description |
|---|---|---|
| `type` | ✅ | `"agent_end"` |
| `agent` | ✅ | Agent identifier (must match the corresponding `agent_start`) |
| `summary` | | Short completion summary shown below the group |

---

### `handoff`

Emitted when the orchestrator delegates to a sub-agent. Renders a transition
arrow in the thinking steps.

```json
{
  "type": "handoff",
  "from": "orchestrator",
  "to": "db_agent",
  "reason": "SQL query needed to answer this question"
}
```

| Field | Required | Description |
|---|---|---|
| `type` | ✅ | `"handoff"` |
| `from` | ✅ | Source agent name |
| `to` | ✅ | Destination agent name |
| `reason` | | Brief explanation of why control is being transferred |

---

### `plan`

Emitted by the orchestrator after it decides what to do. Renders a visual
plan tracker with step-by-step progress icons.

```json
{
  "type": "plan",
  "steps": [
    {"id": "classify", "label": "Classify intent"},
    {"id": "execute",  "label": "Run specialist agent"},
    {"id": "evaluate", "label": "Evaluate completeness"},
    {"id": "respond",  "label": "Generate response"}
  ],
  "agent": "orchestrator"
}
```

| Field | Required | Description |
|---|---|---|
| `type` | ✅ | `"plan"` |
| `steps` | ✅ | Array of `{id: string, label: string}` — order matters for display |
| `agent` | | Which agent owns this plan (for multi-plan scenarios) |

---

### `plan_step`

Update the status of a single plan step. Call with `running` when the step
starts and `done`/`error` when it finishes.

```json
{"type": "plan_step", "step_id": "execute", "status": "running", "agent": "orchestrator"}
{"type": "plan_step", "step_id": "execute", "status": "done",    "agent": "orchestrator"}
```

| Field | Required | Description |
|---|---|---|
| `type` | ✅ | `"plan_step"` |
| `step_id` | ✅ | Must match an `id` from a prior `plan` event |
| `status` | ✅ | `"running"` \| `"done"` \| `"error"` \| `"skipped"` |
| `agent` | | Must match the `agent` on the parent `plan` event |

---

### `status`

General-purpose progress message. Renders as an informational step in the
thinking panel. Use for "Analyzing your request…", "Evaluating completeness…",
etc.

```json
{"type": "status", "message": "Evaluating research completeness…", "agent": "orchestrator"}
{"type": "status", "message": "Downloading page…", "progress": 42, "agent": "web_agent"}
```

| Field | Required | Description |
|---|---|---|
| `type` | ✅ | `"status"` |
| `message` | ✅ | Human-readable status text |
| `agent` | | Which agent emitted it |
| `progress` | | Optional 0–100 percentage for progress bar rendering |

---

### `mcp_server`

Emitted when a sub-agent connects to (or loses) its MCP server. Renders a
connectivity badge (🔌 or ⚠️).

```json
{"type": "mcp_server", "server": "postgres_mcp", "status": "connected"}
{"type": "mcp_server", "server": "postgres_mcp", "status": "disconnected", "error": "Connection refused"}
```

| Field | Required | Description |
|---|---|---|
| `type` | ✅ | `"mcp_server"` |
| `server` | ✅ | MCP server name (matches the key in `MCP_CONNECTIONS`) |
| `status` | ✅ | `"connected"` \| `"disconnected"` |
| `error` | | Error message when `status` is `"disconnected"` |

---

## Tier 4 — Debug Events

These events are only useful during development. They should be suppressed in
production unless the user explicitly enables a debug mode.

### `node_response`

Emitted after a graph node produces its LLM output. Carries a `debug` object
with raw LLM content, tool calls, and other diagnostics.

```json
{
  "type": "node_response",
  "node": "classify_intent",
  "agent": "orchestrator",
  "debug": {
    "raw_llm_output": "document",
    "resolved_intent": "document",
    "plan_steps": [...]
  }
}
```

| Field | Required | Description |
|---|---|---|
| `type` | ✅ | `"node_response"` |
| `node` | ✅ | Graph node name |
| `agent` | | Which agent this node belongs to |
| `debug` | | Arbitrary diagnostic object (only rendered when debug mode is on) |

---

### `graph_state_update`

Emitted after each graph node completes with a snapshot of the state delta.
Useful for tracing state mutations across the graph.

```json
{
  "type": "graph_state_update",
  "node": "evaluate",
  "debug": {
    "state_delta": {
      "resolution_status": "needs_web_agent",
      "resolution_round": 1,
      "followup_query": "Find recent CVEs for nginx 1.25"
    }
  }
}
```

| Field | Required | Description |
|---|---|---|
| `type` | ✅ | `"graph_state_update"` |
| `node` | ✅ | Graph node that produced the state delta |
| `debug` | | State delta object |

---

### `debug` field on any event

Any event from any tier can carry an extra `debug` field. When a consumer is
in debug mode, it renders this as a syntax-highlighted expandable JSON block.
When debug mode is off, the field is stripped before display.

```json
{
  "type": "tool_call",
  "name": "search_docs",
  "args": {"query": "memory leak"},
  "debug": {
    "full_args": {"query": "memory leak", "limit": 10, "filters": {"date": ">2024"}},
    "model": "gemini-2.5-flash",
    "tool_call_id": "call_abc123",
    "latency_ms": 342
  }
}
```

---

## Human-in-the-Loop Events

These events are handled by the reference UI but not yet emitted by the
reference backend. Implement them when you add LangGraph `interrupt()` flows.

### `approval_request`

Pause and prompt the user before executing a dangerous action.

```json
{
  "type": "approval_request",
  "action": "delete_resource",
  "detail": "{\"namespace\": \"prod\", \"resource\": \"deployment/api-server\"}",
  "agent": "k8s_agent"
}
```

| Field | Required | Description |
|---|---|---|
| `type` | ✅ | `"approval_request"` |
| `action` | ✅ | Short name of the action requiring approval |
| `detail` | | Full action parameters shown to the user |
| `agent` | | Agent requesting the approval |

---

### `approval_result`

Report the outcome of a human approval decision.

```json
{"type": "approval_result", "approved": true}
{"type": "approval_result", "approved": false}
```

| Field | Required | Description |
|---|---|---|
| `type` | ✅ | `"approval_result"` |
| `approved` | ✅ | `true` if approved, `false` if rejected |

---

## Context Fields (All Events)

These optional fields can be added to any event to enrich display:

| Field | Type | Effect |
|---|---|---|
| `agent` | string | Routes the event into the named sub-agent's nested UI container |
| `server` | string | Adds an MCP server badge to the step label |
| `ts` | ISO-8601 string | Timestamp shown in the activity panel |

When `agent` is omitted or `"root"`, the event renders at the top-level
thinking steps. When `agent` is set, the event nests inside the group created
by the matching `agent_start` event.

---

## Complete Event Sequence Examples

### Simple single-agent (Tier 1 only)

```
{"type": "llm_start"}
{"type": "text", "content": "The answer to your question is "}
{"type": "text", "content": "a misconfigured timeout."}
{"type": "done"}
```

### ReAct agent with tools (Tiers 1–2)

```
{"type": "llm_start"}
{"type": "tool_call",   "name": "search_docs", "args": {"query": "timeout config"}, "id": "c1"}
{"type": "tool_result", "name": "search_docs", "content": "See: /docs/timeouts.md", "id": "c1"}
{"type": "llm_start"}
{"type": "text",        "content": "According to the docs, set `pool_timeout=30`."}
{"type": "done"}
```

### Multi-agent orchestrator (Tiers 1–3)

```
{"type": "agent_start",  "agent": "orchestrator", "role": "coordinator"}
{"type": "status",       "message": "Analyzing your request…", "agent": "orchestrator"}
{"type": "plan",         "steps": [{"id":"classify","label":"Classify"},{"id":"execute","label":"Run agent"},{"id":"respond","label":"Respond"}], "agent": "orchestrator"}
{"type": "plan_step",    "step_id": "classify", "status": "done", "agent": "orchestrator"}
{"type": "handoff",      "from": "orchestrator", "to": "db_agent", "reason": "SQL query needed"}

{"type": "agent_start",  "agent": "db_agent", "role": "Database specialist"}
{"type": "mcp_server",   "server": "postgres_mcp", "status": "connected"}
{"type": "llm_start",    "agent": "db_agent"}
{"type": "tool_call",    "name": "run_query", "args": {"sql": "SELECT ..."}, "id": "c1", "agent": "db_agent"}
{"type": "tool_result",  "name": "run_query", "content": "42 rows returned", "agent": "db_agent"}
{"type": "agent_end",    "agent": "db_agent", "summary": "Query returned 42 rows"}

{"type": "plan_step",    "step_id": "execute", "status": "done", "agent": "orchestrator"}
{"type": "plan_step",    "step_id": "respond", "status": "running", "agent": "orchestrator"}
{"type": "text",         "content": "Based on the query results, the slow queries are caused by…"}
{"type": "agent_end",    "agent": "orchestrator", "summary": "Response complete"}
{"type": "plan_step",    "step_id": "respond", "status": "done", "agent": "orchestrator"}
{"type": "done"}
```

---

## Adoption Checklist

| Step | Action | Tier |
|---|---|---|
| 1 | Emit `llm_start` at the start of each LLM call | 1 |
| 2 | Emit `text` with the final answer content | 1 |
| 3 | Emit `done` after the last `text` | 1 |
| 4 | Emit `error` (instead of `done`) on failure | 1 |
| 5 | Emit `tool_call` for each tool call decision | 2 |
| 6 | Emit `tool_result` from `ToolMessage` outputs | 2 |
| 7 | Emit `agent_start` / `agent_end` per sub-agent | 3 |
| 8 | Emit `handoff` when delegating between agents | 3 |
| 9 | Emit `plan` + `plan_step` updates from orchestrator | 3 |
| 10 | Emit `status` for long-running intermediate steps | 3 |
| 11 | Emit `mcp_server` on tool server connect/disconnect | 3 |
| 12 | Add `debug` field to events for dev diagnostics | 4 |

All events accept unknown extra fields — they are ignored by strict consumers
and preserved in `debug` blocks by debug-mode consumers.

---

## Reference Implementation

This event protocol is fully implemented in:

- **Backend emitter:** `app/orchestrator_agent.py` — all `writer({...})` calls
- **Backend forwarder:** `app/orchestrator_agent.py` → `stream_agent()` — SSE stream entrypoint
- **Frontend handler:** `static/app.js` → `handleEvent()` — complete event switch

The reference UI (`static/`) is a drop-in SSE consumer that handles all
events above, gracefully renders unknown future event types, and supports
debug mode toggling without any backend changes.
