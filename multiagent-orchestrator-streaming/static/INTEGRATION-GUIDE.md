# LangGraph Agent Chat UI — Integration Guide

A portable, drop-in chat interface for any LangGraph-based agent or multi-agent
orchestrator. The UI renders thinking steps, tool usage, sub-agent lifecycles,
execution plans, and streamed answers — all driven by a simple SSE event
protocol.

![Chat UI](chatbot.png)

---

## Table of Contents

- [Quick Start](#quick-start)
- [Files](#files)
- [Backend Contract](#backend-contract)
  - [Endpoint](#endpoint)
  - [SSE Format](#sse-format)
- [Event Reference](#event-reference)
  - [Tier 1 — Core (Minimum Viable Chat)](#tier-1--core-minimum-viable-chat)
  - [Tier 2 — Tool Visibility (ReAct Loop)](#tier-2--tool-visibility-react-loop)
  - [Tier 3 — Orchestrator (Multi-Agent)](#tier-3--orchestrator-multi-agent)
  - [Tier 4 — Debug (Optional Enrichment)](#tier-4--debug-optional-enrichment)
- [Full Event Lifecycle](#full-event-lifecycle)
- [Customising for Your Agent](#customising-for-your-agent)
  - [Branding](#branding)
  - [Sample Questions](#sample-questions)
  - [Chat History](#chat-history)
  - [Dashboard Tabs](#dashboard-tabs)
- [Feature Reference](#feature-reference)
  - [Thinking Steps](#thinking-steps)
  - [Activity Panel](#activity-panel)
  - [Debug Mode](#debug-mode)
  - [Markdown Rendering](#markdown-rendering)
  - [Friendly Summaries](#friendly-summaries)
  - [Unknown Events](#unknown-events)
- [Integration Checklist](#integration-checklist)
- [Python (FastAPI) Example](#python-fastapi-example)
- [Architecture](#architecture)

---

## Quick Start

1. Copy the `static/` folder into your project.
2. Serve the files as static assets from your web framework.
3. Implement `POST /agent/conversation` returning an SSE stream.
4. Emit at minimum: `llm_start`, `text`, `done` events.

That's it — you have a working chat UI.

---

## Files

| File          | Purpose                                                    |
| ------------- | ---------------------------------------------------------- |
| `index.html`  | Page structure — sidebar, chat area, input bar, tabs       |
| `styles.css`  | All styling — thinking steps, activity panel, dashboards   |
| `app.js`      | Application logic — SSE handling, rendering, state         |
| `chatbot.png` | Default bot avatar (replace with your own)                 |

---

## Backend Contract

### Endpoint

The frontend makes **one API call**:

```
POST /agent/conversation
Content-Type: application/json

{"query": "How do I fix this error?"}
```

**Response:** `text/event-stream` (SSE) with recommended headers:

```
Content-Type: text/event-stream
Cache-Control: no-cache
X-Accel-Buffering: no
```

> **No other endpoints are required.** Debug mode is controlled entirely
> client-side via a toggle button — no `/api/debug` call needed.

### SSE Format

Each event is a single line prefixed with `data:` followed by JSON, terminated
by two newlines:

```
data: {"type": "llm_start"}\n\n
data: {"type": "text", "content": "The answer is..."}\n\n
data: {"type": "done"}\n\n
```

---

## Event Reference

Events are **additive** — implement only the tiers you need. The UI gracefully
ignores events it doesn't receive and renders unknown event types as
informational steps.

### Tier 1 — Core (Minimum Viable Chat)

These 4 events give you a full chat with a thinking indicator and streamed
answer.

| Event       | Fields                    | UI Effect                                |
| ----------- | ------------------------- | ---------------------------------------- |
| `llm_start` | —                         | ◆ "Reasoning…" spinner in thinking steps |
| `text`      | `content: string`         | Streams answer text (send multiple)      |
| `done`      | —                         | Stops spinner, shows ⏱ elapsed time      |
| `error`     | `detail: string`          | Red error message in chat bubble         |

```jsonc
// 1. LLM starts thinking
{"type": "llm_start"}

// 2. Stream the answer (emit once or many times for token-by-token streaming)
{"type": "text", "content": "The issue is caused by "}
{"type": "text", "content": "a memory leak in the connection pool."}

// 3. Stream complete
{"type": "done"}

// 4. On failure (instead of done)
{"type": "error", "detail": "An internal error occurred. Please try again."}
```

### Tier 2 — Tool Visibility (ReAct Loop)

Show the agent calling tools with auto-generated friendly summaries.

| Event         | Fields                                            | UI Effect                                  |
| ------------- | ------------------------------------------------- | ------------------------------------------ |
| `tool_call`   | `name`, `args`, `id?`, `agent?`, `server?`        | ⚙ Tool step with args preview              |
| `tool_result` | `name`, `content`, `id?`, `agent?`, `server?`     | ✓ Result step with auto-summarised content |

```jsonc
// Agent calls a tool
{
  "type": "tool_call",
  "name": "search_docs",
  "args": {"query": "connection pool tuning"},
  "id": "call_abc123"
}

// Tool returns result
{
  "type": "tool_result",
  "name": "search_docs",
  "content": "[{\"title\": \"Pool Config Guide\"}, {\"title\": \"Timeout Settings\"}]",
  "id": "call_abc123"
}
```

The UI auto-generates friendly labels from tool args and results:

| Tool args pattern                             | Rendered as                              |
| --------------------------------------------- | ---------------------------------------- |
| `{"query": "..."}` or `{"q": "..."}`          | `Searching for: "..."`                   |
| `{"url": "https://example.com/path"}`         | `Fetching example.com`                   |
| `{"path": "/etc/config"}`                     | `Reading: /etc/config`                   |
| `{"command": "kubectl get pods"}`             | `Running: kubectl get pods`              |
| `{"agent_name": "k8s", "query": "..."}`       | `Asking Kubernetes: "..."`               |
| JSON array result with `.title` fields        | `Found 3 results: Title1, Title2, ...`   |
| JSON with `.markdown` field                   | `Scraped page: <heading> (N words)`      |
| Plain text result                             | Truncated to 150 chars                   |

### Tier 3 — Orchestrator (Multi-Agent)

Full multi-agent visualisation with nested sub-agent groups, plans, and
handoffs. All events accept an optional `agent` field to attribute them to a
specific sub-agent.

| Event         | Fields                                           | UI Effect                                    |
| ------------- | ------------------------------------------------ | -------------------------------------------- |
| `agent_start` | `agent`, `role?`                                 | 🤖 Nested sub-agent group with spinner       |
| `agent_end`   | `agent`, `summary?`                              | ✓ Sub-agent group marked complete             |
| `plan`        | `steps: [{id, label}]`, `agent?`                 | 📋 Visual plan with numbered steps           |
| `plan_step`   | `step_id`, `status: running\|done\|error\|skipped` | Updates step icon in the plan               |
| `handoff`     | `from`, `to`, `reason?`                          | 🔀 Transition arrow between agents           |
| `status`      | `message`, `agent?`, `progress?`                 | ℹ️ Informational progress message            |
| `mcp_server`  | `server`, `status: connected\|disconnected`, `error?` | 🔌 MCP server connectivity badge        |

```jsonc
// Orchestrator starts
{"type": "agent_start", "agent": "orchestrator", "role": "coordinator"}
{"type": "status", "message": "Analyzing your request...", "agent": "orchestrator"}

// Show execution plan
{"type": "plan", "steps": [
  {"id": "classify", "label": "Classify intent"},
  {"id": "execute",  "label": "Run specialist agent"},
  {"id": "evaluate", "label": "Evaluate completeness"},
  {"id": "respond",  "label": "Generate response"}
], "agent": "orchestrator"}

// Update plan steps as work progresses
{"type": "plan_step", "step_id": "classify", "status": "done"}
{"type": "plan_step", "step_id": "execute",  "status": "running"}

// Hand off to a sub-agent
{"type": "handoff", "from": "orchestrator", "to": "db-agent", "reason": "Database query needed"}

// Sub-agent lifecycle (events nest visually under the sub-agent)
{"type": "agent_start",  "agent": "db-agent", "role": "Database specialist"}
{"type": "mcp_server",   "server": "postgres-mcp", "status": "connected"}
{"type": "llm_start",    "agent": "db-agent"}
{"type": "tool_call",    "name": "run_query", "args": {"sql": "SELECT ..."}, "agent": "db-agent"}
{"type": "tool_result",  "name": "run_query", "content": "42 rows returned", "agent": "db-agent"}
{"type": "agent_end",    "agent": "db-agent", "summary": "Query returned 42 rows"}

// Continue with plan
{"type": "plan_step", "step_id": "execute", "status": "done"}
{"type": "plan_step", "step_id": "respond", "status": "running"}
{"type": "text", "content": "Based on the database analysis..."}
{"type": "agent_end", "agent": "orchestrator", "summary": "Response generated"}
{"type": "plan_step", "step_id": "respond", "status": "done"}
{"type": "done"}
```

#### The `agent` Field

Any Tier 1 or Tier 2 event can include `"agent": "sub-agent-name"` to nest it
inside a sub-agent group. Without this field, events render at the root level.

```jsonc
// Root-level reasoning
{"type": "llm_start"}

// Sub-agent reasoning (nests inside "db-agent" group)
{"type": "llm_start", "agent": "db-agent"}
```

#### The `server` Field

Any `tool_call` or `tool_result` can include `"server": "mcp-server-name"` to
show which MCP server the tool belongs to. This renders as a badge in both the
thinking steps and the activity panel.

### Tier 4 — Debug (Optional Enrichment)

Any event can carry an extra `"debug": {...}` field. When the user toggles
debug mode ON (via the 🐛 button in the header), these render as
syntax-highlighted JSON blocks below the corresponding step.

```jsonc
{
  "type": "tool_call",
  "name": "search_docs",
  "args": {"query": "memory leak"},
  "debug": {
    "full_args": {"query": "memory leak", "limit": 10, "filters": {"date": ">2024"}},
    "model": "gemini-2.0-flash",
    "tool_call_id": "call_abc123",
    "latency_ms": 342
  }
}
```

Two additional debug-only event types are also supported:

| Event                | Fields                 | UI Effect (debug ON only)             |
| -------------------- | ---------------------- | ------------------------------------- |
| `node_response`      | `node`, `agent?`       | 🐛 Shows raw LLM node output         |
| `graph_state_update` | `node`, `agent?`       | 🐛 Shows graph state delta            |

These are invisible when debug mode is off.

---

## Full Event Lifecycle

A complete orchestrator query produces this sequence:

```
User submits query
  → UI instantly shows "Thinking..." (client-side, no server event needed)

Backend SSE stream:
  ┌─ Orchestrator ──────────────────────────────────────────────────┐
  │  1.  agent_start    (orchestrator)     → 🤖 group opens        │
  │  2.  status         ("Analyzing...")   → ℹ️ progress text      │
  │  3.  plan           (4 steps)          → 📋 plan appears       │
  │  4.  plan_step      (classify → done)  → ✓ step done           │
  │  5.  handoff        (→ db-agent)       → 🔀 transition         │
  │                                                                 │
  │  ┌─ Sub-agent ─────────────────────────────────────────────┐   │
  │  │  6.  agent_start  (db-agent)        → 🤖 nested group   │   │
  │  │  7.  mcp_server   (connected)       → 🔌 badge          │   │
  │  │  8.  llm_start    (db-agent)        → ◆ "Reasoning…"    │   │
  │  │  9.  tool_call    (run_query)       → ⚙ tool step       │   │
  │  │  10. tool_result  (42 rows)         → ✓ result step     │   │
  │  │  11. agent_end    (db-agent)        → ✓ group closes     │   │
  │  └─────────────────────────────────────────────────────────┘   │
  │                                                                 │
  │  12. plan_step      (execute → done)   → ✓ step done           │
  │  13. plan_step      (respond → running)→ ◆ step active         │
  │  14. text           ("Based on...")    → answer streams         │
  │  15. text           ("...analysis")    → answer continues       │
  │  16. agent_end      (orchestrator)     → orchestrator done      │
  │  17. plan_step      (respond → done)   → ✓ final step done     │
  └─────────────────────────────────────────────────────────────────┘
  18. done                                  → ⏱ elapsed time shown
```

For a **simple single-agent** (no orchestration), only 4 events are needed:

```
  1. llm_start                             → ◆ "Reasoning…"
  2. text          ("The answer is...")     → answer streams
  3. text          ("...more text")        → answer continues
  4. done                                  → ⏱ elapsed time shown
```

---

## Customising for Your Agent

### Branding

Update these in `index.html`:

```html
<!-- Line 6: page title -->
<title>Your Agent Name</title>

<!-- Line 54-55: header -->
<h1>Your Agent Name</h1>
<div class="subtitle">Your description here</div>
```

And replace `chatbot.png` with your own icon.

### Sample Questions

Update the welcome screen cards in `index.html` (inside `<div id="placeholder">`):

```html
<div class="sample-card" data-query="Your sample question here?">
  <div class="card-icon">🔍</div>
  <div class="card-text">Short label for the card</div>
</div>
```

Also update the matching `PLACEHOLDER_HTML` constant in `app.js` (used when
the user clicks "New Chat"):

```js
const PLACEHOLDER_HTML =
  `<div class="placeholder" id="placeholder">` +
    `<span class="icon">💬</span>` +
    `<p>Your welcome message here.</p>` +
    `<div class="sample-cards">` +
      `<div class="sample-card" data-query="Your question?">...</div>` +
    `</div>` +
  `</div>`;
```

### Chat History

Replace the `DUMMY_CHAT_HISTORY` array in `app.js` with your own data or an
API call:

```js
const DUMMY_CHAT_HISTORY = [
  {
    id: 'chat-001',
    title: 'Conversation title',
    preview: 'First message preview text',
    timestamp: new Date().toISOString(),
    messageCount: 4,
  },
];
```

### Dashboard Tabs

The UI includes two additional tabs (AI Insights and AI Monitoring) with
domain-specific dashboard content. If you don't need them:

1. **Remove the tab buttons** from `index.html` (the `<button class="top-tab">`
   elements for `insights` and `monitoring`).
2. **Remove the tab content divs** (`#tab-insights` and `#tab-monitoring`).
3. **Remove the dashboard code** from `app.js` — everything from the
   `INCIDENTS` constant through `renderScalingTable()` (~700 lines), plus
   `initInsightsDashboard()` and `initMonitoringDashboard()`.

Or keep the tab infrastructure and replace the content with your own domain
dashboards.

### API Endpoint

If your backend uses a different URL, update this line in the `send()` function
in `app.js`:

```js
const res = await fetch('/agent/conversation', {   // ← change this URL
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ query }),
});
```

---

## Feature Reference

### Thinking Steps

Every event renders as a collapsible step in the thinking panel above the
answer. Steps show icons, labels, expandable details, and connection lines.
Active steps pulse with a spinner animation.

### Activity Panel

A slide-out side panel (toggled via the 📋 button) logs every event with
timestamps. Hovering over a row shows a popover with full event details
including any debug payload.

### Debug Mode

A client-side toggle (🐛 button in the header) that persists in
`localStorage`. When ON:

- Events with `"debug": {...}` fields render as syntax-highlighted JSON blocks.
- `node_response` and `graph_state_update` events become visible.
- A debug summary banner shows event/tool counts after completion.

No server endpoint is required — the toggle is always available.

### Markdown Rendering

The `text` event content is accumulated and rendered as Markdown with support
for:

- Headings (`#` through `####`)
- Bold, italic, bold-italic
- Fenced code blocks with language hints
- Inline code
- Ordered and unordered lists
- Blockquotes
- Horizontal rules
- Links (both `[text](url)` and raw URLs)

### Friendly Summaries

The UI auto-generates human-readable labels from raw tool args and results.
Common patterns are detected:

- `query`/`q` args → `Searching for: "..."`
- `url` args → `Fetching example.com`
- `path`/`file` args → `Reading: /path/to/file`
- `command`/`cmd` args → `Running: kubectl get pods`
- `agent_name` args → `Delegating to Agent Name`
- JSON array results → `Found N results: Title1, Title2, ...`
- JSON with `.markdown` → `Scraped page: <heading> (N words)`

No special formatting needed from your backend — raw tool names, args, and
content are automatically prettified.

### Unknown Events

Any event type not listed above is gracefully rendered as an informational step
(ℹ️ icon with the type name as the label). This means you can emit custom event
types from your backend and they will never break the UI — they'll appear as
labelled steps in the thinking panel.

---

## Integration Checklist

| #   | Action                                                       | Required? |
| --- | ------------------------------------------------------------ | --------- |
| 1   | Implement `POST /agent/conversation` returning SSE           | ✅ Yes     |
| 2   | Emit `llm_start`, `text`, `done`, `error`                    | ✅ Yes     |
| 3   | Emit `tool_call`, `tool_result`                              | Recommended |
| 4   | Emit `agent_start/end`, `plan`, `plan_step`, `handoff`       | Optional  |
| 5   | Add `"agent"` field to sub-agent events                      | Optional  |
| 6   | Add `"server"` field to MCP tool events                      | Optional  |
| 7   | Add `"debug": {...}` to events for debug panel               | Optional  |
| 8   | Replace title, heading, subtitle in `index.html`             | ✅ Yes     |
| 9   | Replace sample question cards                                | ✅ Yes     |
| 10  | Replace or clear `DUMMY_CHAT_HISTORY` in `app.js`            | Recommended |
| 11  | Replace `chatbot.png`                                        | Optional  |
| 12  | Remove/replace Insights & Monitoring tabs (if not applicable)| Optional  |

---

## Python (FastAPI) Example

A minimal backend that works with this UI:

```python
import json
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    query: str

def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"

async def stream_agent(query: str):
    """Minimal SSE stream — replace with your LangGraph agent."""

    # Tier 1: Core events
    yield sse({"type": "llm_start"})

    # Simulate LLM thinking
    await asyncio.sleep(1)

    # Stream the answer
    answer = f"You asked: {query}. Here is my analysis..."
    for word in answer.split():
        yield sse({"type": "text", "content": word + " "})
        await asyncio.sleep(0.05)

    yield sse({"type": "done"})

@app.post("/agent/conversation")
async def conversation(request: ChatRequest):
    return StreamingResponse(
        stream_agent(request.query),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# Serve the chat UI
app.mount("/", StaticFiles(directory="static", html=True), name="static")
```

### With LangGraph streaming:

```python
from langgraph.config import get_stream_writer

async def my_agent_node(state):
    writer = get_stream_writer()

    # Emit events that the UI understands
    writer({"type": "llm_start", "agent": "my-agent"})

    response = await llm.ainvoke(state["messages"])

    for tool_call in response.tool_calls:
        writer({
            "type": "tool_call",
            "name": tool_call["name"],
            "args": tool_call["args"],
            "id": tool_call["id"],
            "agent": "my-agent",
        })

    return {"messages": [response]}

# In your stream endpoint, forward custom events:
async def stream_agent(query: str):
    graph = build_graph()
    async for mode, chunk in graph.astream(
        {"messages": [HumanMessage(content=query)]},
        stream_mode=["updates", "custom"],
    ):
        if mode == "custom":
            yield sse(chunk)       # Forward agent events to the UI
        elif mode == "updates":
            # Emit tool_result events from ToolMessages
            for node_name, state_delta in chunk.items():
                for msg in state_delta.get("messages", []):
                    if isinstance(msg, ToolMessage):
                        yield sse({
                            "type": "tool_result",
                            "name": msg.name,
                            "content": str(msg.content)[:1000],
                            "agent": node_name,
                        })
    yield sse({"type": "done"})
```

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Browser (static/)                                   │
│                                                      │
│  index.html ── structure + tabs                      │
│  styles.css ── all visual styling                    │
│  app.js ────── SSE handling + rendering              │
│                                                      │
│  POST /agent/conversation {"query": "..."}           │
│       │                                              │
│       ▼  SSE stream                                  │
│  ┌─────────────────────────────┐                     │
│  │ data: {"type":"llm_start"}  │──→ Thinking step    │
│  │ data: {"type":"tool_call"}  │──→ Tool step        │
│  │ data: {"type":"tool_result"}│──→ Result step      │
│  │ data: {"type":"text"}       │──→ Answer bubble    │
│  │ data: {"type":"done"}       │──→ Complete         │
│  └─────────────────────────────┘                     │
└──────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────┐
│  Your Backend (any framework)                        │
│                                                      │
│  FastAPI / Flask / Express / etc.                     │
│       │                                              │
│       ▼                                              │
│  LangGraph Agent / Orchestrator                      │
│  - get_stream_writer() → emit SSE events             │
│  - graph.astream(stream_mode=["updates", "custom"])  │
└──────────────────────────────────────────────────────┘
```

---

## License

This chat UI is part of the
[langgraph-blueprint-apps](https://github.com/your-org/langgraph-blueprint-apps)
project. See the root repository for license details.
