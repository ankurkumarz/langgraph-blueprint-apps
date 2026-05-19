# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run the server (dev mode with auto-reload)
python main.py

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_agent.py

# Run a single test by name
uv run pytest tests/test_agent.py::TestHandleMcpToolErrors::test_connect_error_returns_connectivity_message

# Run tests with verbose output
uv run pytest -v

# The server and tests both require these env vars (or a .env file):
# FIRECRAWL_API_KEY=...
# GOOGLE_API_KEY=...
```

The chat UI is served at `http://localhost:8000/agent/copilot`.

## Architecture

This is a **registry-driven multi-agent LangGraph orchestrator** that streams responses to a browser UI via SSE.

### Module layout

| File | Responsibility |
|---|---|
| `app/settings.py` | `Settings` (pydantic-settings singleton) — all config/secrets, MCP URLs, computed properties. Validated at import time; app crashes on missing required vars. |
| `app/subagents.py` | `AGENT_CONFIGS` registry, shared utilities (`handle_mcp_tool_errors`, `_partition_tools`, `_build_subagent_graph`, `sse`, `_debug_payload`), sub-agent state schema, per-request delegation log via `ContextVar`. |
| `app/orchestrator_agent.py` | Both orchestrator graphs (`build_evaluator_orchestrator`, `build_think_orchestrator`), the singleton graph cache (`_get_or_build_graph`), in-flight request tracking, and the `stream_agent()` async generator. |
| `app/agent.py` | Backward-compat shim — re-exports everything from `subagents` and `orchestrator_agent`. Code should import from the two source modules directly. |
| `app/app.py` | FastAPI routes, CORS/security middleware, lifespan (eager graph warm-up, SIGHUP hot-reload, graceful drain). The `/agent/conversation` endpoint returns `StreamingResponse` from `stream_agent()`. |
| `app/tracing.py` | MLflow autolog — called before any LangChain imports in `app.py`; conditional on `MLFLOW_ENABLED=true`. |
| `static/` | Drop-in chat UI. `app.js:handleEvent()` is the SSE event switch — all SSE types map to a `case` there. |

### Agent registry pattern

All sub-agents are declared in `AGENT_CONFIGS` in `subagents.py`. Each entry is a dict with `name`, `role`, `description`, `system_prompt`, `mcp_server`, `tool_prefix`, and `evaluator_intent`. **Adding one entry is the only change needed to add a new agent** — both orchestrators auto-discover it.

`_partition_tools()` assigns MCP tools to agents by `tool_prefix` matching. Unmatched tools fall to the last agent in the list.

Every sub-agent is an identical ReAct graph built by `_build_subagent_graph()`: `agent_node → ToolNode → tool_result_emitter → agent_node (loop)`.

### Two orchestrator modes (`ORCHESTRATOR_MODE` env var)

**`evaluator` (default):** Multi-node structured graph — `classify_intent → agent(s) → evaluate → synthesize`. A separate evaluator LLM grades completeness after each agent run and can trigger follow-up loops. Nodes and follow-up paths are generated dynamically from `AGENT_CONFIGS`.

**`think`:** Single ReAct loop — the orchestrator LLM has two tools: `think_tool` (forces structured reflection using a per-request `ContextVar` delegation log) and `delegate` (runs a sub-agent; multiple calls in one LLM response run via `ToolNode` concurrently). Simpler but less deterministic.

### SSE streaming

Nodes emit events via `get_stream_writer()` (LangGraph's custom stream mode). `stream_agent()` iterates `graph.astream(stream_mode=["updates", "custom"])`: `"custom"` chunks are forwarded directly as SSE; `"updates"` chunks are inspected for `ToolMessage`s to emit `tool_result` events and (in debug mode) `graph_state_update` events.

The SSE event protocol is documented in `SSE-STREAMING-EVENTS.md` (tiered: core → tools → orchestrator → debug).

### Singleton graph cache

The compiled graph is built once on the first request (or eagerly at startup via `warm_up()`), then reused. MCP tool discovery and graph compilation are both expensive. The cache uses a double-checked lock (`asyncio.Lock`). `invalidate_graph_cache()` resets it for hot-reloads (also triggered by `SIGHUP` and `POST /api/admin/invalidate-cache`).

Per-request state is passed through `astream()` inputs — the cached graph itself is stateless.

### Testing

`conftest.py` **must** set required env vars before any app import because `settings.py` instantiates `Settings()` at import time. The `client` fixture mocks `warm_up` to avoid MCP connections during tests.

Tests in `test_agent.py` cover pure utility functions only. Integration tests that require a live graph or MCP connection are out of scope for the unit suite.
