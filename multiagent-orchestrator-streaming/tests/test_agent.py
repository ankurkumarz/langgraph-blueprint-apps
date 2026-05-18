"""
Unit tests for app/agent.py.

Tests pure/utility functions that do not require live MCP connections or a
running LangGraph graph.  Functions that depend on the compiled graph
(stream_agent, _get_or_build_graph) are covered by integration tests only.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


# ---------------------------------------------------------------------------
# handle_mcp_tool_errors
# ---------------------------------------------------------------------------

class TestHandleMcpToolErrors:
    def _call(self, exc):
        from app.agent import handle_mcp_tool_errors
        return handle_mcp_tool_errors(exc)

    def test_connect_error_returns_connectivity_message(self):
        err = httpx.ConnectError("Connection refused")
        result = self._call(err)
        assert "MCP server connection failed" in result
        assert "unavailable" in result.lower() or "unreachable" in result.lower()

    def test_read_timeout_returns_timeout_message(self):
        err = httpx.ReadTimeout("Timed out")
        result = self._call(err)
        assert "timed out" in result.lower()
        assert "ReadTimeout" in result

    def test_http_status_error_includes_status_code(self):
        request = httpx.Request("POST", "https://mcp.example.com/v2/mcp")
        response = httpx.Response(503, request=request)
        err = httpx.HTTPStatusError("Service unavailable", request=request, response=response)
        result = self._call(err)
        assert "503" in result
        assert "MCP server returned HTTP" in result

    def test_connection_error_returns_network_message(self):
        err = ConnectionError("Socket closed")
        result = self._call(err)
        assert "Network connectivity error" in result

    def test_os_error_returns_network_message(self):
        err = OSError("Broken pipe")
        result = self._call(err)
        assert "Network connectivity error" in result

    def test_generic_exception_returns_repr(self):
        err = ValueError("something went wrong")
        result = self._call(err)
        assert "ValueError" in result
        assert "something went wrong" in result


# ---------------------------------------------------------------------------
# _extract_final_answer
# ---------------------------------------------------------------------------

class TestExtractFinalAnswer:
    def _call(self, messages):
        from app.agent import _extract_final_answer
        return _extract_final_answer(messages)

    def test_returns_last_ai_text_message(self):
        msgs = [
            HumanMessage(content="Hi"),
            AIMessage(content="First answer"),
            AIMessage(content="Better answer"),
        ]
        assert self._call(msgs) == "Better answer"

    def test_skips_ai_messages_with_tool_calls(self):
        ai_with_tools = AIMessage(content="")
        ai_with_tools.tool_calls = [{"name": "search", "args": {}, "id": "tc1"}]
        msgs = [
            HumanMessage(content="Hi"),
            ai_with_tools,
            AIMessage(content="Final text"),
        ]
        assert self._call(msgs) == "Final text"

    def test_handles_list_content(self):
        msg = AIMessage(content=[{"type": "text", "text": "From list"}])
        assert self._call([msg]) == "From list"

    def test_empty_messages_returns_fallback(self):
        result = self._call([])
        assert "couldn't formulate" in result.lower() or result  # non-empty fallback

    def test_only_human_messages_returns_fallback(self):
        msgs = [HumanMessage(content="Hello")]
        result = self._call(msgs)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# _partition_tools
# ---------------------------------------------------------------------------

class TestPartitionTools:
    def _make_tool(self, name: str):
        t = MagicMock()
        t.name = name
        return t

    def test_tools_routed_by_prefix(self):
        from app.agent import _partition_tools, AGENT_CONFIGS
        prefixes = [cfg["tool_prefix"] for cfg in AGENT_CONFIGS]
        tools = [self._make_tool(f"{p}_some_action") for p in prefixes]
        buckets = _partition_tools(tools)
        for cfg, tool in zip(AGENT_CONFIGS, tools):
            assert tool in buckets[cfg["name"]]

    def test_unmatched_tool_goes_to_last_agent(self):
        from app.agent import _partition_tools, AGENT_CONFIGS
        unmatched = self._make_tool("totally_unknown_tool_xyz")
        buckets = _partition_tools([unmatched])
        last_agent = AGENT_CONFIGS[-1]["name"]
        assert unmatched in buckets[last_agent]

    def test_empty_tools_returns_empty_buckets(self):
        from app.agent import _partition_tools, AGENT_CONFIGS
        buckets = _partition_tools([])
        for cfg in AGENT_CONFIGS:
            assert buckets[cfg["name"]] == []


# ---------------------------------------------------------------------------
# _MCP_CONNECTIVITY_MARKERS — module-level constant
# ---------------------------------------------------------------------------

class TestMcpConnectivityMarkers:
    def test_is_module_level_tuple(self):
        import app.agent as agent_module
        assert hasattr(agent_module, "_MCP_CONNECTIVITY_MARKERS")
        assert isinstance(agent_module._MCP_CONNECTIVITY_MARKERS, tuple)

    def test_contains_expected_prefixes(self):
        from app.agent import _MCP_CONNECTIVITY_MARKERS
        expected = {
            "MCP server connection failed",
            "MCP server timed out",
            "MCP server returned HTTP",
            "MCP StreamableHTTP transport error",
            "MCP protocol error",
            "Network connectivity error",
        }
        assert expected == set(_MCP_CONNECTIVITY_MARKERS)

    def test_handle_mcp_tool_errors_produces_marker_prefixes(self):
        """Error messages from handle_mcp_tool_errors must start with a known marker."""
        from app.agent import handle_mcp_tool_errors, _MCP_CONNECTIVITY_MARKERS

        errs = [
            httpx.ConnectError("refused"),
            httpx.ReadTimeout("slow"),
            ConnectionError("reset"),
        ]
        for err in errs:
            msg = handle_mcp_tool_errors(err)
            assert any(marker in msg for marker in _MCP_CONNECTIVITY_MARKERS), (
                f"Message '{msg[:80]}' doesn't contain any known marker"
            )


# ---------------------------------------------------------------------------
# _make_tool_result_emitter factory
# ---------------------------------------------------------------------------

class TestMakeToolResultEmitter:
    def test_returns_async_callable(self):
        from app.agent import _make_tool_result_emitter
        emitter = _make_tool_result_emitter("test_agent", "test_server")
        assert asyncio.iscoroutinefunction(emitter)

    def test_returns_async_callable_without_server_key(self):
        from app.agent import _make_tool_result_emitter
        emitter = _make_tool_result_emitter("orchestrator")
        assert asyncio.iscoroutinefunction(emitter)

    def test_different_agents_produce_independent_callables(self):
        from app.agent import _make_tool_result_emitter
        e1 = _make_tool_result_emitter("agent_a", "server_a")
        e2 = _make_tool_result_emitter("agent_b", "server_b")
        assert e1 is not e2


# ---------------------------------------------------------------------------
# warm_up
# ---------------------------------------------------------------------------

class TestWarmUp:
    def test_warm_up_is_async(self):
        from app.agent import warm_up
        assert asyncio.iscoroutinefunction(warm_up)

    def test_warm_up_is_exported(self):
        import app.agent as agent_module
        assert hasattr(agent_module, "warm_up")


# ---------------------------------------------------------------------------
# Message trimming helpers
# ---------------------------------------------------------------------------

class TestTrimMessages:
    def _make_msgs(self, n: int):
        msgs = [SystemMessage(content="System")]
        for i in range(n):
            msgs.append(HumanMessage(content=f"Q{i}"))
            msgs.append(AIMessage(content=f"A{i}"))
        return msgs

    def test_trim_subagent_below_limit_unchanged(self):
        from app.agent import _trim_subagent_messages, MAX_SUBAGENT_MESSAGES
        msgs = self._make_msgs(2)
        assert len(msgs) <= MAX_SUBAGENT_MESSAGES
        result = _trim_subagent_messages(msgs)
        assert result == msgs

    def test_trim_subagent_above_limit_truncates(self):
        from app.agent import _trim_subagent_messages, MAX_SUBAGENT_MESSAGES
        msgs = self._make_msgs(MAX_SUBAGENT_MESSAGES + 5)
        result = _trim_subagent_messages(msgs)
        assert len(result) <= MAX_SUBAGENT_MESSAGES

    def test_trim_subagent_preserves_system_message(self):
        from app.agent import _trim_subagent_messages, MAX_SUBAGENT_MESSAGES
        msgs = self._make_msgs(MAX_SUBAGENT_MESSAGES + 5)
        result = _trim_subagent_messages(msgs)
        assert any(isinstance(m, SystemMessage) for m in result)
