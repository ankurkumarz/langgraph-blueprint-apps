"""
Integration tests for app/app.py FastAPI endpoints.

warm_up() is mocked in the client fixture (conftest.py) so no live MCP
connections are made during test startup.  stream_agent() is mocked per-test
where needed.
"""

import copy
import json
from unittest.mock import AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# Health & readiness probes
# ---------------------------------------------------------------------------

class TestHealthProbes:
    def test_healthz_returns_200(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_readyz_returns_200_when_running(self, client):
        resp = client.get("/readyz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_readyz_returns_503_when_shutting_down(self, client):
        with patch("app.app.is_shutting_down", return_value=True):
            resp = client.get("/readyz")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "shutting_down"


# ---------------------------------------------------------------------------
# Security headers
# ---------------------------------------------------------------------------

class TestSecurityHeaders:
    REQUIRED_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "strict-origin-when-cross-origin",
    }

    def test_security_headers_present(self, client):
        resp = client.get("/healthz")
        for header, value in self.REQUIRED_HEADERS.items():
            assert resp.headers.get(header) == value, (
                f"Header {header!r} missing or wrong: {resp.headers.get(header)!r}"
            )

    def test_content_security_policy_present(self, client):
        resp = client.get("/healthz")
        csp = resp.headers.get("Content-Security-Policy", "")
        assert csp, "Content-Security-Policy header is missing"
        assert "default-src" in csp
        assert "object-src 'none'" in csp

    def test_strict_transport_security_present(self, client):
        resp = client.get("/healthz")
        hsts = resp.headers.get("Strict-Transport-Security", "")
        assert hsts, "Strict-Transport-Security header is missing"
        assert "max-age=" in hsts

    def test_permissions_policy_present(self, client):
        resp = client.get("/healthz")
        pp = resp.headers.get("Permissions-Policy", "")
        assert "camera=()" in pp
        assert "microphone=()" in pp


# ---------------------------------------------------------------------------
# Incidents endpoints
# ---------------------------------------------------------------------------

class TestIncidents:
    def test_list_incidents_returns_200(self, client):
        resp = client.get("/api/incidents")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
        assert len(resp.json()) > 0

    def test_get_known_incident_returns_200(self, client):
        resp = client.get("/api/incidents/INC-4821")
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == "INC-4821"

    def test_get_unknown_incident_returns_404(self, client):
        """Fixed: must return 404, not 200 with an error body."""
        resp = client.get("/api/incidents/NONEXISTENT-9999")
        assert resp.status_code == 404

    def test_filter_by_severity(self, client):
        resp = client.get("/api/incidents?severity=P1")
        assert resp.status_code == 200
        for inc in resp.json():
            assert inc["severity"] == "P1"


# ---------------------------------------------------------------------------
# Monitoring health — _SERVICES not mutated in-place
# ---------------------------------------------------------------------------

class TestMonitoringHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/api/monitoring/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "overallScore" in body
        assert "services" in body

    def test_services_list_not_mutated(self, client):
        """Repeated calls must not accumulate score drift on the module-level list."""
        from app.app import _SERVICES

        original_scores = {s["name"]: s["score"] for s in _SERVICES}

        # Call the endpoint multiple times
        for _ in range(5):
            client.get("/api/monitoring/health")

        # Module-level _SERVICES scores must be unchanged
        for svc in _SERVICES:
            assert svc["score"] == original_scores[svc["name"]], (
                f"_SERVICES[{svc['name']}] was mutated in-place: "
                f"expected {original_scores[svc['name']]}, got {svc['score']}"
            )

    def test_anomalies_returns_200(self, client):
        resp = client.get("/api/monitoring/anomalies")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_predictions_returns_200(self, client):
        resp = client.get("/api/monitoring/predictions")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


# ---------------------------------------------------------------------------
# Admin endpoints — authentication
# ---------------------------------------------------------------------------

class TestAdminAuth:
    def test_invalidate_cache_without_key_returns_422(self, client):
        resp = client.post("/api/admin/invalidate-cache")
        assert resp.status_code == 422  # missing required header

    def test_invalidate_cache_with_wrong_key_returns_403(self, client):
        resp = client.post(
            "/api/admin/invalidate-cache",
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 403

    def test_invalidate_cache_with_correct_key_returns_200(self, client, admin_key):
        resp = client.post(
            "/api/admin/invalidate-cache",
            headers={"X-API-Key": admin_key},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_shutdown_with_wrong_key_returns_403(self, client):
        resp = client.post(
            "/api/admin/shutdown",
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Chat history & search
# ---------------------------------------------------------------------------

class TestChatHistory:
    def test_history_returns_list(self, client):
        resp = client.get("/api/chat/history")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_search_empty_query_returns_empty(self, client):
        resp = client.get("/api/chat/search?q=")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_search_matching_query_returns_results(self, client):
        resp = client.get("/api/chat/search?q=kubernetes")
        assert resp.status_code == 200
        results = resp.json()
        assert isinstance(results, list)
        assert len(results) > 0
        for r in results:
            assert "highlight" in r


# ---------------------------------------------------------------------------
# Agent conversation endpoint
# ---------------------------------------------------------------------------

class TestAgentConversation:
    def test_conversation_streams_sse(self, client):
        async def mock_stream(query, *, include_debug_events=False):
            yield 'data: {"type": "text", "content": "Hello"}\n\n'
            yield 'data: {"type": "done"}\n\n'

        with patch("app.app.stream_agent", side_effect=mock_stream):
            resp = client.post(
                "/agent/conversation",
                json={"query": "test question"},
                headers={"Accept": "text/event-stream"},
            )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_conversation_rejects_empty_query(self, client):
        resp = client.post("/agent/conversation", json={"query": ""})
        assert resp.status_code == 422  # pydantic min_length=1 validation

    def test_conversation_rejects_oversized_query(self, client):
        from app.settings import settings
        big = "x" * (settings.max_query_length + 1)
        resp = client.post("/agent/conversation", json={"query": big})
        assert resp.status_code == 422
