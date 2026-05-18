"""
Pytest configuration and shared fixtures.

IMPORTANT: env vars MUST be set at module level here — before any app module
is imported — because settings.py instantiates Settings() at import time.
"""

import os

# Set required env vars before any app import happens.
os.environ.setdefault("FIRECRAWL_API_KEY", "test-firecrawl-key-abc123")
os.environ.setdefault("GOOGLE_API_KEY", "test-google-key-abc123")
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key-secure")

import pytest
from unittest.mock import AsyncMock, patch


@pytest.fixture
def client():
    """FastAPI test client with warm_up mocked to avoid MCP connections at startup."""
    from app.app import app
    from starlette.testclient import TestClient

    with patch("app.app.warm_up", new_callable=AsyncMock):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


@pytest.fixture
def admin_key() -> str:
    return os.environ["ADMIN_API_KEY"]
