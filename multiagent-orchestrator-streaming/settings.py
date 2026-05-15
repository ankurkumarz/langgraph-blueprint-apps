"""
Centralised, validated application settings using pydantic-settings.

All environment variables are validated at import time.  If a required
variable is missing or a value has the wrong type, the application fails
immediately with a clear error message — no silent runtime surprises.

Usage
─────
    from settings import settings

    # Required secrets (no default — startup fails if unset)
    settings.google_api_key

    # Optional tunables with validated defaults
    settings.gemini_model          # "gemini-2.5-flash"
    settings.max_agent_steps       # 25

    # Derived / computed
    settings.firecrawl_mcp_url     # https://mcp.firecrawl.dev/<key>/v2/mcp
"""

from __future__ import annotations

import logging
import secrets
from typing import Literal

from pydantic import Field, SecretStr, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings — loaded from environment variables.

    Required variables raise ``ValidationError`` at startup if missing.
    Optional variables have sensible defaults.

    The ``model_config`` block tells pydantic-settings to read from a
    ``.env`` file (if present) and to treat variable names as
    case-insensitive.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Required secrets (no default → startup fails if unset) ────────

    firecrawl_api_key: SecretStr
    google_api_key: SecretStr

    # ── Model / LLM ──────────────────────────────────────────────────

    gemini_model: str = "gemini-2.5-flash"

    # ── Orchestrator tuning ───────────────────────────────────────────

    orchestrator_mode: Literal["evaluator", "think"] = "evaluator"
    max_agent_steps: int = Field(default=25, ge=1)
    max_subagent_steps: int = Field(default=50, ge=1)
    max_resolution_rounds: int = Field(default=2, ge=0)
    max_subagent_messages: int = Field(default=30, ge=1)
    max_orchestrator_messages: int = Field(default=50, ge=1)
    mcp_init_timeout: float = Field(default=30.0, gt=0)

    # ── API / server ─────────────────────────────────────────────────

    allowed_origins: str = ""  # comma-separated
    max_query_length: int = Field(default=4000, ge=1)
    debug_mode: bool = False
    admin_api_key: str = ""
    shutdown_timeout: float = Field(default=30.0, gt=0)

    # ── MCP server URLs ──────────────────────────────────────────────

    langchain_docs_mcp_url: str = "https://docs.langchain.com/mcp"

    # ── Validators ───────────────────────────────────────────────────

    @field_validator("orchestrator_mode", mode="before")
    @classmethod
    def _normalise_mode(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("debug_mode", mode="before")
    @classmethod
    def _parse_bool(cls, v: object) -> bool:
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes")
        return bool(v)

    @model_validator(mode="after")
    def _generate_admin_key_if_empty(self) -> "Settings":
        if not self.admin_api_key:
            generated = secrets.token_urlsafe(32)
            object.__setattr__(self, "admin_api_key", generated)
            logger.warning(
                "No ADMIN_API_KEY set — generated ephemeral key: %s",
                generated,
            )
        return self

    # ── Computed / derived ───────────────────────────────────────────

    @computed_field  # type: ignore[prop-decorator]
    @property
    def firecrawl_mcp_url(self) -> str:
        return (
            f"https://mcp.firecrawl.dev/"
            f"{self.firecrawl_api_key.get_secret_value()}/v2/mcp"
        )

    @property
    def allowed_origins_list(self) -> list[str]:
        """Pre-split and stripped list of allowed CORS origins."""
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def mcp_connections(self) -> dict[str, dict]:
        """MCP connection map derived from validated URLs."""
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


# Singleton — imported by agent.py and api.py.
# Crashes immediately with a ``ValidationError`` if required env vars
# are missing or values fail validation.
settings = Settings()  # type: ignore[call-arg]
