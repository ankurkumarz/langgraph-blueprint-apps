"""
MLflow tracing integration — conditional on MLFLOW_ENABLED env var.

Usage:
    from tracing import setup_mlflow_tracing
    setup_mlflow_tracing()          # call once at startup

When MLFLOW_ENABLED=true, this module:
  1. Sets the MLflow tracking URI to the local (or configured) server.
  2. Creates / selects an experiment for this application.
  3. Enables `mlflow.langchain.autolog()` which automatically traces
     every LangChain / LangGraph invocation — including streaming —
     with zero code changes to the agent logic.

When MLFLOW_ENABLED is absent or false, this module is a no-op.
"""

import logging
import os

logger = logging.getLogger(__name__)


def setup_mlflow_tracing() -> bool:
    """
    Configure MLflow tracing if MLFLOW_ENABLED=true.

    Returns True if tracing was activated, False otherwise.
    """
    enabled = os.environ.get("MLFLOW_ENABLED", "false").strip().lower() == "true"
    if not enabled:
        logger.info("MLflow tracing is disabled (MLFLOW_ENABLED != true)")
        return False

    try:
        import mlflow
        from mlflow.langchain import autolog

        # ── Tracking URI ──────────────────────────────────────────────
        tracking_uri = os.environ.get(
            "MLFLOW_TRACKING_URI",
            "http://localhost:5050",
        )
        mlflow.set_tracking_uri(tracking_uri)
        logger.info("MLflow tracking URI: %s", tracking_uri)

        # ── Experiment ────────────────────────────────────────────────
        experiment_name = os.environ.get(
            "MLFLOW_EXPERIMENT_NAME",
            "langgraph-fastapi-streaming",
        )
        mlflow.set_experiment(experiment_name)
        logger.info("MLflow experiment: %s", experiment_name)

        # ── Auto-log LangChain / LangGraph traces ────────────────────
        # `log_traces=True` captures every chain / graph invocation,
        # including async streaming (astream), with full span hierarchy:
        #   LangGraph run → node calls → LLM calls → tool calls
        autolog(log_traces=True)
        logger.info("MLflow LangChain autolog enabled — tracing is active")

        return True

    except Exception:
        logger.exception("Failed to initialise MLflow tracing — continuing without it")
        return False
