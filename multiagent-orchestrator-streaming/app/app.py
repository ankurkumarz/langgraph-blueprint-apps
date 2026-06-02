"""
FastAPI application — routes, middleware, and mock data endpoints.

POST /agent/conversation             → SSE stream from the orchestrator agent
GET  /api/debug                      → current debug mode state
POST /api/admin/invalidate-cache     → force graph rebuild (requires X-API-Key)
POST /api/admin/shutdown             → initiate graceful shutdown (requires X-API-Key)
GET  /healthz                        → liveness probe (always 200)
GET  /readyz                         → readiness probe (503 during shutdown)
GET  /api/chat/history               → recent chat sessions (dummy)
GET  /api/chat/search?q=...         → keyword search over chat history (dummy)
GET  /api/incidents                  → active incidents with AI root-cause analysis
GET  /api/incidents/{id}             → detail for a single incident
GET  /api/monitoring/health          → AI-scored service health map
GET  /api/monitoring/anomalies       → currently detected anomalies
GET  /api/monitoring/predictions     → predictive alerts (next 4 h)
"""

import asyncio
import json
import logging
import random
import secrets
import signal
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from app.tracing import setup_mlflow_tracing

# Activate MLflow tracing *before* any LangChain / LangGraph imports
setup_mlflow_tracing()

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.agent import (
    begin_shutdown,
    get_cached_graph,
    get_inflight_count,
    invalidate_graph_cache,
    is_shutting_down,
    set_debug_mode,
    stream_agent,
    wait_for_inflight,
    warm_up,
)
from app.checkpoint import (
    get_chat_history,
    get_eval_results,
    get_eval_results_by_session,
    save_message,
    search_chat_history,
    update_message_response,
)
from app.evals import run_evals_background
from app.settings import settings

logger = logging.getLogger(__name__)

# ── Config (validated centrally in settings.py) ───────────────────────────────

ALLOWED_ORIGINS = settings.allowed_origins.split(",")
MAX_QUERY_LENGTH = settings.max_query_length
DEBUG_MODE = settings.debug_mode
ADMIN_API_KEY = settings.admin_api_key
SHUTDOWN_TIMEOUT = settings.shutdown_timeout

# ── FastAPI app ───────────────────────────────────────────────────────────────

# ── Admin Auth Dependency ──────────────────────────────────────────────────────────


async def verify_admin_key(
    x_api_key: str = Header(..., alias="X-API-Key"),
) -> None:
    """Validate the admin API key on /api/admin/* endpoints."""
    if not secrets.compare_digest(x_api_key, ADMIN_API_KEY):
        raise HTTPException(status_code=403, detail="Invalid API key")


# ── Lifespan (startup / shutdown + signal handlers) ───────────────────────


def _handle_sighup() -> None:
    """SIGHUP handler — reload config by invalidating the graph cache.

    This lets operators send ``kill -HUP <pid>`` to trigger a hot-reload
    of MCP connections and agent configs without a full restart.
    In-flight requests continue with the old graph; the next request
    builds a fresh graph from the (potentially updated) config.
    """
    logger.info("Received SIGHUP — invalidating graph cache for hot reload")
    invalidate_graph_cache()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan handler for startup and graceful shutdown."""
    # ── Startup ───────────────────────────────────────────────────────
    logger.info("Starting up — orchestrator mode=%s",
                settings.orchestrator_mode)

    # Register signal handlers on the running event loop.
    loop = asyncio.get_running_loop()

    # SIGHUP → hot-reload config (invalidate graph cache)
    try:
        loop.add_signal_handler(signal.SIGHUP, _handle_sighup)
    except (NotImplementedError, OSError):
        # SIGHUP not available on Windows
        logger.info("SIGHUP handler not available on this platform")

    # Eagerly build the orchestrator graph so the first user request is
    # not delayed by MCP tool discovery (which can take up to
    # MCP_INIT_TIMEOUT seconds).  Failure is logged but not fatal —
    # the graph will be retried on the first actual request.
    try:
        await warm_up()
        logger.info("Orchestrator graph pre-built successfully")
    except Exception:
        logger.exception(
            "Orchestrator graph pre-build failed — will retry on first request"
        )

    # SIGTERM / SIGINT → graceful shutdown is handled by uvicorn, but
    # we hook into the lifespan shutdown phase below to drain in-flight
    # requests before the process exits.

    yield  # ─── application is running ───

    # ── Shutdown ──────────────────────────────────────────────────────
    logger.info("Shutdown initiated — draining in-flight requests...")
    begin_shutdown()

    inflight = get_inflight_count()
    if inflight > 0:
        logger.info(
            "Waiting up to %.0fs for %d in-flight request(s) to complete",
            SHUTDOWN_TIMEOUT, inflight,
        )
        drained = await wait_for_inflight(timeout=SHUTDOWN_TIMEOUT)
        if drained:
            logger.info("All in-flight requests completed — shutting down")
        else:
            logger.warning(
                "Shutdown timeout expired with %d request(s) still active",
                get_inflight_count(),
            )
    else:
        logger.info("No in-flight requests — shutting down immediately")

    # Invalidate the graph cache so any dangling references are released
    invalidate_graph_cache()
    logger.info("Shutdown complete")


# ── FastAPI app ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="LangGraph Multi-Agent Streaming",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)

# Propagate debug mode to agent module
set_debug_mode(DEBUG_MODE)

# ── Middleware ────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS if o.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "object-src 'none'; "
            "base-uri 'self'"
        )
        response.headers["Strict-Transport-Security"] = (
            "max-age=63072000; includeSubDomains"
        )
        return response


app.add_middleware(SecurityHeadersMiddleware)

# ── Static files ──────────────────────────────────────────────────────────────

# Resolve static directory relative to the project root (one level above app/)
_STATIC_DIR = str(Path(__file__).resolve().parent.parent / "static")
app.mount("/agent/copilot", StaticFiles(directory=_STATIC_DIR, html=True), name="copilot")

# ── Request model ─────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)
    session_id: Optional[str] = Field(
        None,
        description=(
            "Conversation thread ID.  Reuse the same value across turns to "
            "enable multi-turn memory via the SQLite checkpointer.  "
            "Omit (or pass null) to start a fresh, stateless session."
        ),
    )
    debug: bool = Field(False, description="Include debug events in the SSE stream")


# ══════════════════════════════════════════════════════════════════════════════
#  Agent Conversation Endpoint
# ══════════════════════════════════════════════════════════════════════════════


@app.post("/agent/conversation")
async def agent_conversation(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    msg_id = save_message(session_id, request.query)

    async def _tracked_stream():
        text_chunks: list[str] = []
        async for raw in stream_agent(
            request.query,
            session_id=session_id,
            include_debug_events=request.debug,
        ):
            yield raw
            # Accumulate text_chunk content to build the final response text.
            if '"type": "text_chunk"' in raw:
                try:
                    data = json.loads(raw.split("data: ", 1)[1])
                    if data.get("type") == "text_chunk":
                        text_chunks.append(data.get("content", ""))
                except Exception:
                    pass
            elif '"type": "done"' in raw:
                final_response = "".join(text_chunks)
                update_message_response(msg_id, final_response)
                # Fire-and-forget background evals — does not block the stream.
                asyncio.create_task(
                    run_evals_background(
                        session_id=session_id,
                        query=request.query,
                        response=final_response,
                        graph=get_cached_graph(),
                    )
                )

    return StreamingResponse(
        _tracked_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Debug Mode Endpoint
# ══════════════════════════════════════════════════════════════════════════════


@app.get("/api/debug")
async def get_debug_mode():
    """Returns the current debug mode state for the frontend."""
    return {"debug": DEBUG_MODE}


@app.post("/api/admin/invalidate-cache", dependencies=[Depends(verify_admin_key)])
async def admin_invalidate_cache():
    """Force the orchestrator graph to be rebuilt on the next request.

    Use after changing MCP server configs or agent registry at runtime.
    Requires ``X-API-Key`` header matching the ADMIN_API_KEY env var.
    """
    inflight = get_inflight_count()
    invalidate_graph_cache()
    return {
        "status": "ok",
        "message": "Graph cache invalidated",
        "inflight_requests": inflight,
    }


@app.post("/api/admin/shutdown", dependencies=[Depends(verify_admin_key)])
async def admin_shutdown():
    """Initiate graceful shutdown — reject new requests, drain in-flight.

    This does NOT terminate the process; it signals the agent layer to
    reject new requests while allowing in-flight ones to complete.
    Pair with orchestration tools (Kubernetes, systemd) for full shutdown.
    """
    begin_shutdown()
    return {
        "status": "ok",
        "message": "Shutdown initiated — new requests will be rejected",
        "inflight_requests": get_inflight_count(),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Health & Readiness
# ══════════════════════════════════════════════════════════════════════════════


@app.get("/healthz")
async def healthz():
    """Liveness probe — always returns 200 if the process is running."""
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    """Readiness probe — returns 503 during graceful shutdown.

    Kubernetes / load balancers should stop routing new traffic when
    this returns non-200.  In-flight requests continue to completion.
    """
    if is_shutting_down():
        return Response(
            content='{"status": "shutting_down", "inflight": '
                    + str(get_inflight_count()) + '}',
            status_code=503,
            media_type="application/json",
        )
    return {
        "status": "ok",
        "inflight_requests": get_inflight_count(),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Chat History
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/chat/history")
async def chat_history():
    """Return recent chat sessions from the SQLite message log."""
    return get_chat_history()


@app.get("/api/chat/search")
async def chat_search(q: Optional[str] = Query(default="")):
    """Keyword search over the SQLite message log."""
    if not q or not q.strip():
        return []
    return search_chat_history(q.strip())


# ══════════════════════════════════════════════════════════════════════════════
#  Agent Eval Results
# ══════════════════════════════════════════════════════════════════════════════


@app.get("/api/eval/results")
async def eval_results():
    """Return all agent eval results (trajectory + relevance), newest first.

    Results are populated asynchronously after each conversation ends;
    allow a few seconds after receiving the ``done`` SSE event before
    querying this endpoint.
    """
    return get_eval_results()


@app.get("/api/eval/results/{session_id}")
async def eval_results_by_session(session_id: str):
    """Return trajectory and relevance eval results for one conversation thread."""
    results = get_eval_results_by_session(session_id)
    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"No eval results found for session '{session_id}'",
        )
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  AI Insights — Incidents
# ══════════════════════════════════════════════════════════════════════════════

_INCIDENTS = [
    {
        "id": "INC-4821", "severity": "P1", "title": "Payment Gateway 5xx Spike",
        "service": "payment-svc", "status": "investigating", "duration": "18m",
        "aiRootCause": "Connection pool exhaustion due to upstream Stripe API latency spike",
        "confidence": 94,
        "blastRadius": ["checkout-flow", "order-svc", "~12K users/min"],
    },
    {
        "id": "INC-4822", "severity": "P1", "title": "Redis Cluster Failover",
        "service": "redis-cluster", "status": "escalated", "duration": "42m",
        "aiRootCause": "OOM kill on primary node, replica promoted with 4min lag",
        "confidence": 97,
        "blastRadius": ["auth-svc", "session-mgr", "~45K sessions"],
    },
    {
        "id": "INC-4823", "severity": "P1", "title": "GPU Node NotReady",
        "service": "k8s-gpu-pool", "status": "mitigating", "duration": "55m",
        "aiRootCause": "NVIDIA driver crash after kernel update",
        "confidence": 89,
        "blastRadius": ["ml-inference", "model-serving", "8 GPU pods"],
    },
]


@app.get("/api/incidents")
async def get_incidents(severity: Optional[str] = Query(default=None)):
    incidents = _INCIDENTS
    if severity:
        incidents = [i for i in incidents if i["severity"] == severity]
    return incidents


@app.get("/api/incidents/{incident_id}")
async def get_incident_detail(incident_id: str):
    for inc in _INCIDENTS:
        if inc["id"] == incident_id:
            return inc
    raise HTTPException(status_code=404, detail="Incident not found")


# ══════════════════════════════════════════════════════════════════════════════
#  AI Monitoring — Proactive
# ══════════════════════════════════════════════════════════════════════════════

_SERVICES = [
    {"name": "api-gateway", "status": "healthy", "score": 98.5, "cpu": "34%", "mem": "62%", "latency": "12ms", "errRate": "0.01%"},
    {"name": "payment-svc", "status": "critical", "score": 42.1, "cpu": "94%", "mem": "87%", "latency": "4200ms", "errRate": "34.7%"},
    {"name": "auth-svc", "status": "degraded", "score": 71.3, "cpu": "67%", "mem": "78%", "latency": "340ms", "errRate": "2.1%"},
    {"name": "order-svc", "status": "healthy", "score": 96.2, "cpu": "28%", "mem": "55%", "latency": "45ms", "errRate": "0.05%"},
    {"name": "ml-inference", "status": "critical", "score": 31.2, "cpu": "12%", "mem": "34%", "latency": "timeout", "errRate": "89.2%"},
    {"name": "redis-cluster", "status": "degraded", "score": 68.5, "cpu": "45%", "mem": "92%", "latency": "8ms", "errRate": "0.5%"},
]

_ANOMALIES = [
    {"type": "critical", "title": "Memory leak — payment-svc", "service": "payment-svc", "metric": "87% → 91%"},
    {"type": "warning", "title": "Error rate spike — search-svc", "service": "search-svc", "metric": "0.1% → 1.4%"},
    {"type": "warning", "title": "CPU anomaly — api-gateway", "service": "api-gateway", "metric": "34% (expected 18-22%)"},
]


@app.get("/api/monitoring/health")
async def get_service_health():
    services = [dict(svc) for svc in _SERVICES]
    for svc in services:
        if svc["status"] == "healthy":
            svc["score"] = round(min(100, max(80, svc["score"] + random.uniform(-0.5, 0.5))), 1)
    return {
        "overallScore": round(sum(s["score"] for s in services) / len(services), 1),
        "services": services,
    }


@app.get("/api/monitoring/anomalies")
async def get_anomalies():
    return _ANOMALIES


@app.get("/api/monitoring/predictions")
async def get_predictions():
    return [
        {"type": "capacity", "title": "Redis memory exhaustion in ~2.1h", "confidence": "91%", "eta": "2.1h"},
        {"type": "failure", "title": "payment-svc OOM crash in ~47min", "confidence": "87%", "eta": "47min"},
        {"type": "scaling", "title": "Traffic surge expected at 16:00 UTC", "confidence": "82%", "eta": "1.5h"},
    ]
