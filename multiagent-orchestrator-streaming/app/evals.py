"""
Agent evaluations using agentevals and openevals.

Two eval types run as asyncio background tasks after each conversation:

  trajectory  — did the orchestrator's node-execution sequence make logical
                sense?  Uses agentevals graph_trajectory LLM-as-judge, which
                extracts the full step history from the SQLite checkpointer and
                asks the judge LLM to score the overall path (0.0 – 1.0).

  relevance   — does the final answer actually address the user's query?
                Uses openevals answer_relevance LLM-as-judge (0.0 – 1.0).

Both evals use the same Gemini model that powers the orchestrator so no
additional API keys are required.  Results are written to the eval_results
table in data/messages.db and exposed via GET /api/eval/results.
"""

import asyncio
import logging

from agentevals.graph_trajectory.llm import create_async_graph_trajectory_llm_as_judge
from agentevals.graph_trajectory.utils import aextract_langgraph_trajectory_from_thread
from langchain_google_genai import ChatGoogleGenerativeAI
from openevals.llm import create_async_llm_as_judge
from openevals.prompts.quality.answer_relevance import ANSWER_RELEVANCE_PROMPT

from app.checkpoint import save_eval_result
from app.subagents import GEMINI_MODEL, GOOGLE_API_KEY

logger = logging.getLogger(__name__)


def _make_judge() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )


async def run_evals_background(
    session_id: str,
    query: str,
    response: str,
    graph,
) -> None:
    """Run trajectory + relevance evals concurrently and persist results.

    Designed to be called via asyncio.create_task() — all exceptions are
    caught and logged so they never surface to the user.
    """
    if graph is None:
        logger.warning("eval skipped for session=%s: graph not yet built", session_id)
        return

    judge = _make_judge()
    results = await asyncio.gather(
        _run_trajectory_eval(session_id, graph, judge),
        _run_relevance_eval(session_id, query, response, judge),
        return_exceptions=True,
    )
    for exc in results:
        if isinstance(exc, Exception):
            logger.error("eval task raised: %s", exc, exc_info=exc)


async def _run_trajectory_eval(session_id: str, graph, judge) -> None:
    """Score the orchestrator's node-execution path via LLM-as-judge."""
    extracted = await aextract_langgraph_trajectory_from_thread(
        graph,
        {"configurable": {"thread_id": session_id}},
    )
    if not extracted["outputs"]["steps"]:
        logger.info(
            "trajectory eval skipped for session=%s: no steps in checkpoint",
            session_id,
        )
        return

    evaluator = create_async_graph_trajectory_llm_as_judge(
        judge=judge,
        feedback_key="graph_trajectory_accuracy",
        continuous=True,
    )
    result = await evaluator(
        inputs=extracted["inputs"],
        outputs=extracted["outputs"],
    )
    score = result.get("score")
    reasoning = result.get("reasoning")
    save_eval_result(
        session_id=session_id,
        eval_type="trajectory",
        score=float(score) if score is not None else None,
        reasoning=reasoning,
    )
    logger.info("trajectory eval session=%s score=%.2f", session_id, score or 0)


async def _run_relevance_eval(
    session_id: str, query: str, response: str, judge
) -> None:
    """Score whether the final answer is relevant to the user's query."""
    if not response:
        logger.info(
            "relevance eval skipped for session=%s: no response captured", session_id
        )
        return

    evaluator = create_async_llm_as_judge(
        prompt=ANSWER_RELEVANCE_PROMPT,
        judge=judge,
        feedback_key="answer_relevance",
        continuous=True,
    )
    result = await evaluator(
        inputs=query,
        outputs=response,
    )
    score = result.get("score")
    reasoning = result.get("reasoning")
    save_eval_result(
        session_id=session_id,
        eval_type="relevance",
        score=float(score) if score is not None else None,
        reasoning=reasoning,
    )
    logger.info("relevance eval session=%s score=%.2f", session_id, score or 0)
