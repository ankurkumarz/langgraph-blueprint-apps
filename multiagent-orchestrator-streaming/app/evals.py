"""
Agent evaluations using agentevals and openevals.

Three complementary eval categories run as asyncio background tasks after
each conversation:

  trajectory_match  — deterministic, zero-LLM-cost comparison of the
                      agent's actual message/tool-call sequence against a
                      stored "gold standard" reference trajectory.
                      Runs four match-mode variants in parallel:
                        • strict    – exact order + exact tool args
                        • unordered – same tool calls, any order
                        • subset    – agent only called allowed tools
                        • superset  – agent called at least the required tools
                      Only runs when a named reference trajectory is stored.

  graph_trajectory  — LLM-as-judge that reads the full LangGraph node-
                      execution history from the SQLite checkpointer and
                      scores the orchestrator's overall reasoning path
                      (0.0 – 1.0).  Uses agentevals
                      create_async_graph_trajectory_llm_as_judge.

  relevance         — LLM-as-judge that scores whether the final answer
                      actually addressed the user's query (0.0 – 1.0).
                      Uses openevals ANSWER_RELEVANCE_PROMPT.

All evals use the same Gemini model that powers the orchestrator — no
additional API keys required.  Results land in the eval_results table in
data/messages.db and are exposed via GET /api/eval/results.
"""

import asyncio
import logging
from typing import Literal, Optional

from agentevals.graph_trajectory.llm import create_async_graph_trajectory_llm_as_judge
from agentevals.graph_trajectory.utils import aextract_langgraph_trajectory_from_thread
from agentevals.trajectory.match import create_async_trajectory_match_evaluator
from langchain_core.messages import messages_to_dict
from langchain_google_genai import ChatGoogleGenerativeAI
from openevals.llm import create_async_llm_as_judge
from openevals.prompts.quality.answer_relevance import ANSWER_RELEVANCE_PROMPT

from app.checkpoint import get_reference_trajectory, save_eval_result
from app.subagents import GEMINI_MODEL, GOOGLE_API_KEY

logger = logging.getLogger(__name__)

_MATCH_MODES: tuple[str, ...] = ("strict", "unordered", "subset", "superset")


def _make_judge() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )


async def extract_session_messages(graph, session_id: str) -> list:
    """Return the current message list from the checkpointed thread.

    Uses ``graph.aget_state`` so the checkpointer (SQLite) is queried
    without replaying any graph nodes.  Returns an empty list if the
    thread has no recorded state.
    """
    try:
        snapshot = await graph.aget_state(
            {"configurable": {"thread_id": session_id}}
        )
        return snapshot.values.get("messages", []) if snapshot.values else []
    except Exception:
        logger.exception(
            "Failed to extract messages for session=%s", session_id
        )
        return []


async def run_evals_background(
    session_id: str,
    query: str,
    response: str,
    graph,
    reference_name: Optional[str] = None,
) -> None:
    """Orchestrate all eval types concurrently as a fire-and-forget task.

    Call via ``asyncio.create_task(run_evals_background(...))`` — errors
    are caught/logged and never surface to the user.
    """
    if graph is None:
        logger.warning("eval skipped for session=%s: graph not yet built", session_id)
        return

    judge = _make_judge()

    tasks = [
        _run_graph_trajectory_eval(session_id, graph, judge),
        _run_relevance_eval(session_id, query, response, judge),
    ]

    # Trajectory match only runs when a reference is available.
    if reference_name:
        reference = get_reference_trajectory(reference_name)
        if reference:
            actual_messages = await extract_session_messages(graph, session_id)
            if actual_messages:
                tasks.append(
                    _run_all_match_modes(
                        session_id, actual_messages, reference["messages"]
                    )
                )
            else:
                logger.warning(
                    "match eval skipped for session=%s: no messages in checkpoint",
                    session_id,
                )
        else:
            logger.warning(
                "match eval skipped for session=%s: reference '%s' not found",
                session_id,
                reference_name,
            )

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for exc in results:
        if isinstance(exc, Exception):
            logger.error("eval task raised: %s", exc, exc_info=exc)


async def _run_all_match_modes(
    session_id: str,
    actual_messages: list,
    reference_messages: list,
) -> None:
    """Run all four trajectory match modes concurrently and save each score."""
    await asyncio.gather(
        *[
            _run_single_match_eval(session_id, actual_messages, reference_messages, mode)
            for mode in _MATCH_MODES
        ],
        return_exceptions=True,
    )


async def _run_single_match_eval(
    session_id: str,
    actual_messages: list,
    reference_messages: list,
    mode: Literal["strict", "unordered", "subset", "superset"],
) -> None:
    """Run one trajectory match mode and persist the boolean score.

    ``tool_args_match_mode="ignore"`` is used so minor argument
    differences (e.g. formatting, whitespace) don't cause false
    failures — set to "exact" for stricter regression testing.
    """
    try:
        evaluator = create_async_trajectory_match_evaluator(
            trajectory_match_mode=mode,
            tool_args_match_mode="ignore",
        )
        result = await evaluator(
            outputs=actual_messages,
            reference_outputs=reference_messages,
        )
        score = result.get("score")
        save_eval_result(
            session_id=session_id,
            eval_type=f"match_{mode}",
            score=1.0 if score is True else (0.0 if score is False else score),
            reasoning=result.get("comment"),
        )
        logger.info(
            "match_%s eval session=%s score=%s", mode, session_id, score
        )
    except Exception:
        logger.exception(
            "match_%s eval failed for session=%s", mode, session_id
        )


async def _run_graph_trajectory_eval(session_id: str, graph, judge) -> None:
    """Score the orchestrator's node-execution path via LLM-as-judge (0–1)."""
    extracted = await aextract_langgraph_trajectory_from_thread(
        graph,
        {"configurable": {"thread_id": session_id}},
    )
    if not extracted["outputs"]["steps"]:
        logger.info(
            "graph_trajectory eval skipped for session=%s: no steps in checkpoint",
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
    save_eval_result(
        session_id=session_id,
        eval_type="graph_trajectory",
        score=float(score) if score is not None else None,
        reasoning=result.get("comment"),
    )
    logger.info("graph_trajectory eval session=%s score=%.2f", session_id, score or 0)


async def _run_relevance_eval(
    session_id: str, query: str, response: str, judge
) -> None:
    """Score whether the final answer is relevant to the user's query (0–1)."""
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
    save_eval_result(
        session_id=session_id,
        eval_type="relevance",
        score=float(score) if score is not None else None,
        reasoning=result.get("comment"),
    )
    logger.info("relevance eval session=%s score=%.2f", session_id, score or 0)
