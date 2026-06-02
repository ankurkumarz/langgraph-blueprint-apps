"""
LangGraph multi-agent orchestrator.

Architecture
────────────
  OrchestratorAgent  ─►  Agent Registry  ─┬─►  Doc and Knowledge Search Agent  (LangChain Docs MCP)
                                           ├─►  WebAgent       (Firecrawl MCP)
                                           └─►  ... (add more agents here)

Two orchestrator modes  (ORCHESTRATOR_MODE env var)
───────────────────────────────────────────────────
  "evaluator" (default)
      Structured multi-node graph with a separate evaluator LLM that
      judges research completeness after each agent run.  Supports
      parallel fan-out and sequential follow-up with automatic
      cross-agent escalation.
        classify → agent(s) → evaluate → follow-up? → synthesize

  "think"
      Single ReAct loop where the orchestrator LLM has think_tool and
      a generic delegate(agent_name, query) tool.  The LLM self-decides
      what to call, reflects via think_tool, and stops when satisfied.
      Parallel fan-out happens when the LLM issues multiple delegate
      calls in a single response — ToolNode runs them concurrently.
        orchestrator → tools (think/delegate) → orchestrator (loop) → END

Both modes share the same agent registry.  To add a new sub-agent,
add one entry to AGENT_CONFIGS in subagents.py and it becomes available
in both modes.

SSE event protocol (v2 — extensible)
─────────────────────────────────────
Core events:
  {"type": "llm_start"}
  {"type": "tool_call",    "name": "...", "args": {...}, "id": "..."}
  {"type": "tool_result",  "name": "...", "content": "..."}
  {"type": "text_chunk",   "content": "...", "agent": "..."}
  {"type": "text",         "content": "..."}
  {"type": "usage",        "usage": {"input_tokens": int, "output_tokens": int, "total_tokens": int, "per_model": {...}}}
  {"type": "done"}
  {"type": "error",        "detail": "..."}

Orchestration events:
  {"type": "agent_start",  "agent": "...", "role": "..."}
  {"type": "agent_end",    "agent": "...", "summary": "..."}
  {"type": "handoff",      "from": "...", "to": "...", "reason": "..."}
  {"type": "plan",         "steps": [...], "agent": "orchestrator"}
  {"type": "plan_step",    "step_id": "...", "status": "running|done"}
  {"type": "status",       "message": "...", "agent": "..."}
  {"type": "mcp_server",   "server": "...", "status": "connected|disconnected", "error?": "..."}

Debug-only events (sent only when ``debug: true`` in the POST body):
  {"type": "graph_state_update", "node": "...", "debug": {...}}
  {"type": "node_response",      "node": "...", "debug": {...}}
"""

import asyncio
import json
import logging
import operator
from typing import Annotated

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.config import get_stream_writer
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

import app.subagents as _sub
from app.checkpoint import get_checkpointer
from app.token_usage import make_callback, usage_event
from app.subagents import (
    AGENT_CONFIGS,
    GEMINI_MODEL,
    GOOGLE_API_KEY,
    MAX_AGENT_STEPS,
    MAX_RESOLUTION_ROUNDS,
    MAX_SUBAGENT_STEPS,
    MCP_CONNECTIONS,
    MCP_INIT_TIMEOUT,
    ORCHESTRATOR_MODE,
    SubAgentState,
    _build_agent_registry,
    _debug_payload,
    _extract_final_answer,
    _invoke_agent,
    _make_tool_result_emitter,
    _partition_tools,
    _think_delegation_log_var,
    _trim_orchestrator_messages,
    handle_mcp_tool_errors,
    set_debug_mode,
    sse,
)

# ══════════════════════════════════════════════════════════════════════════════
#  State Schema
# ══════════════════════════════════════════════════════════════════════════════


class OrchestratorState(TypedDict):
    """Top-level orchestrator state (evaluator mode)."""
    messages: Annotated[list, operator.add]
    intent: str
    active_agent: str
    final_answer: str
    agent_answers: dict          # {agent_name: answer_str}
    resolution_status: str       # "resolved" | "needs_<agent_name>"
    resolution_round: int
    followup_query: str


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluator-Mode Orchestrator  (ORCHESTRATOR_MODE=evaluator)
# ══════════════════════════════════════════════════════════════════════════════

# --- Prompts ---------------------------------------------------------------

_agent_names = [c["name"] for c in AGENT_CONFIGS]
_intent_lines = "\n".join(
    f'  - "{c["evaluator_intent"]}"  → {c["description"]}'
    for c in AGENT_CONFIGS
)

ORCHESTRATOR_SYSTEM_PROMPT = f"""\
You are a routing orchestrator. Your ONLY job is to classify the user's
intent and pick which specialist agent should handle the request.

Reply with EXACTLY one of these intents (nothing else):
{_intent_lines}
  - "both"      → complex questions that need multiple specialists
  - "general"   → simple greetings, chitchat, or questions you can answer
                   directly without any tools

Respond with ONLY the intent word, nothing else."""

_needs_options = "\n\n".join(
    f'  {{"status": "needs_{c["name"]}", "followup_query": "..."}}\n'
    f'    → we need research from {c["name"]}; provide a focused follow-up query'
    for c in AGENT_CONFIGS
)

EVALUATOR_SYSTEM_PROMPT = f"""\
You are a quality evaluator. Given the user's original question and the
research collected so far, decide whether the answer is complete.

You MUST reply with EXACTLY one JSON object (no markdown, no explanation):

  {{"status": "resolved"}}
    → the research fully answers the question

{_needs_options}

IMPORTANT:
- If only ONE source has been consulted so far and it clearly cannot
  answer the question, request a DIFFERENT source.
- Be pragmatic: if the existing research is 80%+ sufficient, return
  "resolved" and let the synthesizer fill minor gaps.  Only request
  a follow-up when there is a clear, significant gap."""


def build_evaluator_orchestrator(registry: dict[str, dict], checkpointer=None):
    """
    Build the evaluator-mode orchestrator graph.

    All agent paths converge on the evaluate node, which can escalate
    to any other agent.  Supports N agents, parallel fan-out, and
    sequential follow-up with automatic cross-agent escalation.

    document / web / ...:
      classify → agent → evaluate ──┬─► synthesize → END
                                    └─► followup_agent → evaluate (loop)
    both:
      classify → parallel_agents → evaluate ──┬─► synthesize → END
                                              └─► followup_agent → evaluate (loop)
    general:
      classify → direct_answer → synthesize → END
    """
    agent_names = list(registry.keys())
    intent_to_agent = {cfg["evaluator_intent"]: cfg["name"] for cfg in AGENT_CONFIGS}

    classifier_llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL, google_api_key=GOOGLE_API_KEY)
    evaluator_llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL, google_api_key=GOOGLE_API_KEY)
    direct_llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL, google_api_key=GOOGLE_API_KEY)

    # ── Nodes ─────────────────────────────────────────────────────────────

    async def classify_intent(state: OrchestratorState) -> dict:
        writer = get_stream_writer()
        writer({"type": "agent_start", "agent": "orchestrator", "role": "coordinator"})
        writer({"type": "status", "message": "Analyzing your request...",
                "agent": "orchestrator"})

        user_query = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break

        response = await classifier_llm.ainvoke([
            SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
            HumanMessage(content=user_query),
        ])
        raw = response.content.strip().lower().strip('"').strip("'")

        if "both" in raw:
            intent = "both"
        else:
            intent = "general"
            for key in intent_to_agent:
                if key in raw:
                    intent = key
                    break

        plan_steps = [{"id": "classify", "label": "Classify intent"}]
        if intent == "both":
            for name in agent_names:
                plan_steps.append({"id": f"execute_{name}",
                                   "label": f"Run {name}"})
        elif intent != "general":
            agent = intent_to_agent[intent]
            plan_steps.append({"id": "execute", "label": f"Run {agent}"})
        else:
            plan_steps.append({"id": "execute", "label": "Direct answer"})
        if intent != "general":
            plan_steps.append({"id": "evaluate",
                               "label": "Evaluate completeness"})
        plan_steps.append({"id": "respond", "label": "Generate response"})

        writer({"type": "plan", "steps": plan_steps, "agent": "orchestrator"})
        writer({"type": "plan_step", "step_id": "classify", "status": "done"})

        if _sub._DEBUG_MODE:
            writer(_debug_payload(
                {"type": "node_response", "node": "classify_intent", "agent": "orchestrator"},
                raw_llm_output=raw,
                resolved_intent=intent,
                plan_steps=plan_steps,
            ))

        return {"intent": intent, "active_agent": intent, "resolution_round": 0}

    def route_after_classify(state: OrchestratorState) -> str:
        intent = state["intent"]
        if intent == "both":
            return "parallel_agents"
        elif intent in intent_to_agent:
            return intent_to_agent[intent]
        return "direct_answer"

    # -- Single-agent nodes (one per registered agent) ---------------------

    async def _make_agent_node(agent_name: str, state: OrchestratorState) -> dict:
        writer = get_stream_writer()
        writer({"type": "handoff", "from": "orchestrator", "to": agent_name,
                "reason": f"{agent_name} lookup requested"})
        answer = await _invoke_agent(registry, agent_name,
                                     state["messages"], writer, "execute")
        answers = dict(state.get("agent_answers", {}))
        answers[agent_name] = answer
        return {"agent_answers": answers}

    # -- Parallel fan-out --------------------------------------------------

    async def parallel_agents_node(state: OrchestratorState) -> dict:
        writer = get_stream_writer()
        writer({"type": "handoff", "from": "orchestrator",
                "to": "parallel_agents",
                "reason": "Complex query — running all agents in parallel"})

        async def _run(name: str) -> tuple[str, str]:
            answer = await _invoke_agent(
                registry, name, state["messages"], writer,
                f"execute_{name}")
            return name, answer

        results = await asyncio.gather(*[_run(n) for n in agent_names])
        return {"agent_answers": dict(results)}

    # -- Direct answer (no tools) ------------------------------------------

    async def direct_answer_node(state: OrchestratorState) -> dict:
        writer = get_stream_writer()
        writer({"type": "agent_start", "agent": "orchestrator",
                "role": "direct responder"})
        writer({"type": "plan_step", "step_id": "execute", "status": "running"})

        response = await direct_llm.ainvoke(state["messages"])
        content = response.content
        if isinstance(content, list):
            content = "".join(
                p["text"] if isinstance(p, dict) else str(p) for p in content)

        writer({"type": "plan_step", "step_id": "execute", "status": "done"})
        return {"final_answer": content}

    # -- Evaluate completeness ---------------------------------------------

    async def evaluate_node(state: OrchestratorState) -> dict:
        writer = get_stream_writer()
        writer({"type": "plan_step", "step_id": "evaluate", "status": "running"})
        writer({"type": "status",
                "message": "Evaluating research completeness…",
                "agent": "orchestrator"})

        user_query = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break

        current_round = state.get("resolution_round", 0)
        answers = state.get("agent_answers", {})

        research_sections = []
        for name in agent_names:
            answer = answers.get(name, "")
            label = "(not consulted yet)" if not answer else answer
            research_sections.append(f"## {name}\n{label}")

        eval_input = (
            f"## Original Question\n{user_query}\n\n"
            + "\n\n".join(research_sections)
            + f"\n\n## Resolution Round\n{current_round + 1} of "
            f"{MAX_RESOLUTION_ROUNDS}"
        )

        if current_round >= MAX_RESOLUTION_ROUNDS:
            writer({"type": "status",
                    "message": "Max rounds reached — synthesizing",
                    "agent": "orchestrator"})
            writer({"type": "plan_step", "step_id": "evaluate",
                    "status": "done"})
            return {"resolution_status": "resolved",
                    "resolution_round": current_round}

        response = await evaluator_llm.ainvoke([
            SystemMessage(content=EVALUATOR_SYSTEM_PROMPT),
            HumanMessage(content=eval_input),
        ])

        raw = response.content.strip()
        try:
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            verdict = json.loads(raw)
        except (json.JSONDecodeError, IndexError):
            logging.warning("Evaluator unparseable: %s", raw)
            verdict = {"status": "resolved"}

        status = verdict.get("status", "resolved")
        followup = verdict.get("followup_query", "")

        valid = {"resolved"} | {f"needs_{n}" for n in agent_names}
        if status not in valid:
            status = "resolved"

        writer({"type": "status",
                "message": f"Evaluation: {status}"
                + (f" — {followup}" if followup else ""),
                "agent": "orchestrator"})
        writer({"type": "plan_step", "step_id": "evaluate", "status": "done"})

        if status != "resolved":
            target = status.replace("needs_", "")
            writer({"type": "plan", "steps": [
                {"id": f"followup_r{current_round + 1}",
                 "label": f"Follow-up: {target} (round {current_round + 1})"},
                {"id": "evaluate", "label": "Re-evaluate"},
                {"id": "respond", "label": "Generate response"},
            ], "agent": "orchestrator"})

        if _sub._DEBUG_MODE:
            writer(_debug_payload(
                {"type": "node_response", "node": "evaluate", "agent": "orchestrator"},
                raw_evaluator_output=raw,
                parsed_verdict=verdict,
                resolution_status=status,
                followup_query=followup,
                resolution_round=current_round + 1,
                eval_input_preview=eval_input[:2000],
            ))

        return {
            "resolution_status": status,
            "resolution_round": current_round + 1,
            "followup_query": followup,
        }

    def route_after_evaluate(state: OrchestratorState) -> str:
        status = state.get("resolution_status", "resolved")
        for name in agent_names:
            if status == f"needs_{name}":
                return f"followup_{name}"
        return "synthesize"

    # -- Follow-up agent nodes (one per registered agent) ------------------

    async def _make_followup_node(
        agent_name: str, state: OrchestratorState,
    ) -> dict:
        writer = get_stream_writer()
        rnd = state.get("resolution_round", 1)
        step_id = f"followup_r{rnd}"

        writer({"type": "handoff", "from": "orchestrator", "to": agent_name,
                "reason": f"Follow-up round {rnd}"})

        followup_q = state.get("followup_query", "")
        answers = state.get("agent_answers", {})

        context_parts = []
        for other_name, other_answer in answers.items():
            if other_name != agent_name and other_answer:
                context_parts.append(
                    f"Previous {other_name} research:\n{other_answer}")
        context_parts.append(f"Focus on:\n{followup_q}")
        messages = [HumanMessage(content="\n\n".join(context_parts))]

        answer = await _invoke_agent(registry, agent_name,
                                     messages, writer, step_id)

        prev = answers.get(agent_name, "")
        combined = (f"{prev}\n\n---\n### Follow-up (round {rnd})\n{answer}"
                    if prev else answer)
        new_answers = dict(answers)
        new_answers[agent_name] = combined
        return {"agent_answers": new_answers}

    # -- Synthesize --------------------------------------------------------

    async def synthesize(state: OrchestratorState) -> dict:
        writer = get_stream_writer()
        writer({"type": "plan_step", "step_id": "respond", "status": "running"})

        answers = state.get("agent_answers", {})
        filled = {k: v for k, v in answers.items() if v}

        if len(filled) > 1:
            user_query = ""
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    user_query = msg.content
                    break

            sections = "\n\n".join(
                f"## {name}\n{ans}" for name, ans in filled.items())
            merge_prompt = (
                f"## Original Question\n{user_query}\n\n"
                "Combine the following research into one comprehensive, "
                "well-structured answer. Cite which source each fact comes "
                f"from.\n\n{sections}"
            )
            merged = await direct_llm.ainvoke([
                SystemMessage(content="You synthesise multi-source research "
                              "into a single authoritative answer."),
                HumanMessage(content=merge_prompt),
            ])
            answer = merged.content
            if isinstance(answer, list):
                answer = "".join(
                    p["text"] if isinstance(p, dict) else str(p)
                    for p in answer)
        elif filled:
            answer = next(iter(filled.values()))
        else:
            answer = state.get("final_answer",
                               "I wasn't able to generate a response.")

        writer({"type": "text", "content": answer})
        writer({"type": "agent_end", "agent": "orchestrator",
                "summary": "Response complete"})
        writer({"type": "plan_step", "step_id": "respond", "status": "done"})

        if _sub._DEBUG_MODE:
            writer(_debug_payload(
                {"type": "node_response", "node": "synthesize", "agent": "orchestrator"},
                source_count=len(filled),
                sources=list(filled.keys()),
                answer_length=len(answer),
            ))

        return {"messages": [AIMessage(content=answer)]}

    # ── Build Graph ───────────────────────────────────────────────────────

    builder = StateGraph(OrchestratorState)

    builder.add_node("classify_intent", classify_intent)
    builder.add_node("direct_answer", direct_answer_node)
    builder.add_node("parallel_agents", parallel_agents_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("synthesize", synthesize)

    route_classify_targets = ["parallel_agents", "direct_answer"]
    route_evaluate_targets = ["synthesize"]

    for name in agent_names:
        async def _agent_node(state, _name=name):
            return await _make_agent_node(_name, state)
        builder.add_node(name, _agent_node)
        route_classify_targets.append(name)
        builder.add_edge(name, "evaluate")

        followup_name = f"followup_{name}"
        async def _followup_node(state, _name=name):
            return await _make_followup_node(_name, state)
        builder.add_node(followup_name, _followup_node)
        route_evaluate_targets.append(followup_name)
        builder.add_edge(followup_name, "evaluate")

    builder.add_edge(START, "classify_intent")
    builder.add_conditional_edges(
        "classify_intent", route_after_classify, route_classify_targets)
    builder.add_edge("parallel_agents", "evaluate")
    builder.add_edge("direct_answer", "synthesize")
    builder.add_conditional_edges(
        "evaluate", route_after_evaluate, route_evaluate_targets)
    builder.add_edge("synthesize", END)

    return builder.compile(checkpointer=checkpointer)


# ══════════════════════════════════════════════════════════════════════════════
#  Think-Mode Orchestrator  (ORCHESTRATOR_MODE=think)
# ══════════════════════════════════════════════════════════════════════════════

THINK_ORCHESTRATOR_SYSTEM_PROMPT = """\
You are an orchestrator agent with access to specialist sub-agents.
Your job is to answer the user's question by delegating research to
the right sub-agents, reflecting on results, and synthesizing a final
answer.

Available tools:
  • think_tool  — reflect on what you know, what's missing, and what
                  to do next.  MUST be used after every delegation.
  • delegate    — send a focused query to one sub-agent by name.

Available sub-agents:
{agent_descriptions}

Parallel execution:
  To run multiple agents in parallel, call `delegate` multiple times
  in the SAME response (one call per agent, each with its own query).
  They will execute concurrently.

Workflow:
  1. Read the user's question.
  2. Decide which agent(s) to call (or answer directly for greetings).
  3. Call delegate(agent_name=..., query=...) for each agent you need.
     For parallel execution, include multiple delegate calls in one response.
  4. Use think_tool to assess: Is the answer complete?  What's missing?
  5. If gaps remain, delegate again with a refined query.
  6. When satisfied, respond directly to the user with the final answer.
     Do NOT call any more tools — just write the answer.

Rules:
  • You MUST use think_tool after each round of delegation to reflect.
  • Do NOT delegate more than {max_rounds} times total.
  • If one agent can't answer, try a different one.
  • For complex questions, delegate to multiple agents in parallel.
  • When you have enough information, stop calling tools and respond.
"""


def build_think_orchestrator(registry: dict[str, dict], checkpointer=None):
    """
    Build the think-tool orchestrator: a single ReAct loop.

      START → orchestrator → tools_condition ─┬─► tools → orchestrator (loop)
                                              └─► END
    """
    agent_names = sorted(registry.keys())
    agent_names_str = ", ".join(f'"{n}"' for n in agent_names)

    agent_desc_lines = [
        f'  • "{name}" — {registry[name]["description"]}'
        for name in agent_names
    ]
    agent_descriptions_block = "\n".join(agent_desc_lines)

    # ── Delegation tracking ────────────────────────────────────────────────
    #
    # Per-request log shared between delegate() and think_tool() via
    # ContextVar.  Each delegate call appends a record so think_tool
    # can return a real state summary instead of a static string —
    # forcing the LLM to process actual research results rather than
    # hallucinating a reflection.

    def _get_delegation_log() -> list[dict]:
        """Get the per-request delegation log, creating it if needed."""
        log = _think_delegation_log_var.get(None)
        if log is None:
            log = []
            _think_delegation_log_var.set(log)
        return log

    # ── Tools ─────────────────────────────────────────────────────────────

    @tool
    def think_tool(reflection: str) -> str:
        """Reflect on research progress and decide next steps.

        Use this after every round of delegation to assess:
        - What key information did I get?
        - Is it sufficient to answer the user?
        - What specific gaps remain?
        - Should I delegate again or respond now?

        Args:
            reflection: Your analysis of findings, gaps, and next action.

        Returns:
            A structured summary of all delegations so far, including which
            agents were consulted, their queries, and answer previews.
        """
        delegation_log = _get_delegation_log()

        if not delegation_log:
            return (
                "Reflection noted. No delegations have been made yet.\n"
                f"Available agents: {agent_names_str}\n"
                "Decide which agent(s) to consult, or respond directly."
            )

        total = len(delegation_log)
        succeeded = sum(1 for d in delegation_log if d["status"] == "ok")
        failed = total - succeeded
        agents_consulted = sorted(set(d["agent"] for d in delegation_log))
        agents_not_consulted = sorted(
            set(agent_names) - set(d["agent"] for d in delegation_log)
        )

        lines = [
            "═══ Research State Summary ═══",
            f"Delegations: {total} total, {succeeded} succeeded, {failed} failed",
            f"Agents consulted: {', '.join(agents_consulted)}",
        ]
        if agents_not_consulted:
            lines.append(
                f"Agents NOT yet consulted: {', '.join(agents_not_consulted)}"
            )

        lines.append("")
        for i, entry in enumerate(delegation_log, 1):
            status_icon = "✓" if entry["status"] == "ok" else "✗"
            lines.append(f"── Delegation {i} [{status_icon}] ──")
            lines.append(f"  Agent: {entry['agent']}")
            lines.append(f"  Query: {entry['query']}")
            preview = entry["answer_preview"]
            lines.append(f"  Result preview ({entry['answer_len']} chars total):")
            lines.append(f"    {preview}")
            lines.append("")

        lines.append(
            "Based on the above, decide: is the research sufficient to "
            "answer the user's question, or do you need to delegate again "
            "with a more focused query?"
        )
        return "\n".join(lines)

    @tool
    async def delegate(agent_name: str, query: str) -> str:
        """Delegate a research query to a specialist sub-agent.

        To run agents IN PARALLEL, call `delegate` multiple times in the
        SAME response — each call runs concurrently.

        Args:
            agent_name: The name of the agent to delegate to.
            query: A focused research question for this agent.

        Returns:
            The agent's research findings.
        """
        if agent_name not in registry:
            error_msg = (
                f"Unknown agent '{agent_name}'. "
                f"Available agents: {agent_names_str}"
            )
            _get_delegation_log().append({
                "agent": agent_name,
                "query": query,
                "answer_preview": error_msg,
                "answer_len": 0,
                "status": "error",
            })
            return error_msg

        entry = registry[agent_name]
        writer = get_stream_writer()
        writer({"type": "handoff", "from": "orchestrator", "to": agent_name,
                "reason": query[:200]})
        writer({"type": "agent_start", "agent": agent_name,
                "role": entry["role"]})
        writer({"type": "mcp_server", "server": entry["mcp_server"],
                "status": "connected"})

        final_messages = []
        hit_limit = False
        try:
            async for mode, chunk in entry["graph"].astream(
                {"messages": [HumanMessage(content=query)]},
                stream_mode=["custom", "updates"],
                config={"recursion_limit": MAX_SUBAGENT_STEPS},
            ):
                if mode == "custom":
                    writer(chunk)
                elif mode == "updates":
                    for _node_name, state_delta in chunk.items():
                        for msg in state_delta.get("messages", []):
                            final_messages.append(msg)
        except GraphRecursionError:
            hit_limit = True
            logging.warning(
                "Sub-agent '%s' hit recursion limit (%s steps) for query: %s",
                agent_name, MAX_SUBAGENT_STEPS, query[:200],
            )
            writer({"type": "error",
                    "detail": f"The {agent_name} needed more research steps than allowed. "
                               "Returning the best partial answer so far.",
                    "agent": agent_name})
        answer = _extract_final_answer(final_messages)

        _get_delegation_log().append({
            "agent": agent_name,
            "query": query,
            "answer_preview": answer[:500] + ("…" if len(answer) > 500 else ""),
            "answer_len": len(answer),
            "status": "truncated" if hit_limit else "ok",
        })

        writer({"type": "agent_end", "agent": agent_name,
                "summary": f"{agent_name} complete"})
        return answer

    # ── Build ReAct graph ─────────────────────────────────────────────────

    orchestrator_tools = [think_tool, delegate]

    orch_llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
    ).bind_tools(orchestrator_tools)

    system_prompt = (
        THINK_ORCHESTRATOR_SYSTEM_PROMPT
        .replace("{agent_descriptions}", agent_descriptions_block)
        .replace("{max_rounds}", str(MAX_RESOLUTION_ROUNDS + 2))
    )

    async def orchestrator_agent_node(state: SubAgentState) -> dict:
        writer = get_stream_writer()
        writer({"type": "llm_start", "agent": "orchestrator"})

        msgs = list(state["messages"])
        if not any(isinstance(m, SystemMessage) for m in msgs):
            msgs.insert(0, SystemMessage(content=system_prompt))

        msgs = _trim_orchestrator_messages(msgs)

        response: AIMessage = await orch_llm.ainvoke(msgs)

        if _sub._DEBUG_MODE:
            raw_content = response.content
            if isinstance(raw_content, list):
                raw_content = "".join(
                    p["text"] if isinstance(p, dict) else str(p) for p in raw_content
                )
            writer(_debug_payload(
                {"type": "node_response", "node": "orchestrator", "agent": "orchestrator"},
                raw_content=raw_content,
                tool_calls=[
                    {"name": tc["name"], "args": tc["args"], "id": tc["id"]}
                    for tc in response.tool_calls
                ],
                message_type=response.__class__.__name__,
                has_final_answer=not bool(response.tool_calls),
            ))

        for tc in response.tool_calls:
            writer(_debug_payload(
                {
                    "type": "tool_call",
                    "name": tc["name"],
                    "args": tc["args"],
                    "id": tc["id"],
                    "agent": "orchestrator",
                },
                full_args=tc["args"],
            ))

        if not response.tool_calls:
            content = response.content
            if isinstance(content, list):
                content = "".join(
                    p["text"] if isinstance(p, dict) else str(p)
                    for p in content)
            writer({"type": "text", "content": content})
            writer({"type": "agent_end", "agent": "orchestrator",
                    "summary": "Response complete"})

        return {"messages": [response]}

    builder = StateGraph(SubAgentState)
    builder.add_node("orchestrator", orchestrator_agent_node)
    builder.add_node("tools", ToolNode(
        orchestrator_tools, handle_tool_errors=handle_mcp_tool_errors))
    builder.add_node("tool_result_emitter",
                     _make_tool_result_emitter("orchestrator"))
    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges("orchestrator", tools_condition)
    builder.add_edge("tools", "tool_result_emitter")
    builder.add_edge("tool_result_emitter", "orchestrator")

    return builder.compile(checkpointer=checkpointer)


# ══════════════════════════════════════════════════════════════════════════════
#  Singleton Graph Cache
# ══════════════════════════════════════════════════════════════════════════════
#
# Optimization: build the MCP client, discover tools, and compile the
# LangGraph orchestrator graph ONCE on first request, then reuse on
# every subsequent request.  This avoids:
#   - Reconnecting to every MCP server to list tools on each request
#   - Rebuilding the full agent registry and LangGraph graph per request
#
# The compiled graph is stateless (state is passed in per invocation),
# so sharing it across requests is safe.  Individual tool calls still
# open their own MCP sessions (connection-per-call, built into the
# langchain-mcp-adapters library).

_cached_graph = None
_graph_lock = asyncio.Lock()

# In-flight request tracking — used for graceful shutdown and safe cache
# invalidation.  Each active stream_agent() generator increments on entry
# and decrements on exit.
_inflight_count: int = 0
_inflight_lock = asyncio.Lock()
_inflight_zero = asyncio.Event()
_inflight_zero.set()
_shutting_down: bool = False


async def _get_or_build_graph():
    """Return the cached orchestrator graph, building it on first call.

    Uses asyncio.Lock to ensure only one coroutine builds the graph
    even if multiple requests arrive concurrently at startup.
    """
    global _cached_graph
    if _cached_graph is not None:
        return _cached_graph

    async with _graph_lock:
        if _cached_graph is not None:
            return _cached_graph

        logging.info("Building orchestrator graph (first request)...")
        client = MultiServerMCPClient(MCP_CONNECTIONS)
        try:
            all_tools = await asyncio.wait_for(
                client.get_tools(), timeout=MCP_INIT_TIMEOUT
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"MCP tool discovery timed out after {MCP_INIT_TIMEOUT}s. "
                f"One or more MCP servers in MCP_CONNECTIONS may be "
                f"unreachable. Increase MCP_INIT_TIMEOUT or check server "
                f"health."
            )

        tool_buckets = _partition_tools(all_tools)
        registry = _build_agent_registry(tool_buckets)
        checkpointer = get_checkpointer()

        if ORCHESTRATOR_MODE == "think":
            graph = build_think_orchestrator(registry, checkpointer=checkpointer)
        else:
            graph = build_evaluator_orchestrator(registry, checkpointer=checkpointer)

        _cached_graph = graph
        logging.info("Orchestrator graph ready (mode=%s, tools=%d, checkpointer=sqlite)",
                     ORCHESTRATOR_MODE, len(all_tools))
        return _cached_graph


def invalidate_graph_cache() -> None:
    """Force the orchestrator graph to be rebuilt on the next request.

    Call this after changing MCP_CONNECTIONS or AGENT_CONFIGS at runtime.
    In-flight requests continue using the old graph; the next request
    triggers a fresh build.
    """
    global _cached_graph
    _cached_graph = None
    logging.info("Graph cache invalidated — will rebuild on next request")


def get_cached_graph():
    """Return the cached compiled graph, or None if not yet built.

    Used by the eval module to extract graph trajectories from checkpointed
    threads without triggering a fresh graph build.
    """
    return _cached_graph


async def wait_for_inflight(timeout: float = 30.0) -> bool:
    """Wait until all in-flight requests have completed.

    Returns True if all requests finished within the timeout, False if
    the timeout expired with requests still in progress.
    """
    try:
        await asyncio.wait_for(_inflight_zero.wait(), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        logging.warning(
            "Timed out waiting for in-flight requests (%d still active)",
            _inflight_count,
        )
        return False


def get_inflight_count() -> int:
    """Return the current number of in-flight requests (for health checks)."""
    return _inflight_count


def is_shutting_down() -> bool:
    """Return True if a graceful shutdown has been initiated."""
    return _shutting_down


def begin_shutdown() -> None:
    """Mark the agent module as shutting down.

    New requests via stream_agent() will be rejected with a shutdown
    notice.  Existing in-flight requests are allowed to complete.
    """
    global _shutting_down
    _shutting_down = True
    logging.info("Shutdown initiated — rejecting new requests")


async def warm_up() -> None:
    """Eagerly build the orchestrator graph during application startup.

    Calling this from the FastAPI lifespan handler ensures MCP tool discovery
    and graph compilation happen before the first user request arrives,
    avoiding a cold-start delay of up to MCP_INIT_TIMEOUT seconds.
    """
    await _get_or_build_graph()


# ══════════════════════════════════════════════════════════════════════════════
#  Stream Entrypoint
# ══════════════════════════════════════════════════════════════════════════════


async def stream_agent(
    query: str,
    *,
    session_id: str | None = None,
    include_debug_events: bool = False,
):
    """Async generator that yields SSE strings from the orchestrator.

    Args:
        query: The user's question.
        session_id: Conversation thread identifier.  When supplied, the
            SQLite checkpointer persists state across requests that share
            the same session_id, enabling multi-turn memory.  When omitted
            each request runs in its own isolated thread.
        include_debug_events: When True *and* DEBUG_MODE is enabled on the
            server, debug-only events (``graph_state_update``,
            ``node_response``) are included in the SSE stream.  When False
            (the default) these events are suppressed so integrating apps
            receive only the core protocol events.
    """
    global _inflight_count

    if _shutting_down:
        yield sse({"type": "error",
                   "detail": "Server is shutting down. Please retry shortly."})
        return

    async with _inflight_lock:
        _inflight_count += 1
        _inflight_zero.clear()

    try:
        graph = await _get_or_build_graph()

        cb = make_callback()

        _think_delegation_log_var.set(None)

        if ORCHESTRATOR_MODE == "think":
            inputs = {"messages": [HumanMessage(content=query)]}
        else:
            inputs = {
                "messages": [HumanMessage(content=query)],
                "intent": "",
                "active_agent": "",
                "final_answer": "",
                "agent_answers": {},
                "resolution_status": "",
                "resolution_round": 0,
                "followup_query": "",
            }

        run_config: dict = {
            "recursion_limit": MAX_AGENT_STEPS,
            "callbacks": [cb],
        }
        if session_id:
            run_config["configurable"] = {"thread_id": session_id}

        logging.info("Orchestrator mode: %s, session_id: %s",
                     ORCHESTRATOR_MODE, session_id or "<stateless>")

        async for mode, chunk in graph.astream(
            inputs,
            stream_mode=["updates", "custom", "messages"],
            config=run_config,
        ):
            if mode == "custom":
                if not include_debug_events and isinstance(chunk, dict):
                    if chunk.get("type") in ("node_response", "graph_state_update"):
                        continue
                    if "debug" in chunk:
                        chunk = {k: v for k, v in chunk.items() if k != "debug"}
                yield sse(chunk)

            elif mode == "messages":
                token, metadata = chunk
                content = token.content if hasattr(token, "content") else ""
                if content:
                    yield sse({
                        "type": "text_chunk",
                        "content": content,
                        "agent": metadata.get("langgraph_node", ""),
                    })

            elif mode == "updates":
                for node_name, state_delta in chunk.items():
                    if _sub._DEBUG_MODE and include_debug_events:
                        debug_state = {}
                        for k, v in state_delta.items():
                            if k == "messages":
                                debug_state[k] = [
                                    {
                                        "type": m.__class__.__name__,
                                        "content": str(m.content)[:500] if hasattr(m, "content") else None,
                                        **({"tool_calls": [
                                            {"name": tc["name"], "args": tc["args"], "id": tc["id"]}
                                            for tc in m.tool_calls
                                        ]} if hasattr(m, "tool_calls") and m.tool_calls else {}),
                                        **({"name": m.name} if hasattr(m, "name") and m.name else {}),
                                    }
                                    for m in v
                                ]
                            else:
                                try:
                                    json.dumps(v)
                                    debug_state[k] = v
                                except (TypeError, ValueError):
                                    debug_state[k] = str(v)[:500]
                        yield sse(_debug_payload(
                            {"type": "graph_state_update", "node": node_name},
                            state_delta=debug_state,
                        ))

                    for msg in state_delta.get("messages", []):
                        if isinstance(msg, ToolMessage):
                            server = None
                            if msg.name:
                                for srv_key in MCP_CONNECTIONS:
                                    if msg.name.startswith(srv_key):
                                        server = srv_key
                                        break
                            payload = _debug_payload(
                                {
                                    "type": "tool_result",
                                    "name": msg.name,
                                    "content": str(msg.content)[:1000],
                                    **({"server": server} if server else {}),
                                    "agent": node_name,
                                },
                                full_content=str(msg.content),
                                tool_call_id=getattr(msg, "tool_call_id", None),
                            )
                            if not include_debug_events and "debug" in payload:
                                payload = {k: v for k, v in payload.items() if k != "debug"}
                            yield sse(payload)

        yield sse(usage_event(cb))
        yield sse({"type": "done"})

    except GraphRecursionError:
        logging.error(
            "Orchestrator hit recursion limit (%s steps)", MAX_AGENT_STEPS
        )
        yield sse({"type": "error",
                   "detail": "This question required more processing than the "
                              "system allows. Try breaking it into smaller, "
                              "more specific questions."})
    except Exception:
        logging.exception("stream_agent error")
        yield sse({"type": "error",
                   "detail": "An internal error occurred. Please try again."})
    finally:
        async with _inflight_lock:
            _inflight_count -= 1
            if _inflight_count <= 0:
                _inflight_count = 0
                _inflight_zero.set()
