"""Agent6 — Four-role agentic architecture (production-hardened).

Roles: Memory, Perception, Decision, Action
Substrate: LLM Gateway V3 with retry/backoff
Transport: MCP over stdio
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid
from contextlib import asynccontextmanager

# Suppress noisy MCP/httpx/crawl4ai logs
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("crawl4ai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import action
import decision
import perception
from artifacts import artifact_store
from config import settings
from logger import get_logger
from memory import memory
from schemas import Goal, SYNTHESIS_KEYWORDS

log = get_logger("agent6")


@asynccontextmanager
async def mcp_session():
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],
        env={**__import__("os").environ, "MCP_LOG_LEVEL": "error"},
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def load_tools(session: ClientSession) -> list[dict]:
    result = await session.list_tools()
    tools = []
    for tool in result.tools:
        tool_def = {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema if tool.inputSchema else {"type": "object", "properties": {}},
        }
        tools.append(tool_def)
    return tools


async def run(query: str) -> str:
    run_id = uuid.uuid4().hex[:8]
    history: list[dict] = []
    prior_goals: list[Goal] = []

    log.info("run_start", run_id=run_id, query=query[:100])

    try:
        from tracer import trace_run_start
        trace_run_start(run_id, query)
    except Exception:
        pass

    # Cleanup old artifacts on startup
    artifact_store.cleanup(max_age_hours=settings.artifact_ttl_hours)

    # Durable memory: classify the user's query
    memory.remember(query, source="user_query", run_id=run_id)

    async with mcp_session() as session:
        mcp_tools = await load_tools(session)
        log.info("mcp_connected", tool_count=len(mcp_tools))

        for it in range(1, settings.max_iterations + 1):
            log.info("iteration_start", iter=it, run_id=run_id)

            # Step 1: Memory read
            hits = memory.read(query, history)
            log.info("memory_read", count=len(hits))

            # Step 2: Perception
            obs = perception.observe(query, hits, history, prior_goals, run_id)
            prior_goals = obs.goals

            for i, g in enumerate(obs.goals):
                status = "done" if g.done else "open"
                log.info("goal", status=status, text=g.text[:80],
                         attach=g.attach_artifact_id or "", first=(i == 0))

            if obs.all_done:
                has_answer = any(e.get("kind") == "answer" for e in history)
                if not has_answer and hits:
                    log.info("generating_answer_from_memory")
                    summary_goal = obs.goals[-1]
                    out = decision.next_step(summary_goal, hits, [], history, mcp_tools)
                    if out.is_answer:
                        history.append({
                            "iter": it, "kind": "answer",
                            "goal_id": summary_goal.id, "text": out.answer,
                        })
                    elif hits:
                        fact_text = "; ".join(
                            f"{h.descriptor} ({json.dumps(h.value, default=str)})"
                            for h in hits if h.kind == "fact"
                        )
                        if fact_text:
                            history.append({
                                "iter": it, "kind": "answer",
                                "goal_id": summary_goal.id, "text": fact_text,
                            })
                log.info("all_goals_done", goal_count=len(obs.goals))
                break

            # Step 3: Get next goal and prepare attachments
            goal = obs.next_unfinished()
            if goal is None:
                break

            attached: list[tuple[str, bytes]] = []
            goal_lower = goal.text.lower()
            is_synthesis = any(kw in goal_lower for kw in SYNTHESIS_KEYWORDS)

            if is_synthesis:
                seen_arts = set()
                for h in hits:
                    if h.artifact_id and h.artifact_id not in seen_arts and artifact_store.exists(h.artifact_id):
                        blob = artifact_store.get_bytes(h.artifact_id)
                        attached.append((h.artifact_id, blob))
                        seen_arts.add(h.artifact_id)
                if attached:
                    log.info("multi_attach", count=len(attached))
            elif goal.attach_artifact_id and artifact_store.exists(goal.attach_artifact_id):
                blob = artifact_store.get_bytes(goal.attach_artifact_id)
                attached.append((goal.attach_artifact_id, blob))
                log.info("single_attach", art_id=goal.attach_artifact_id, size=len(blob))

            # Step 4: Decision
            out = decision.next_step(goal, hits, attached, history, mcp_tools)

            # Skip error iterations — don't pollute history
            if out.is_error:
                log.info("decision_error_skipped", iter=it, goal=goal.text[:60])
                continue

            if out.is_answer:
                log.info("decision_answer", text=out.answer[:200])
                history.append({
                    "iter": it,
                    "kind": "answer",
                    "goal_id": goal.id,
                    "text": out.answer,
                })
                # If this was the last unfinished goal, done
                unfinished_count = sum(1 for g in obs.goals if not g.done)
                if unfinished_count <= 1:
                    log.info("all_goals_done", goal_count=len(obs.goals))
                    break
                continue

            # Step 5: Action (tool dispatch)
            log.info("decision_tool_call", tool=out.tool_call.name,
                     args=json.dumps(out.tool_call.arguments)[:100])
            result_text, art_id = await action.execute(session, out.tool_call)

            # Log result
            if art_id:
                log.info("artifact_stored", art_id=art_id,
                         size=len(artifact_store.get_bytes(art_id)),
                         preview=result_text[:60])
            else:
                log.info("tool_result", tool=out.tool_call.name,
                         size=len(result_text), result=result_text[:80])

            memory.record_outcome(
                tool_call=out.tool_call,
                result_text=result_text,
                artifact_id=art_id,
                run_id=run_id,
                goal_id=goal.id,
            )

            history.append({
                "iter": it,
                "kind": "action",
                "goal_id": goal.id,
                "tool": out.tool_call.name,
                "arguments": out.tool_call.arguments,
                "result_descriptor": result_text[:300],
                "artifact_id": art_id,
            })

    final = _final_answer_from(history, query)
    log.info("final_answer", text=final)

    try:
        from tracer import trace_run_end
        trace_run_end(run_id, final, it if 'it' in dir() else 0)
    except Exception:
        pass

    return final


def _final_answer_from(history: list[dict], query: str) -> str:
    answers = [e["text"] for e in history if e.get("kind") == "answer"]
    if answers:
        return "\n\n".join(answers)

    # Derive facts from action history instead of re-reading memory
    actions = [e.get("result_descriptor", "") for e in history if e.get("kind") == "action"]
    if actions:
        return "Actions completed:\n" + "\n".join(f"- {a[:200]}" for a in actions)
    return "No answer produced."


async def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter your query: ")

    await run(query)


if __name__ == "__main__":
    asyncio.run(main())
