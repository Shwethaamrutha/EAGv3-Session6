"""Demo script — runs all 4 target queries in sequence for video recording."""
import asyncio
import json
import logging
import os
import shutil
import sys
import uuid
from contextlib import asynccontextmanager

logging.getLogger("mcp").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("crawl4ai").setLevel(logging.ERROR)

import structlog
structlog.configure(
    processors=[structlog.dev.ConsoleRenderer()],
    wrapper_class=structlog.make_filtering_bound_logger(50),
    logger_factory=structlog.PrintLoggerFactory(),
)

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import action
import decision
import perception
from artifacts import artifact_store
from config import settings
from memory import memory
from schemas import Goal

P = 16  # column width


@asynccontextmanager
async def mcp_session():
    server_params = StdioServerParameters(
        command="python", args=["mcp_server.py"],
        env={**os.environ, "MCP_LOG_LEVEL": "error"},
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def load_tools(session):
    result = await session.list_tools()
    return [{"name": t.name, "description": t.description or "",
             "parameters": t.inputSchema or {"type": "object", "properties": {}}}
            for t in result.tools]


async def run_query(query, session, mcp_tools):
    run_id = uuid.uuid4().hex[:8]
    history = []
    prior_goals = []

    print(f"\n{'='*70}")
    print(f"  QUERY: {query}")
    print(f"{'='*70}\n")

    mem_item = memory.remember(query, source="user_query", run_id=run_id)
    if mem_item:
        print(f"{'[memory.remember]':<{P}} stored [{mem_item.kind}] {mem_item.descriptor}")

    for it in range(1, settings.max_iterations + 1):
        print(f"\n{'─'*3} iter {it} {'─'*3}")

        hits = memory.read(query, history)
        print(f"{'[memory.read]':<{P}}{len(hits)} hits")

        obs = perception.observe(query, hits, history, prior_goals, run_id)
        prior_goals = obs.goals

        for i, g in enumerate(obs.goals):
            prefix = f"{'[perception]':<{P}}" if i == 0 else " " * P
            status = "[done]" if g.done else "[open]"
            print(f"{prefix}{status} {g.text}")
            if g.attach_artifact_id and not g.done:
                print(f"{' ' * P}  attach={g.attach_artifact_id}")

        if obs.all_done:
            has_answer = any(e.get("kind") == "answer" for e in history)
            if not has_answer:
                summary_goal = obs.goals[-1]
                out = decision.next_step(summary_goal, hits, [], history, mcp_tools)
                if out.is_answer:
                    print(f"{'[decision]':<{P}}ANSWER: {out.answer[:100]}...")
                    history.append({"iter": it, "kind": "answer", "goal_id": summary_goal.id, "text": out.answer})
            print(f"\n[done] all {len(obs.goals)} goals satisfied")
            break

        goal = obs.next_unfinished()
        if goal is None:
            break

        attached = []
        synthesis_kw = {"synthesize", "synthesise", "extract", "list", "compare",
                        "decide", "choose", "summarize", "common", "agree", "advice",
                        "tell me", "which one", "appropriate", "recommend", "most"}
        goal_tokens = set(goal.text.lower().split())
        is_synthesis = any(kw in goal.text.lower() for kw in synthesis_kw)

        if is_synthesis:
            seen = set()
            for h in hits:
                if h.artifact_id and h.artifact_id not in seen and artifact_store.exists(h.artifact_id):
                    attached.append((h.artifact_id, artifact_store.get_bytes(h.artifact_id)))
                    seen.add(h.artifact_id)
            if attached:
                print(f"{'[attach]':<{P}}{len(attached)} artifacts for synthesis")
        elif goal.attach_artifact_id and artifact_store.exists(goal.attach_artifact_id):
            blob = artifact_store.get_bytes(goal.attach_artifact_id)
            attached.append((goal.attach_artifact_id, blob))
            print(f"{'[attach]':<{P}}{goal.attach_artifact_id} ({len(blob)} bytes)")

        out = decision.next_step(goal, hits, attached, history, mcp_tools)

        if out.is_error:
            print(f"{'[decision]':<{P}}(transient error, retrying...)")
            continue

        if out.is_answer:
            print(f"{'[decision]':<{P}}ANSWER: {out.answer[:100]}...")
            history.append({"iter": it, "kind": "answer", "goal_id": goal.id, "text": out.answer})
            unfinished = sum(1 for g in obs.goals if not g.done)
            if unfinished <= 1:
                print(f"\n[done] all {len(obs.goals)} goals satisfied")
                break
            continue

        print(f"{'[decision]':<{P}}TOOL_CALL: {out.tool_call.name}({json.dumps(out.tool_call.arguments)[:80]})")
        result_text, art_id = await action.execute(session, out.tool_call)
        memory.record_outcome(tool_call=out.tool_call, result_text=result_text,
                              artifact_id=art_id, run_id=run_id, goal_id=goal.id)
        history.append({"iter": it, "kind": "action", "goal_id": goal.id,
                        "tool": out.tool_call.name, "arguments": out.tool_call.arguments,
                        "result_descriptor": result_text[:300], "artifact_id": art_id})

        # Format action output
        import re
        text = result_text.strip()
        if text.startswith("Title:"):
            titles = re.findall(r'Title:\s*(.+)', text)
            preview = "; ".join(t.strip()[:50] for t in titles[:3])
            print(f"{'[action]':<{P}}{chr(8594)} [{len(titles)} results] {preview}")
        elif art_id:
            size = len(artifact_store.get_bytes(art_id))
            print(f"{'[action]':<{P}}{chr(8594)} [artifact {art_id}, {size} bytes]")
        elif "°C" in text or "°F" in text or "wttr" in text.lower():
            lines = [l.strip() for l in text.split("\n") if l.strip() and not l.startswith("#")]
            print(f"{'[action]':<{P}}{chr(8594)} {lines[0][:80]}" if lines else f"{'[action]':<{P}}{chr(8594)} {text[:80]}")
        else:
            lines = [l.strip() for l in text.split("\n") if l.strip() and not l.startswith("#")]
            print(f"{'[action]':<{P}}{chr(8594)} {lines[0][:80]}" if lines else f"{'[action]':<{P}}{chr(8594)} {text[:80]}")

    # Final answer
    answers = [e["text"] for e in history if e.get("kind") == "answer"]
    if answers:
        final = "\n\n".join(answers)
    else:
        final = "No answer produced."

    print(f"\nFINAL: {final}\n")


async def main():
    # Clean state
    shutil.rmtree("state", ignore_errors=True)
    os.makedirs("state/artifacts", exist_ok=True)
    os.makedirs("state/sandbox", exist_ok=True)

    queries = [
        # Query A
        "Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his birth date, death date, and three key contributions to information theory.",
        # Query B
        "Find 3 family-friendly things to do in Tokyo this weekend. Check Saturday's weather forecast there and tell me which one is most appropriate.",
        # Query C - Run 1
        "My mom's birthday is 15 May 2026. Remember that and give me a calendar reminder for two weeks before and on the day.",
        # Query C - Run 2
        "When is mom's birthday?",
        # Query D
        "Search for 'Python asyncio best practices', read the top 3 results, and give me a short numbered list of the advice they agree on.",
    ]

    print("\n" + "=" * 70)
    print("  AGENT6 DEMO — All 4 Target Queries")
    print("=" * 70)

    async with mcp_session() as session:
        mcp_tools = await load_tools(session)
        print(f"\n  [{len(mcp_tools)} tools loaded]")

        for query in queries:
            await run_query(query, session, mcp_tools)

    print("\n" + "=" * 70)
    print("  ALL QUERIES COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
