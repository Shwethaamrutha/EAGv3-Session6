"""Agent6 CLI Chat — interactive REPL interface similar to Claude Code."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager

# Suppress noisy third-party logs
logging.getLogger("mcp").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("crawl4ai").setLevel(logging.ERROR)

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import action
import decision
import perception
from artifacts import artifact_store
from config import settings
from memory import memory
from schemas import Goal
from thinking import think, done

# Suppress structlog in chat mode — we do our own formatted output
import structlog
structlog.configure(
    processors=[structlog.dev.ConsoleRenderer()],
    wrapper_class=structlog.make_filtering_bound_logger(50),  # suppress all
    logger_factory=structlog.PrintLoggerFactory(),
)

# ANSI — soft accent palette (professional on dark terminals)
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
BLUE = "\033[38;5;75m"    # soft blue for labels
GREEN = "\033[38;5;114m"  # muted green for success
AMBER = "\033[38;5;179m"  # warm amber for open/pending
WHITE = "\033[38;5;252m"  # off-white for content
GRAY = "\033[38;5;242m"   # medium gray for borders
PURPLE = "\033[38;5;141m" # soft purple for artifacts


def print_banner():
    print(f"""
{GRAY}╭──────────────────────────────────────────────────╮{RESET}
{GRAY}│{RESET}  {BOLD}{WHITE}AGENT6{RESET}  {GRAY}— four-role agentic architecture{RESET}     {GRAY}│{RESET}
{GRAY}│{RESET}  {BLUE}Memory{RESET} {GRAY}·{RESET} {BLUE}Perception{RESET} {GRAY}·{RESET} {BLUE}Decision{RESET} {GRAY}·{RESET} {BLUE}Action{RESET}     {GRAY}│{RESET}
{GRAY}╰──────────────────────────────────────────────────╯{RESET}
  {GRAY}Type your query. Press Ctrl+C to exit.{RESET}
  {GRAY}Commands: /memory  /clear  exit{RESET}
""")


def read_input() -> str | None:
    """Read user input, supporting multi-line with blank-line termination."""
    try:
        first_line = input(f"{BLUE}{BOLD}>{RESET} ")
    except (EOFError, KeyboardInterrupt):
        return None

    if not first_line.strip():
        return None

    # Collect continuation lines (user can type multi-line, blank line ends)
    lines = [first_line]
    while True:
        try:
            cont = input(f"{GRAY}..{RESET} ")
        except (EOFError, KeyboardInterrupt):
            break
        if not cont.strip():
            break
        lines.append(cont)

    return " ".join(line.strip() for line in lines).strip()


@asynccontextmanager
async def mcp_session():
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],
        env={**os.environ, "MCP_LOG_LEVEL": "error"},
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def load_tools(session: ClientSession) -> list[dict]:
    result = await session.list_tools()
    tools = []
    for tool in result.tools:
        tools.append({
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema if tool.inputSchema else {"type": "object", "properties": {}},
        })
    return tools


async def run_query(query: str, session: ClientSession = None, mcp_tools: list[dict] = None):
    """Run a single query through the agent loop — trace format matches agent6.py."""
    run_id = uuid.uuid4().hex[:8]
    history: list[dict] = []
    prior_goals: list[Goal] = []

    # Durable memory
    P = 16
    think("memory", "Classifying query...")
    mem_item = memory.remember(query, source="user_query", run_id=run_id)
    done()
    if mem_item:
        print(f"{BLUE}{'[memory.remember]':<{P}}{RESET} stored [{mem_item.kind}] {mem_item.descriptor}")

    # Cleanup old artifacts
    artifact_store.cleanup(max_age_hours=settings.artifact_ttl_hours)

    # If no session provided, create one (one-shot mode)
    if session is None:
        async with mcp_session() as sess:
            tools = await load_tools(sess)
            return await _run_loop(query, run_id, history, prior_goals, sess, tools)
    else:
        return await _run_loop(query, run_id, history, prior_goals, session, mcp_tools)


async def _run_loop(query, run_id, history, prior_goals, session, mcp_tools):
    """Inner loop — output matches the course trace format exactly."""
    # Column width for aligned prefix (16 chars)
    P = 16

    for it in range(1, settings.max_iterations + 1):
        print(f"\n{GRAY}{'─'*3} iter {it} {'─'*3}{RESET}")

        hits = memory.read(query, history)
        print(f"{BLUE}{'[memory.read]':<{P}}{RESET}{len(hits)} hits")

        think("perception", "Analyzing goals...")
        obs = perception.observe(query, hits, history, prior_goals, run_id)
        done()
        prior_goals = obs.goals

        for i, g in enumerate(obs.goals):
            prefix = f"{BLUE}{'[perception]':<{P}}{RESET}" if i == 0 else " " * P
            if g.done:
                status = f"{GREEN}[✓]{RESET}"
            else:
                status = f"{GRAY}[ ]{RESET}"
            print(f"{prefix}{status} {g.text}")
            if g.attach_artifact_id and not g.done:
                print(f"{' ' * P}      {PURPLE}attach={g.attach_artifact_id}{RESET}")

        if obs.all_done:
            has_answer = any(e.get("kind") == "answer" for e in history)
            if not has_answer:
                # Decision must produce an explicit answer before we finish
                think("decision", "Generating answer...")
                summary_goal = obs.goals[-1]
                out = decision.next_step(summary_goal, hits, [], history, mcp_tools)
                done()
                if out.is_answer:
                    print(f"{'[decision]':<{P}}ANSWER: {out.answer[:100]}...")
                    history.append({"iter": it, "kind": "answer", "goal_id": summary_goal.id, "text": out.answer})
                elif hits:
                    # Fallback: construct from facts
                    fact_text = "; ".join(
                        f"{h.descriptor}: {json.dumps(h.value, default=str)}"
                        for h in hits if h.kind == "fact"
                    )
                    if fact_text:
                        print(f"{'[decision]':<{P}}ANSWER: {fact_text[:100]}...")
                        history.append({"iter": it, "kind": "answer", "goal_id": summary_goal.id, "text": fact_text})
            print(f"\n{GREEN}[done] all {len(obs.goals)} goals satisfied{RESET}")
            break

        goal = obs.next_unfinished()
        if goal is None:
            break

        attached: list[tuple[str, bytes]] = []
        synthesis_keywords = {"synthesize", "synthesise", "extract", "list", "compare",
                              "decide", "choose", "summarize", "common", "agree", "advice"}
        goal_tokens = set(goal.text.lower().split())
        is_synthesis = bool(goal_tokens & synthesis_keywords)

        if is_synthesis:
            seen_arts = set()
            for h in hits:
                if h.artifact_id and h.artifact_id not in seen_arts and artifact_store.exists(h.artifact_id):
                    blob = artifact_store.get_bytes(h.artifact_id)
                    attached.append((h.artifact_id, blob))
                    seen_arts.add(h.artifact_id)
            if attached:
                print(f"{PURPLE}{'[attach]':<{P}}{RESET}{len(attached)} artifacts for synthesis")
        elif goal.attach_artifact_id and artifact_store.exists(goal.attach_artifact_id):
            blob = artifact_store.get_bytes(goal.attach_artifact_id)
            attached.append((goal.attach_artifact_id, blob))
            print(f"{'[attach]':<{P}}{goal.attach_artifact_id} ({len(blob)} bytes)")

        think("decision", f"Deciding: {goal.text[:40]}...")
        out = decision.next_step(goal, hits, attached, history, mcp_tools)
        done()

        if out.is_error:
            print(f"{BLUE}{'[decision]':<{P}}{RESET}{AMBER}(transient error, retrying...){RESET}")
            continue

        if out.is_answer:
            print(f"{BLUE}{'[decision]':<{P}}{RESET}ANSWER: {out.answer[:100]}...")
            history.append({"iter": it, "kind": "answer", "goal_id": goal.id, "text": out.answer})
            # If this was the last unfinished goal, we're done
            unfinished_count = sum(1 for g in obs.goals if not g.done)
            if unfinished_count <= 1:
                print(f"\n{GREEN}[done] all {len(obs.goals)} goals satisfied{RESET}")
                break
            continue

        print(f"{BLUE}{'[decision]':<{P}}{RESET}TOOL_CALL: {out.tool_call.name}({json.dumps(out.tool_call.arguments)[:80]})")
        think("action", f"Calling {out.tool_call.name}...")
        result_text, art_id = await action.execute(session, out.tool_call)
        done()
        memory.record_outcome(
            tool_call=out.tool_call, result_text=result_text,
            artifact_id=art_id, run_id=run_id, goal_id=goal.id,
        )
        history.append({
            "iter": it, "kind": "action", "goal_id": goal.id,
            "tool": out.tool_call.name, "arguments": out.tool_call.arguments,
            "result_descriptor": result_text[:300], "artifact_id": art_id,
        })
        arrow = chr(8594)
        label = f"{BLUE}{'[action]':<{P}}{RESET}"
        if art_id:
            size = len(artifact_store.get_bytes(art_id))
            print(f"{label}{arrow} [artifact {art_id}, {size} bytes stored]")
        else:
            text = result_text.strip()
            if text.startswith("Title:"):
                import re
                titles = re.findall(r'Title:\s*(.+)', text)
                print(f"{label}{arrow} [{len(titles)} results returned, descriptors recorded]")
            elif text.startswith("Created:"):
                print(f"{label}{arrow} {text[:80]}")
            elif text.startswith("[error]") or text.startswith("Fetch error"):
                print(f"{label}{arrow} {text[:80]}")
            else:
                size = len(text)
                print(f"{label}{arrow} [{size} bytes received, descriptor recorded]")
    # Final answer — combine unique answers, skip duplicates
    answers = [e["text"] for e in history if e.get("kind") == "answer"]
    if answers:
        # If only 1 answer or last answer is a summary (longer than prior ones combined), use last
        if len(answers) == 1:
            final = answers[0]
        elif len(answers[-1]) > sum(len(a) for a in answers[:-1]):
            final = answers[-1]
        else:
            # Multiple partial answers (e.g. birth date, death date, contributions) — join
            final = "\n\n".join(answers)
    else:
        fact_hits = memory.read(query, history)
        facts = [f"{h.descriptor}: {json.dumps(h.value, default=str)}" for h in fact_hits if h.kind == "fact"]
        if facts:
            final = "\n".join(facts)
        else:
            actions = [e.get("result_descriptor", "") for e in history if e.get("kind") == "action"]
            final = "\n".join(f"- {a[:200]}" for a in actions) if actions else "No answer produced."

    # Render markdown in terminal
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    console = Console()
    print()
    console.print(Panel(Markdown(final), title="FINAL", border_style="green", padding=(1, 2)))


async def main():
    print_banner()

    # If args provided, run as one-shot
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        await run_query(query)
        return

    # Interactive REPL — keep MCP session alive for speed
    async with mcp_session() as session:
        mcp_tools = await load_tools(session)
        print(f"{DIM}  [{len(mcp_tools)} tools ready]{RESET}\n")

        while True:
            query = read_input()
            if query is None:
                print(f"\n{DIM}Goodbye.{RESET}\n")
                break
            if query.lower() in ("exit", "quit", "/exit", "/quit"):
                print(f"\n{DIM}Goodbye.{RESET}\n")
                break
            if query.lower() in ("/clear", "/reset"):
                import shutil
                shutil.rmtree("state", ignore_errors=True)
                os.makedirs("state/artifacts", exist_ok=True)
                os.makedirs("state/sandbox", exist_ok=True)
                memory._items.clear()
                print(f"  {DIM}State cleared.{RESET}\n")
                continue
            if query.lower() == "/memory":
                items = memory._items
                if not items:
                    print(f"  {DIM}(empty){RESET}\n")
                else:
                    for item in items[-10:]:
                        print(f"  {DIM}[{item.kind}]{RESET} {item.descriptor}")
                print()
                continue

            await run_query(query, session=session, mcp_tools=mcp_tools)


if __name__ == "__main__":
    asyncio.run(main())
