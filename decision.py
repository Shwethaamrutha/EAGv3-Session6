"""Decision — selects the next action for one bounded goal.

Returns either a final answer (plain text) or a single tool call.
"""
from __future__ import annotations

import json
import re

from config import settings
from llm_gateway import gateway
from logger import get_logger
from schemas import DecisionOutput, Goal, MemoryItem, ToolCall

log = get_logger("decision")

from datetime import date as _date

def _decision_system():
    today = _date.today()
    return f"""You are the Decision module of an agentic system. You receive ONE goal and must take exactly ONE action.

Today's date is {today.isoformat()}. The current year is {today.year}.

You have two options:
1. ANSWER: If you have enough information to satisfy the goal, respond with a CONCISE answer.
   - Be brief and direct. 2-4 sentences max. No filler, no elaboration.
   - Answer EXACTLY what was asked — nothing more.
   - Format: "Birth date: X. Death date: Y. Contributions: (1)... (2)... (3)..."
   - For recommendations: FIRST list exactly the options found in MEMORY HITS snippets by name,
     THEN pick the best. e.g. "From the three options found (Ueno Zoo, Tsukiji sushi class, Tokyo Skytree),
     the sushi class is most appropriate because it is fully indoors."
   - Extract real names from the search snippets in MEMORY HITS. Do NOT invent options.
   - Never give meta-answers like "based on available information" or "here's my analysis".

2. TOOL CALL: If you need external information or must perform an action, call exactly ONE tool.
   - Pick the most appropriate tool from the available tools.
   - NEVER pass artifact handles (strings starting with "art:") as file paths or URLs.

CRITICAL PRIORITY:
- ALWAYS check MEMORY HITS first. If the answer is in memory hits (look at the "value" fields),
  answer immediately from memory. Do NOT call tools to look up what memory already tells you.
- If ATTACHED ARTIFACTS contains content, use it to form your answer.
- Only call a tool if memory hits AND attached artifacts do NOT contain the answer.
- For "find N things" goals: craft a SPECIFIC search query that will return named items.
  e.g. "find 3 family-friendly things in Tokyo" → search "top 3 family-friendly attractions Tokyo names list"
  The search query should ask for a LIST with specific NAMES, not generic pages.
- For weather/time/price checks: use fetch_url with direct data endpoints (e.g. wttr.in for weather).

Rules:
- Do exactly one thing: answer OR call one tool. Never both.
- Do not narrate or explain your reasoning. Just act.
- Be efficient. One tool call should accomplish the goal if possible.
- NEVER say "I cannot answer", "I need more context", or "goal is incomplete".
  You ALWAYS have enough information. Use what's available and give a concrete answer.
- For choosing/recommending: use the search results and tool outcomes in MEMORY HITS.
  Pick a specific option and explain why. Never ask the user for clarification.
- For weather: use fetch_url with "https://wttr.in/CITY?format=3" for quick weather data.
- For calendar reminders: create .ics files (iCalendar format) so they can be imported into Google Calendar/Apple Calendar.
- When answering a synthesis goal: combine info from ALL memory hits to form your answer.
"""

DECISION_USER = """GOAL: {goal_text}

MEMORY HITS:
{hits_text}

ATTACHED ARTIFACTS:
{attached_text}

RECENT HISTORY:
{history_text}

{pending_urls_text}AVAILABLE TOOLS:
{tools_text}

Decide: respond with EITHER an answer OR a single tool call.
If there are PENDING URLs listed above, fetch the NEXT one (not one you've already fetched).
"""


def _format_hits(hits: list[MemoryItem]) -> str:
    if not hits:
        return "(none)"
    lines = []
    for h in hits:
        lines.append(f"  ({h.kind}) {h.descriptor}")
        if h.value:
            val_str = json.dumps(h.value, default=str)[:300]
            lines.append(f"    value: {val_str}")
        if h.kind == "tool_outcome" and h.value.get("result_preview"):
            preview = h.value["result_preview"][:500]
            lines.append(f"    preview: {preview}")
    return "\n".join(lines)


def _format_attached(attached: list[tuple[str, bytes]]) -> str:
    if not attached:
        return "(none)"
    total_budget = settings.attachment_budget_bytes
    per_artifact = total_budget // max(len(attached), 1)
    parts = []
    for i, (art_id, blob) in enumerate(attached):
        text = blob.decode("utf-8", errors="replace")[:per_artifact]
        parts.append(f"--- SOURCE {i+1}: {art_id} ({len(blob)} bytes) ---\n{text}")
    return "\n".join(parts)


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(none)"
    lines = []
    fetched_urls = set()
    for event in history:
        if event.get("kind") == "action" and event.get("tool") == "fetch_url":
            url = event.get("arguments", {}).get("url", "")
            fetched_urls.add(url)

    for event in history[-8:]:
        if event.get("kind") == "action":
            lines.append(f"  TOOL {event['tool']}({json.dumps(event.get('arguments', {}))[:100]}) → {event.get('result_descriptor', '')[:100]}")
        elif event.get("kind") == "answer":
            lines.append(f"  ANSWER: {event.get('text', '')[:100]}")

    if fetched_urls:
        lines.append(f"\n  ALREADY FETCHED URLs (do NOT re-fetch): {list(fetched_urls)}")
    return "\n".join(lines)


def _format_tools(tools: list[dict]) -> str:
    lines = []
    for t in tools:
        params = ""
        if t.get("parameters", {}).get("properties"):
            params = ", ".join(t["parameters"]["properties"].keys())
        lines.append(f"  {t['name']}({params}): {t.get('description', '')[:80]}")
    return "\n".join(lines)


def next_step(
    goal: Goal,
    hits: list[MemoryItem],
    attached: list[tuple[str, bytes]],
    history: list[dict],
    mcp_tools: list[dict],
) -> DecisionOutput:
    # Compute pending URLs: found in search results but not yet fetched
    fetched_urls = {
        e.get("arguments", {}).get("url", "")
        for e in history
        if e.get("kind") == "action" and e.get("tool") == "fetch_url"
    }
    search_urls = []
    for h in hits:
        if h.kind == "tool_outcome" and h.value.get("tool") == "web_search":
            preview = h.value.get("result_preview", "")
            urls = re.findall(r'URL:\s*(https?://[^\s]+)', preview)
            search_urls.extend(urls)
    unfetched = [u for u in search_urls if u not in fetched_urls]

    pending_urls_text = ""
    if unfetched:
        pending_urls_text = f"PENDING URLs TO FETCH (not yet read):\n  " + "\n  ".join(unfetched[:5]) + "\n\n"

    # Determine if this is a FINAL synthesis/recommendation goal (combines multiple sources)
    synthesis_keywords = {"synthesize", "synthesise", "compare", "common", "agree",
                          "decide", "choose", "appropriate", "recommend", "most",
                          "all 3", "all three", "they agree", "advice they"}
    goal_is_synthesis = any(kw in goal.text.lower() for kw in synthesis_keywords)

    # Force answer (no tools) when:
    # 1. Artifacts attached and no pending URLs to fetch
    # 2. This is a FINAL synthesis goal (not intermediate) and sufficient data exists
    is_final_synthesis = goal_is_synthesis and not unfetched
    has_sufficient_data = sum(1 for h in hits if h.kind == "tool_outcome") >= 2

    if (attached and not unfetched) or (is_final_synthesis and has_sufficient_data):
        use_tools = None
        tools_text = "(tools disabled — answer using MEMORY HITS and any ATTACHED ARTIFACTS)"
        pending_urls_text = ""
    else:
        use_tools = mcp_tools
        tools_text = _format_tools(mcp_tools)

    user_msg = DECISION_USER.format(
        goal_text=goal.text,
        hits_text=_format_hits(hits),
        attached_text=_format_attached(attached),
        history_text=_format_history(history),
        tools_text=tools_text,
        pending_urls_text=pending_urls_text,
    )

    resp = gateway.chat(
        messages=[
            {"role": "system", "content": _decision_system()},
            {"role": "user", "content": user_msg},
        ],
        tools=use_tools,
        tool_choice="auto" if use_tools else None,
        auto_route="decision",
        temperature=0.7,
    )

    # Error detection — never return gateway errors as valid answers
    if resp.is_error:
        log.warning("decision_gateway_error", goal=goal.text, transient=resp.error_transient)
        return DecisionOutput(is_error=True)

    if resp.tool_calls:
        tc = resp.tool_calls[0]
        if tc["name"].upper() == "ANSWER":
            answer_text = tc["arguments"].get("answer", "") or tc["arguments"].get("text", "") or str(tc["arguments"])
            return DecisionOutput(answer=answer_text)
        return DecisionOutput(tool_call=ToolCall(name=tc["name"], arguments=tc["arguments"]))

    if resp.text:
        if resp.text.startswith("[gateway error"):
            log.warning("decision_error_in_text", text=resp.text[:100])
            return DecisionOutput(is_error=True)
        return DecisionOutput(answer=resp.text)

    return DecisionOutput(is_error=True)
