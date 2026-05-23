"""Perception — the orchestrator role.

Decomposes queries into goals, tracks completion, decides artifact attachments.
"""
from __future__ import annotations

import json
import uuid

from llm_gateway import gateway
from logger import get_logger
from schemas import Goal, MemoryItem, Observation

log = get_logger("perception")

from datetime import date as _date
_TODAY = _date.today().isoformat()

PERCEPTION_SYSTEM = f"""You are the Perception module of an agentic system. Today's date is {_TODAY}. Your job is to:

1. DECOMPOSE a user query into bounded goals (short imperative statements).
2. TRACK which goals are done based on the run history.
3. DECIDE whether the next unfinished goal needs raw artifact bytes attached.

Rules:
- If prior_goals is empty, decompose the query into concrete, actionable goals.
- Each goal should be ONE discrete action. Do NOT combine multiple actions into one goal.
  e.g. "reminder for two weeks before AND on the day" → TWO separate goals with explicit dates.
- Compute actual dates when relative dates are given (e.g. "two weeks before May 15" = "May 1").
- Goal text should be specific and actionable with concrete values filled in.
  e.g. "Create a reminder for 1 May 2026 (two weeks before mom's birthday)"
- If a query says "fetch X and tell me Y", decompose into: "Fetch X" and "Tell me Y".
- If prior_goals is provided, preserve the goal list. Only update done flags.
- A goal becomes done when the history contains an action or answer that satisfies it.
- A "find" or "search" goal is done when web_search or fetch_url results exist in history for it.
- A "check" goal (weather, time, price) is done when the relevant tool result is in history.
- An "extract" or "tell me" goal is done when an ANSWER event for that goal is in history.
- Once done, a goal stays done forever.
- Mark goals done AGGRESSIVELY — if relevant info exists in memory/history, the goal is done.
- A "find N things" goal is done after web_search returns results (snippets are enough).
- A "check weather" goal is done after fetch_url to a weather service returns data.
- A "create" goal is done ONLY when a create_file action for THAT SPECIFIC item is in history.
  If 2 separate items need creating, each needs its own create_file action.
- Do NOT require fetching full pages for goals that only need a list or summary.
- For the first unfinished goal, set artifact_index to an integer if it needs bytes from MEMORY HITS.
- artifact_index must reference a valid index from MEMORY HITS that has an artifact_id.
- If no artifact attachment is needed, set artifact_index to -1.
- Preserve goal order. Do not reorder, insert, or drop goals.
- Synthesis goals (synthesize, extract, list, compare, decide, choose) that follow fetch goals
  should have the relevant artifact attached.

Respond in JSON matching the schema provided.
"""

PERCEPTION_USER = """QUERY: {query}

MEMORY HITS:
{hits_text}

HISTORY:
{history_text}

PRIOR GOALS:
{prior_goals_text}

Produce an Observation with the current goal list. For each goal:
- id: keep the same id if updating, or generate a short id for new goals
- text: short imperative description
- done: true/false
- artifact_index: integer index into MEMORY HITS (only for the first unfinished goal, -1 otherwise)
"""


def _format_hits(hits: list[MemoryItem]) -> str:
    if not hits:
        return "(none)"
    lines = []
    for i, h in enumerate(hits):
        art_tag = f" [artifact: {h.artifact_id}]" if h.artifact_id else ""
        lines.append(f"  [{i}] ({h.kind}) {h.descriptor}{art_tag}")
    return "\n".join(lines)


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(none)"
    lines = []
    for event in history[-10:]:
        if event.get("kind") == "action":
            lines.append(f"  iter {event['iter']}: TOOL {event['tool']}({json.dumps(event.get('arguments', {}))[:80]}) → {event.get('result_descriptor', '')[:100]}")
        elif event.get("kind") == "answer":
            lines.append(f"  iter {event['iter']}: ANSWER for goal {event.get('goal_id', '?')}: {event.get('text', '')[:150]}")
    return "\n".join(lines) if lines else "(none)"


def _format_prior_goals(goals: list[Goal]) -> str:
    if not goals:
        return "(none — first iteration, decompose the query)"
    lines = []
    for g in goals:
        status = "DONE" if g.done else "OPEN"
        lines.append(f"  [{status}] {g.id}: {g.text}")
    return "\n".join(lines)


def observe(
    query: str,
    hits: list[MemoryItem],
    history: list[dict],
    prior_goals: list[Goal],
    run_id: str,
) -> Observation:
    hits_text = _format_hits(hits)
    history_text = _format_history(history)
    prior_goals_text = _format_prior_goals(prior_goals)

    user_msg = PERCEPTION_USER.format(
        query=query,
        hits_text=hits_text,
        history_text=history_text,
        prior_goals_text=prior_goals_text,
    )

    response_schema = {
        "type": "object",
        "properties": {
            "goals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "text": {"type": "string"},
                        "done": {"type": "boolean"},
                        "artifact_index": {"type": "integer"},
                    },
                    "required": ["id", "text", "done"],
                },
            }
        },
        "required": ["goals"],
    }

    resp = gateway.chat(
        messages=[
            {"role": "system", "content": PERCEPTION_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        response_format={"schema": response_schema},
        auto_route="perception",
        temperature=0.7,
    )

    if resp.parsed and "goals" in resp.parsed:
        goals = []
        for g_data in resp.parsed["goals"]:
            art_id = None
            art_idx = g_data.get("artifact_index")
            # -1 or missing means no attachment; valid index means attach
            if art_idx is not None and isinstance(art_idx, int) and art_idx >= 0 and art_idx < len(hits):
                art_id = hits[art_idx].artifact_id

            goal_id = g_data.get("id", uuid.uuid4().hex[:8])
            goals.append(Goal(
                id=goal_id,
                text=g_data["text"],
                done=g_data.get("done", False),
                attach_artifact_id=art_id,
            ))

        # Enforce sticky-done: if a prior goal was done, keep it done
        if prior_goals:
            prior_done_ids = {g.id for g in prior_goals if g.done}
            for g in goals:
                if g.id in prior_done_ids:
                    g.done = True

        # Enforce multi-fetch goals: if a goal says "read/fetch top N",
        # don't mark done until N distinct fetch_url calls exist in history
        import re
        for g in goals:
            if g.done:
                match = re.search(r'(?:read|fetch|get|visit)\s+(?:the\s+)?(?:top\s+)?(\d+)', g.text.lower())
                if match:
                    required_count = int(match.group(1))
                    fetch_actions = [
                        e for e in history
                        if e.get("kind") == "action" and e.get("tool") == "fetch_url"
                    ]
                    distinct_urls = len(set(e.get("arguments", {}).get("url", "") for e in fetch_actions))
                    if distinct_urls < required_count:
                        g.done = False

        # Force-attach for final synthesis/extraction goals only
        # NOT for search/find/check goals which just need tool calls
        if goals:
            next_unfinished = None
            for g in goals:
                if not g.done:
                    next_unfinished = g
                    break
            if next_unfinished and not next_unfinished.attach_artifact_id:
                # Only attach for goals that need to READ content (not find/search/check)
                synthesis_keywords = {"synthesize", "synthesise", "extract", "compare",
                                      "summarize", "common", "agree", "appropriate",
                                      "which one", "recommend", "decide", "choose"}
                goal_lower = next_unfinished.text.lower()
                skip_keywords = {"find", "search", "fetch", "check", "get weather"}
                if any(kw in goal_lower for kw in synthesis_keywords) and not any(kw in goal_lower for kw in skip_keywords):
                    for h in hits:
                        if h.artifact_id:
                            next_unfinished.attach_artifact_id = h.artifact_id
                            break

        return Observation(goals=goals)

    # Fallback: single goal from the query
    log.warning("perception_fallback", reason="LLM response unparseable")
    return Observation(goals=[Goal(id=uuid.uuid4().hex[:8], text=query, done=False)])
