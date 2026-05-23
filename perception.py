"""Perception — the orchestrator role.

Decomposes queries into goals, tracks completion, decides artifact attachments.
"""
from __future__ import annotations

import json
import uuid

from llm_gateway import gateway
from logger import get_logger
from schemas import Goal, MemoryItem, Observation, SYNTHESIS_KEYWORDS

log = get_logger("perception")

from datetime import date as _date

def _perception_system():
    today = _date.today()
    weekday = today.strftime("%A")
    return f"""You are the Perception module. Today is {weekday}, {today.isoformat()}.

Your job:
1. DECOMPOSE a query into ordered goals (fewest possible — group related items).
2. TRACK completion: mark a goal done when history shows it's satisfied.
3. ATTACH: set artifact_index to a MEMORY HITS index if the next goal needs fetched content, else -1.

Rules:
- Separate goals only when they need different actions (fetch vs answer vs create).
- Resolve relative dates/times to absolute values.
- Once done, a goal stays done. Preserve goal order.
- If prior_goals provided, only update done flags — do not add or remove goals.
- Respond in JSON matching the provided schema.
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
            {"role": "system", "content": _perception_system()},
            {"role": "user", "content": user_msg},
        ],
        response_format={"schema": response_schema},
        auto_route="perception",
        temperature=0.7,
    )

    if resp.parsed and "goals" in resp.parsed and resp.parsed["goals"]:
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
                goal_lower = next_unfinished.text.lower()
                skip_keywords = {"find", "search", "fetch", "check", "get weather"}
                if any(kw in goal_lower for kw in SYNTHESIS_KEYWORDS) and not any(kw in goal_lower for kw in skip_keywords):
                    for h in hits:
                        if h.artifact_id:
                            next_unfinished.attach_artifact_id = h.artifact_id
                            break

        return Observation(goals=goals)

    # Fallback: single goal from the query
    log.warning("perception_fallback", reason="LLM response unparseable")
    return Observation(goals=[Goal(id=uuid.uuid4().hex[:8], text=query, done=False)])
