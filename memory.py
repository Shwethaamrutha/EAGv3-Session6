"""Memory service — typed store with file locking, dedup, and eviction."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

from filelock import FileLock

from config import settings
from llm_gateway import gateway
from logger import get_logger
from schemas import MemoryItem

log = get_logger("memory")

MEMORY_FILE = Path(settings.memory_file)
MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
MEMORY_LOCK = FileLock(str(MEMORY_FILE) + ".lock", timeout=10)

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "because", "if", "or", "and", "but", "that", "this", "it",
    "its", "i", "me", "my", "we", "our", "you", "your", "he", "him",
    "his", "she", "her", "they", "them", "their", "what", "which", "who",
    "whom", "these", "those", "am", "about", "up", "also", "tell",
    "give", "find", "get", "check", "make", "let", "know",
}

CLASSIFY_PROMPT = """You are a memory classifier. Given the user's text, extract a structured memory item.

Respond in JSON with exactly these fields:
- "kind": one of "fact", "preference", "tool_outcome", "scratchpad"
- "keywords": list of 3-8 relevant keywords (lowercase, no stopwords)
- "descriptor": one short human-readable line summarizing the item
- "value": a structured dict capturing the key information

Rules:
- "fact" = a durable observed truth (dates, names, relationships, locations)
- "preference" = a user-stated or inferred preference
- "tool_outcome" = record of a tool dispatch (only used internally)
- "scratchpad" = a temporary working note

Text to classify:
{text}
"""


class MemoryService:
    def __init__(self):
        self._items: list[MemoryItem] = []
        self._load()

    def _load(self):
        with MEMORY_LOCK:
            if MEMORY_FILE.exists() and MEMORY_FILE.stat().st_size > 0:
                try:
                    data = json.loads(MEMORY_FILE.read_text())
                    self._items = [MemoryItem.model_validate(d) for d in data]
                except (json.JSONDecodeError, Exception) as e:
                    log.error("memory_load_failed", error=str(e))
                    self._items = []

    def _save(self):
        self._evict_if_needed()
        with MEMORY_LOCK:
            MEMORY_FILE.write_text(
                json.dumps([item.model_dump(mode="json") for item in self._items], indent=2)
            )

    def _tokenize(self, text: str) -> set[str]:
        tokens = set()
        for word in text.lower().split():
            word = word.strip(".,!?;:'\"()-[]{}/@#$%^&*")
            if word and word not in STOPWORDS and len(word) > 1:
                tokens.add(word)
        return tokens

    def _is_duplicate(self, new_item: MemoryItem) -> bool:
        new_tokens = set(new_item.keywords)
        if not new_tokens:
            return False
        for existing in self._items:
            if existing.kind != new_item.kind:
                continue
            existing_tokens = set(existing.keywords)
            if not existing_tokens:
                continue
            union = new_tokens | existing_tokens
            overlap = len(new_tokens & existing_tokens) / max(len(union), 1)
            if overlap >= settings.memory_dedup_threshold:
                return True
        return False

    def _evict_if_needed(self):
        if len(self._items) <= settings.memory_max_items:
            return
        evictable = [i for i in self._items if i.kind in ("scratchpad", "tool_outcome")]
        evictable.sort(key=lambda x: x.created_at)
        to_remove = len(self._items) - settings.memory_max_items
        removed = 0
        for item in evictable:
            if removed >= to_remove:
                break
            self._items.remove(item)
            removed += 1
        if removed:
            log.info("memory_evicted", count=removed)

    def read(self, query: str, history: list[dict], kinds: list[str] | None = None, top_k: int = 8) -> list[MemoryItem]:
        query_tokens = self._tokenize(query)
        for event in history[-5:]:
            if "result_descriptor" in event:
                query_tokens |= self._tokenize(event["result_descriptor"])
            if "tool" in event:
                query_tokens |= self._tokenize(event["tool"])

        scored: list[tuple[float, MemoryItem]] = []
        for item in self._items:
            if kinds and item.kind not in kinds:
                continue
            item_tokens = set(item.keywords) | self._tokenize(item.descriptor)
            overlap = len(query_tokens & item_tokens)
            if overlap > 0:
                scored.append((overlap, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def filter(self, kinds: list[str] | None = None, goal_id: str | None = None, recent: int | None = None) -> list[MemoryItem]:
        results = self._items
        if kinds:
            results = [i for i in results if i.kind in kinds]
        if goal_id:
            results = [i for i in results if i.goal_id == goal_id]
        if recent:
            results = results[-recent:]
        return results

    def remember(self, raw_text: str, source: str, run_id: str, goal_id: str | None = None) -> MemoryItem | None:
        # Skip queries that don't contain personal facts worth remembering
        lower = raw_text.lower()
        stripped = raw_text.strip().rstrip(".")

        # Questions without personal info
        if stripped.endswith("?") and not any(kw in lower for kw in ["remember", "my ", "i ", "our "]):
            return None

        # Instructional commands (fetch/search/find/tell me) with no personal facts
        action_verbs = ["fetch", "search", "find", "get", "look up", "show me", "tell me", "list", "compare"]
        has_personal = any(kw in lower for kw in ["my ", "i am", "i'm", "our ", "remember", "birthday", "prefer"])
        if any(lower.startswith(v) for v in action_verbs) and not has_personal:
            return None

        prompt = CLASSIFY_PROMPT.format(text=raw_text)
        response_schema = {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["fact", "preference", "tool_outcome", "scratchpad"]},
                "keywords": {"type": "array", "items": {"type": "string"}},
                "descriptor": {"type": "string"},
                "value": {"type": "object"},
            },
            "required": ["kind", "keywords", "descriptor", "value"],
        }

        resp = gateway.chat(
            messages=[{"role": "user", "content": prompt}],
            response_format={"schema": response_schema},
            auto_route="memory",
            temperature=0.3,
        )

        if resp.is_error:
            log.warning("memory_classify_failed", source=source)
            return None

        if resp.parsed:
            data = resp.parsed
            item = MemoryItem(
                id=uuid.uuid4().hex[:12],
                kind=data.get("kind", "scratchpad"),
                keywords=[k.lower() for k in data.get("keywords", [])],
                descriptor=data.get("descriptor", raw_text[:100]),
                value=data.get("value", {"raw": raw_text}),
                source=source,
                run_id=run_id,
                goal_id=goal_id,
                confidence=0.9,
            )
            if self._is_duplicate(item):
                log.debug("memory_dedup_skipped", descriptor=item.descriptor)
                return None
            self._items.append(item)
            self._save()
            log.info("memory_stored", kind=item.kind, descriptor=item.descriptor[:60])
            return item
        return None

    def record_outcome(
        self,
        tool_call,
        result_text: str,
        artifact_id: str | None,
        run_id: str,
        goal_id: str | None,
    ) -> MemoryItem:
        keywords = [tool_call.name.lower()]
        for k, v in tool_call.arguments.items():
            keywords.extend(self._tokenize(str(v)))
        keywords = list(set(keywords))[:10]

        item = MemoryItem(
            id=uuid.uuid4().hex[:12],
            kind="tool_outcome",
            keywords=keywords,
            descriptor=f"{tool_call.name}({json.dumps(tool_call.arguments)[:80]}) → {result_text[:150]}",
            value={
                "tool": tool_call.name,
                "arguments": tool_call.arguments,
                "result_preview": result_text[:2000],
            },
            artifact_id=artifact_id,
            source=f"action:{tool_call.name}",
            run_id=run_id,
            goal_id=goal_id,
            confidence=1.0,
        )
        if self._is_duplicate(item):
            log.debug("memory_dedup_outcome", tool=tool_call.name)
            return item
        self._items.append(item)
        self._save()
        return item


memory = MemoryService()
