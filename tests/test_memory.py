"""Tests for memory service — read, dedup, eviction."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from schemas import MemoryItem


def test_memory_read_keyword_match(tmp_memory, sample_memory_items):
    tmp_memory._items = sample_memory_items.copy()
    hits = tmp_memory.read("when is mom's birthday?", [])
    assert len(hits) > 0
    assert any(h.kind == "fact" for h in hits)


def test_memory_read_no_match(tmp_memory, sample_memory_items):
    tmp_memory._items = sample_memory_items.copy()
    hits = tmp_memory.read("quantum physics equations", [])
    assert len(hits) == 0


def test_memory_dedup_rejects_duplicate(tmp_memory):
    item1 = MemoryItem(
        id="a", kind="fact", keywords=["mom", "birthday", "may", "2026"],
        descriptor="mom birthday", value={}, source="test", run_id="r1",
    )
    item2 = MemoryItem(
        id="b", kind="fact", keywords=["mom", "birthday", "may", "2026"],
        descriptor="mom birthday date", value={}, source="test", run_id="r1",
    )
    tmp_memory._items.append(item1)
    assert tmp_memory._is_duplicate(item2) is True


def test_memory_dedup_allows_different(tmp_memory):
    item1 = MemoryItem(
        id="a", kind="fact", keywords=["mom", "birthday"],
        descriptor="mom birthday", value={}, source="test", run_id="r1",
    )
    item2 = MemoryItem(
        id="b", kind="fact", keywords=["dad", "anniversary", "june"],
        descriptor="dad anniversary", value={}, source="test", run_id="r1",
    )
    tmp_memory._items.append(item1)
    assert tmp_memory._is_duplicate(item2) is False


def test_memory_eviction(tmp_memory):
    from config import settings
    original_max = settings.memory_max_items
    settings.memory_max_items = 3

    for i in range(5):
        kind = "scratchpad" if i < 3 else "fact"
        tmp_memory._items.append(MemoryItem(
            id=f"item{i}", kind=kind, keywords=[f"key{i}"],
            descriptor=f"item {i}", value={}, source="test", run_id="r1",
        ))

    tmp_memory._evict_if_needed()
    assert len(tmp_memory._items) <= 3
    # Facts should survive eviction
    facts = [i for i in tmp_memory._items if i.kind == "fact"]
    assert len(facts) == 2

    settings.memory_max_items = original_max


def test_memory_read_with_history_context(tmp_memory, sample_memory_items):
    tmp_memory._items = sample_memory_items.copy()
    history = [{"result_descriptor": "fetch_url wikipedia article", "tool": "fetch_url"}]
    hits = tmp_memory.read("tell me about it", history)
    assert any(h.artifact_id for h in hits)
