"""Shared test fixtures — mock gateway, temp memory, etc."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def mock_gateway_response():
    """Factory for mock GatewayResponse objects."""
    from llm_gateway.gateway import GatewayResponse

    def _make(text=None, tool_calls=None, parsed=None, is_error=False):
        return GatewayResponse(
            text=text, tool_calls=tool_calls, parsed=parsed,
            provider="n", model="test-model", is_error=is_error,
        )
    return _make


@pytest.fixture
def tmp_memory(tmp_path):
    """Memory service using a temporary file."""
    mem_file = tmp_path / "memory.json"
    with patch("memory.MEMORY_FILE", mem_file), \
         patch("memory.MEMORY_LOCK", MagicMock()):
        from memory import MemoryService
        svc = MemoryService()
        yield svc


@pytest.fixture
def sample_memory_items():
    """Pre-built memory items for testing."""
    from schemas import MemoryItem
    return [
        MemoryItem(
            id="item1", kind="fact", keywords=["mom", "birthday", "may"],
            descriptor="mom's birthday", value={"name": "mom", "birthday": "15 May 2026"},
            source="user_query", run_id="test1",
        ),
        MemoryItem(
            id="item2", kind="tool_outcome", keywords=["fetch_url", "wikipedia"],
            descriptor="fetch_url(wikipedia) → article content",
            value={"tool": "fetch_url", "arguments": {"url": "https://en.wikipedia.org/wiki/Test"}, "result_preview": "Test article content..."},
            artifact_id="art:abc123", source="action:fetch_url", run_id="test1",
        ),
        MemoryItem(
            id="item3", kind="preference", keywords=["morning", "meetings"],
            descriptor="user prefers morning meetings",
            value={"preference": "morning meetings"},
            source="user_query", run_id="test2",
        ),
    ]
