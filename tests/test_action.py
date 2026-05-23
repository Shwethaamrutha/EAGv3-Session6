"""Tests for action module — artifact threshold, handle guard."""
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from schemas import ToolCall


@pytest.mark.asyncio
async def test_action_rejects_artifact_handles():
    from action import execute
    tc = ToolCall(name="fetch_url", arguments={"url": "art:abc123"})
    session = MagicMock()
    result_text, art_id = await execute(session, tc)
    assert "[error]" in result_text
    assert "artifact handles" in result_text
    assert art_id is None


@pytest.mark.asyncio
async def test_action_small_result_no_artifact():
    from action import execute
    tc = ToolCall(name="get_time", arguments={"timezone": "UTC"})

    mock_result = MagicMock()
    mock_block = MagicMock()
    mock_block.text = "Current time: 2026-05-22 12:00:00"
    mock_result.content = [mock_block]

    session = AsyncMock()
    session.call_tool.return_value = mock_result

    result_text, art_id = await execute(session, tc)
    assert "2026" in result_text
    assert art_id is None


@pytest.mark.asyncio
async def test_action_large_result_creates_artifact():
    from action import execute
    tc = ToolCall(name="fetch_url", arguments={"url": "https://example.com"})

    large_content = "x" * 10000
    mock_result = MagicMock()
    mock_block = MagicMock()
    mock_block.text = large_content
    mock_result.content = [mock_block]

    session = AsyncMock()
    session.call_tool.return_value = mock_result

    with patch("action.artifact_store") as mock_store:
        mock_store.put.return_value = "art:test123"
        result_text, art_id = await execute(session, tc)
        assert art_id == "art:test123"
        assert "[artifact" in result_text
        mock_store.put.assert_called_once()


@pytest.mark.asyncio
async def test_action_handles_tool_exception():
    from action import execute
    tc = ToolCall(name="broken_tool", arguments={})

    session = AsyncMock()
    session.call_tool.side_effect = Exception("connection refused")

    result_text, art_id = await execute(session, tc)
    assert "[error]" in result_text
    assert art_id is None
