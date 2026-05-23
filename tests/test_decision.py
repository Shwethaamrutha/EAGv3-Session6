"""Tests for decision module — error detection, output parsing."""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_gateway.gateway import GatewayResponse
from schemas import DecisionOutput, Goal, MemoryItem


def test_decision_error_not_stored_as_answer(mock_gateway_response):
    with patch("decision.gateway") as mock_gw:
        mock_gw.chat.return_value = mock_gateway_response(
            text="[gateway error: NVIDIA: 429]", is_error=True
        )
        from decision import next_step
        goal = Goal(id="g1", text="find something")
        out = next_step(goal, [], [], [], [])
        assert out.is_error is True
        assert out.is_answer is False


def test_decision_valid_answer(mock_gateway_response):
    with patch("decision.gateway") as mock_gw:
        mock_gw.chat.return_value = mock_gateway_response(
            text="The answer is 42. This is because of deep thought. It took 7.5 million years."
        )
        from decision import next_step
        goal = Goal(id="g1", text="what is the meaning of life")
        out = next_step(goal, [], [], [], [])
        assert out.is_answer is True
        assert "42" in out.answer


def test_decision_tool_call_parsed(mock_gateway_response):
    with patch("decision.gateway") as mock_gw:
        mock_gw.chat.return_value = mock_gateway_response(
            tool_calls=[{"name": "web_search", "arguments": {"query": "test"}}]
        )
        from decision import next_step
        goal = Goal(id="g1", text="search for info")
        out = next_step(goal, [], [], [], [{"name": "web_search", "description": "search", "parameters": {}}])
        assert out.tool_call is not None
        assert out.tool_call.name == "web_search"


def test_decision_no_output_is_error(mock_gateway_response):
    with patch("decision.gateway") as mock_gw:
        mock_gw.chat.return_value = mock_gateway_response(text=None, tool_calls=None)
        from decision import next_step
        goal = Goal(id="g1", text="do something")
        out = next_step(goal, [], [], [], [])
        assert out.is_error is True


def test_decision_disables_tools_when_artifact_attached(mock_gateway_response):
    with patch("decision.gateway") as mock_gw:
        mock_gw.chat.return_value = mock_gateway_response(text="Extracted info from the page.")
        from decision import next_step
        goal = Goal(id="g1", text="extract info")
        attached = [("art:abc", b"Some page content here for extraction")]
        out = next_step(goal, [], attached, [], [{"name": "fetch_url", "description": "fetch", "parameters": {}}])
        # Should have called gateway without tools
        call_kwargs = mock_gw.chat.call_args
        assert call_kwargs.kwargs.get("tools") is None
