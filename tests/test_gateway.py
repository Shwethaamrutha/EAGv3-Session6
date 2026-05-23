"""Tests for gateway — retry logic, rate limiting, error classification."""
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import openai
import pytest


def test_rate_limiter_enforces_minimum_interval():
    from llm_gateway.gateway import RateLimiter
    rl = RateLimiter(calls_per_minute=60)
    rl._min_interval = 0.1

    start = time.time()
    rl.wait_if_needed()
    rl.wait_if_needed()
    elapsed = time.time() - start
    assert elapsed >= 0.1


def test_gateway_response_error_fields():
    from llm_gateway.gateway import GatewayResponse
    resp = GatewayResponse(is_error=True, error_transient=True, text="[gateway error: 429]")
    assert resp.is_error
    assert resp.error_transient


def test_gateway_nvidia_not_configured():
    from llm_gateway.gateway import GatewayClient
    client = GatewayClient.__new__(GatewayClient)
    client.nvidia_client = None
    client.gemini_client = None
    client._rate_limiter = MagicMock()

    resp = client._call_nvidia(
        messages=[{"role": "user", "content": "hi"}],
        tools=None, tool_choice=None, response_format=None,
        temperature=0.5, model="test-model",
    )
    assert resp.is_error
    assert "not configured" in resp.text


def test_parse_openai_response_text():
    from llm_gateway.gateway import GatewayClient

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.tool_calls = None
    mock_response.choices[0].message.content = "Hello world"
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

    client = GatewayClient.__new__(GatewayClient)
    resp = client._parse_openai_response(mock_response, "test-model", None)
    assert resp.text == "Hello world"
    assert resp.input_tokens == 10


def test_parse_openai_response_tool_call():
    from llm_gateway.gateway import GatewayClient

    mock_tc = MagicMock()
    mock_tc.function.name = "web_search"
    mock_tc.function.arguments = '{"query": "test"}'

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.tool_calls = [mock_tc]
    mock_response.choices[0].message.content = None
    mock_response.usage = MagicMock(prompt_tokens=20, completion_tokens=10)

    client = GatewayClient.__new__(GatewayClient)
    resp = client._parse_openai_response(mock_response, "test-model", None)
    assert resp.tool_calls is not None
    assert resp.tool_calls[0]["name"] == "web_search"


def test_parse_openai_response_json_mode():
    from llm_gateway.gateway import GatewayClient

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.tool_calls = None
    mock_response.choices[0].message.content = '{"kind": "fact", "keywords": ["test"]}'
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=15)

    client = GatewayClient.__new__(GatewayClient)
    resp = client._parse_openai_response(
        mock_response, "test-model",
        response_format={"schema": {"type": "object"}},
    )
    assert resp.parsed is not None
    assert resp.parsed["kind"] == "fact"
