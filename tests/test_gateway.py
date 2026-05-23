"""Tests for gateway — Bedrock client, response parsing, error handling."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


def test_gateway_response_error_fields():
    from llm_gateway.gateway import GatewayResponse
    resp = GatewayResponse(is_error=True, error_transient=True, text="[gateway error: 429]")
    assert resp.is_error
    assert resp.error_transient


def test_parse_response_text():
    from llm_gateway.gateway import GatewayClient
    client = GatewayClient.__new__(GatewayClient)

    response = {
        "output": {"message": {"content": [{"text": "Hello world"}]}},
        "usage": {"inputTokens": 10, "outputTokens": 5},
    }
    resp = client._parse_response(response, "test-model", None)
    assert resp.text == "Hello world"
    assert resp.input_tokens == 10


def test_parse_response_tool_call():
    from llm_gateway.gateway import GatewayClient
    client = GatewayClient.__new__(GatewayClient)

    response = {
        "output": {"message": {"content": [
            {"toolUse": {"name": "web_search", "input": {"query": "test"}}}
        ]}},
        "usage": {"inputTokens": 20, "outputTokens": 10},
    }
    resp = client._parse_response(response, "test-model", None)
    assert resp.tool_calls is not None
    assert resp.tool_calls[0]["name"] == "web_search"
    assert resp.tool_calls[0]["arguments"] == {"query": "test"}


def test_parse_response_json_mode():
    from llm_gateway.gateway import GatewayClient
    client = GatewayClient.__new__(GatewayClient)

    response = {
        "output": {"message": {"content": [
            {"text": '{"kind": "fact", "keywords": ["test"]}'}
        ]}},
        "usage": {"inputTokens": 10, "outputTokens": 15},
    }
    resp = client._parse_response(
        response, "test-model",
        response_format={"schema": {"type": "object"}},
    )
    assert resp.parsed is not None
    assert resp.parsed["kind"] == "fact"


def test_get_model_routing():
    from llm_gateway.gateway import GatewayClient
    client = GatewayClient.__new__(GatewayClient)

    assert "haiku" in client._get_model("perception")
    assert "sonnet" in client._get_model("decision")
    assert "haiku" in client._get_model("memory")
    assert "haiku" in client._get_model(None)
