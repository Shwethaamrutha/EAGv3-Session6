"""
LLM Gateway V3 — Bedrock-powered multi-model router.

Uses AWS Bedrock (--profile bedrock) with no rate limit concerns.
Auto-routes by task:
- perception → Claude Haiku 4.5 (fast structured output)
- decision   → Claude Haiku 4.5 (tool-calling)
- memory     → Claude Haiku 4.5 (classification)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import boto3

from config import settings
from logger import get_logger

log = get_logger("gateway")

# Model selection per role (using inference profile IDs)
BEDROCK_MODELS = {
    "perception": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "decision": "us.anthropic.claude-sonnet-4-6",
    "memory": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "default": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
}


@dataclass
class GatewayResponse:
    text: str | None = None
    tool_calls: list[dict] | None = None
    parsed: dict | None = None
    provider: str = ""
    model: str = ""
    tier: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0
    is_error: bool = False
    error_transient: bool = False


@dataclass
class GatewayClient:
    profile: str = field(default_factory=lambda: settings.aws_profile)
    region: str = field(default_factory=lambda: settings.aws_region)

    def __post_init__(self):
        session = boto3.Session(profile_name=self.profile, region_name=self.region)
        self.bedrock = session.client("bedrock-runtime")

    def _get_model(self, auto_route: str | None) -> str:
        if auto_route and auto_route in BEDROCK_MODELS:
            return BEDROCK_MODELS[auto_route]
        return BEDROCK_MODELS["default"]

    def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        response_format: dict | None = None,
        auto_route: str | None = None,
        temperature: float = 1.0,
    ) -> GatewayResponse:
        start = time.time()
        model = self._get_model(auto_route)

        # Build the Bedrock converse request
        bedrock_messages, system_text = self._convert_messages(messages)

        kwargs: dict[str, Any] = {
            "modelId": model,
            "messages": bedrock_messages,
            "inferenceConfig": {
                "temperature": temperature,
                "maxTokens": 768,
            },
        }

        if system_text:
            kwargs["system"] = [{"text": system_text}]

        # Tools
        if tools and not response_format:
            kwargs["toolConfig"] = {
                "tools": [self._convert_tool(t) for t in tools],
            }
            if tool_choice == "auto":
                kwargs["toolConfig"]["toolChoice"] = {"auto": {}}

        # JSON mode via system prompt injection
        if response_format and "schema" in response_format:
            schema_hint = f"\n\nYou MUST respond with valid JSON matching this schema:\n{json.dumps(response_format['schema'], indent=2)}\n\nRespond with ONLY the JSON, no other text."
            if system_text:
                kwargs["system"] = [{"text": system_text + schema_hint}]
            else:
                kwargs["system"] = [{"text": schema_hint}]

        try:
            response = self.bedrock.converse(**kwargs)
        except Exception as e:
            log.error("bedrock_error", model=model, error=str(e))
            return GatewayResponse(
                model=model, is_error=True, error_transient="throttl" in str(e).lower(),
                text=f"[gateway error: {e}]",
            )

        resp = self._parse_response(response, model, response_format)
        resp.provider = "bedrock"
        resp.latency_ms = (time.time() - start) * 1000
        return resp

    def _convert_messages(self, messages: list[dict]) -> tuple[list[dict], str]:
        """Convert OpenAI-style messages to Bedrock converse format."""
        bedrock_msgs = []
        system_text = ""

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_text += content + "\n"
                continue

            bedrock_role = "user" if role == "user" else "assistant"
            bedrock_msgs.append({
                "role": bedrock_role,
                "content": [{"text": content}],
            })

        # Bedrock requires alternating roles — merge consecutive same-role messages
        if bedrock_msgs:
            merged = [bedrock_msgs[0]]
            for msg in bedrock_msgs[1:]:
                if msg["role"] == merged[-1]["role"]:
                    merged[-1]["content"].extend(msg["content"])
                else:
                    merged.append(msg)
            bedrock_msgs = merged

        return bedrock_msgs, system_text.strip()

    def _convert_tool(self, tool: dict) -> dict:
        """Convert OpenAI tool format to Bedrock tool format."""
        params = tool.get("parameters", {})
        # Clean up schema for Bedrock
        params = {k: v for k, v in params.items() if k != "additionalProperties"}

        return {
            "toolSpec": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "inputSchema": {
                    "json": params if params else {"type": "object", "properties": {}},
                },
            }
        }

    def _parse_response(self, response: dict, model: str, response_format: dict | None) -> GatewayResponse:
        """Parse Bedrock converse response."""
        resp = GatewayResponse(model=model)

        output = response.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])

        tool_calls = []
        text_parts = []

        for block in content_blocks:
            if "toolUse" in block:
                tc = block["toolUse"]
                tool_calls.append({
                    "name": tc["name"],
                    "arguments": tc.get("input", {}),
                })
            elif "text" in block:
                text_parts.append(block["text"])

        if tool_calls:
            resp.tool_calls = tool_calls
        if text_parts:
            resp.text = "\n".join(text_parts)

        # Parse JSON if response_format requested
        if response_format and resp.text:
            try:
                resp.parsed = json.loads(resp.text)
            except json.JSONDecodeError:
                # Try to extract JSON from the response
                text = resp.text
                start_idx = text.find("{")
                end_idx = text.rfind("}") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    try:
                        resp.parsed = json.loads(text[start_idx:end_idx])
                    except json.JSONDecodeError:
                        pass

        # Usage
        usage = response.get("usage", {})
        resp.input_tokens = usage.get("inputTokens", 0)
        resp.output_tokens = usage.get("outputTokens", 0)

        return resp


gateway = GatewayClient()
