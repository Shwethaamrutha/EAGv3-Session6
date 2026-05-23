"""
LLM Gateway V3 — Multi-provider router with failover.

Provider priority: NVIDIA (free) → Gemini (free) → Bedrock Sonnet (paid, last resort)

Auto-routes by task:
- perception → NVIDIA llama-3.3-70b (structured JSON)
- decision   → NVIDIA llama-3.3-70b (tool-calling)
- memory     → NVIDIA llama-3.3-70b (classification)
- fallback   → Bedrock Claude Sonnet 4.6 (if NVIDIA fails)
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from config import settings
from logger import get_logger

log = get_logger("gateway")

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")
BEDROCK_MODEL = "us.anthropic.claude-sonnet-4-6"


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
    nvidia_key: str = field(default_factory=lambda: os.getenv("NVIDIA_API_KEY", ""))
    aws_profile: str = field(default_factory=lambda: settings.aws_profile)
    aws_region: str = field(default_factory=lambda: settings.aws_region)

    def __post_init__(self):
        # NVIDIA (primary)
        self.nvidia_client = None
        if self.nvidia_key:
            import openai
            self.nvidia_client = openai.OpenAI(base_url=NVIDIA_BASE_URL, api_key=self.nvidia_key)

        # Bedrock (fallback)
        self.bedrock = None
        try:
            import boto3
            session = boto3.Session(profile_name=self.aws_profile, region_name=self.aws_region)
            self.bedrock = session.client("bedrock-runtime")
        except Exception:
            pass

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

        # Try NVIDIA first
        if self.nvidia_client:
            resp = self._call_nvidia(messages, tools, tool_choice, response_format, temperature)
            if not resp.is_error:
                resp.latency_ms = (time.time() - start) * 1000
                self._trace(auto_route, messages, tools, resp)
                return resp
            log.info("nvidia_failed_trying_bedrock", error=resp.text[:80] if resp.text else "")

        # Fallback to Bedrock
        if self.bedrock:
            resp = self._call_bedrock(messages, tools, tool_choice, response_format, temperature)
            resp.latency_ms = (time.time() - start) * 1000
            self._trace(auto_route, messages, tools, resp)
            return resp

        return GatewayResponse(is_error=True, text="[gateway error: no providers configured]")

    def _call_nvidia(
        self, messages, tools, tool_choice, response_format, temperature
    ) -> GatewayResponse:
        import openai

        kwargs: dict[str, Any] = {
            "model": NVIDIA_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 512,
        }

        if tools and not response_format:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

        if response_format and "schema" in response_format:
            kwargs["response_format"] = {"type": "json_object"}
            schema_hint = f"\n\nYou MUST respond with valid JSON matching this schema:\n{json.dumps(response_format['schema'], indent=2)}\n\nRespond with ONLY the JSON, no other text."
            msgs = [m.copy() for m in messages]
            if msgs and msgs[-1]["role"] == "user":
                msgs[-1]["content"] += schema_hint
            else:
                msgs.append({"role": "user", "content": schema_hint})
            kwargs["messages"] = msgs

        try:
            response = self.nvidia_client.chat.completions.create(**kwargs)
        except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError) as e:
            return GatewayResponse(model=NVIDIA_MODEL, is_error=True, error_transient=True,
                                   text=f"[gateway error: NVIDIA: {e}]")
        except Exception as e:
            return GatewayResponse(model=NVIDIA_MODEL, is_error=True,
                                   text=f"[gateway error: NVIDIA: {e}]")

        resp = GatewayResponse(model=NVIDIA_MODEL, provider="nvidia")
        choice = response.choices[0]

        if choice.message.tool_calls:
            resp.tool_calls = []
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                resp.tool_calls.append({"name": tc.function.name, "arguments": args})
        elif choice.message.content:
            resp.text = choice.message.content
            if response_format:
                try:
                    resp.parsed = json.loads(resp.text)
                except json.JSONDecodeError:
                    text = resp.text
                    start_idx = text.find("{")
                    end_idx = text.rfind("}") + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        try:
                            resp.parsed = json.loads(text[start_idx:end_idx])
                        except json.JSONDecodeError:
                            pass

        if response.usage:
            resp.input_tokens = response.usage.prompt_tokens
            resp.output_tokens = response.usage.completion_tokens

        return resp

    def _call_bedrock(
        self, messages, tools, tool_choice, response_format, temperature
    ) -> GatewayResponse:
        bedrock_messages, system_text = self._convert_messages(messages)

        kwargs: dict[str, Any] = {
            "modelId": BEDROCK_MODEL,
            "messages": bedrock_messages,
            "inferenceConfig": {"temperature": temperature, "maxTokens": 512},
        }

        if system_text:
            kwargs["system"] = [{"text": system_text}]

        if tools and not response_format:
            kwargs["toolConfig"] = {
                "tools": [self._convert_tool(t) for t in tools],
            }
            if tool_choice == "auto":
                kwargs["toolConfig"]["toolChoice"] = {"auto": {}}

        if response_format and "schema" in response_format:
            schema_hint = f"\n\nYou MUST respond with valid JSON matching this schema:\n{json.dumps(response_format['schema'], indent=2)}\n\nRespond with ONLY the JSON, no other text."
            if system_text:
                kwargs["system"] = [{"text": system_text + schema_hint}]
            else:
                kwargs["system"] = [{"text": schema_hint}]

        try:
            response = self.bedrock.converse(**kwargs)
        except Exception as e:
            log.error("bedrock_error", model=BEDROCK_MODEL, error=str(e))
            return GatewayResponse(model=BEDROCK_MODEL, is_error=True,
                                   error_transient="throttl" in str(e).lower(),
                                   text=f"[gateway error: Bedrock: {e}]")

        return self._parse_bedrock_response(response, response_format)

    def _parse_bedrock_response(self, response: dict, response_format: dict | None) -> GatewayResponse:
        resp = GatewayResponse(model=BEDROCK_MODEL, provider="bedrock")

        output = response.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])

        tool_calls = []
        text_parts = []

        for block in content_blocks:
            if "toolUse" in block:
                tc = block["toolUse"]
                tool_calls.append({"name": tc["name"], "arguments": tc.get("input", {})})
            elif "text" in block:
                text_parts.append(block["text"])

        if tool_calls:
            resp.tool_calls = tool_calls
        if text_parts:
            resp.text = "\n".join(text_parts)

        if response_format and resp.text:
            try:
                resp.parsed = json.loads(resp.text)
            except json.JSONDecodeError:
                text = resp.text
                start_idx = text.find("{")
                end_idx = text.rfind("}") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    try:
                        resp.parsed = json.loads(text[start_idx:end_idx])
                    except json.JSONDecodeError:
                        pass

        usage = response.get("usage", {})
        resp.input_tokens = usage.get("inputTokens", 0)
        resp.output_tokens = usage.get("outputTokens", 0)

        return resp

    def _convert_messages(self, messages: list[dict]) -> tuple[list[dict], str]:
        bedrock_msgs = []
        system_text = ""

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_text += content + "\n"
                continue
            bedrock_role = "user" if role == "user" else "assistant"
            bedrock_msgs.append({"role": bedrock_role, "content": [{"text": content}]})

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
        params = tool.get("parameters", {})
        params = {k: v for k, v in params.items() if k != "additionalProperties"}
        return {
            "toolSpec": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "inputSchema": {"json": params if params else {"type": "object", "properties": {}}},
            }
        }

    def _trace(self, auto_route, messages, tools, resp):
        try:
            from tracer import trace_llm_call
            trace_llm_call(
                role=auto_route or "default", model=resp.model, messages=messages,
                tools=tools, response_text=resp.text, tool_calls=resp.tool_calls,
                is_error=resp.is_error, latency_ms=resp.latency_ms,
                tokens_in=resp.input_tokens, tokens_out=resp.output_tokens,
            )
        except Exception:
            pass


gateway = GatewayClient()
