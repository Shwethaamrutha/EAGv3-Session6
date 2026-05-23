# Agent6 — Four-Role Agentic Architecture

A production-hardened AI agent built on typed cognitive roles, persistent memory, and multi-provider LLM routing.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        AGENT LOOP                           │
│                                                             │
│  ┌──────────┐   ┌────────────┐   ┌──────────┐   ┌───────┐ │
│  │  Memory  │──▶│ Perception │──▶│ Decision │──▶│Action │ │
│  │          │   │            │   │          │   │       │ │
│  │ read()   │   │ observe()  │   │next_step()│  │execute│ │
│  │ remember │   │ → Goals[]  │   │→ Answer  │   │→ MCP  │ │
│  │ record() │   │ → Done?    │   │→ ToolCall│   │       │ │
│  └──────────┘   └────────────┘   └──────────┘   └───────┘ │
│       │                                              │      │
│       └──────────── record_outcome ◀─────────────────┘      │
│                                                             │
│  Substrate: LLM Gateway V3 (Bedrock / NVIDIA / Gemini)      │
│  Transport: MCP over stdio                                  │
│  Contracts: Pydantic v2 on every boundary                   │
└─────────────────────────────────────────────────────────────┘
```

### The Four Roles


| Role           | File            | Responsibility                                                    | LLM Call?               |
| -------------- | --------------- | ----------------------------------------------------------------- | ----------------------- |
| **Memory**     | `memory.py`     | Persist facts, preferences, tool outcomes. Keyword-search reads.  | Yes (classify on write) |
| **Perception** | `perception.py` | Decompose query into goals, track completion, decide attachments. | Yes (structured output) |
| **Decision**   | `decision.py`   | Pick next action for one goal: answer OR one tool call.           | Yes (tool-calling)      |
| **Action**     | `action.py`     | Dispatch MCP tool, threshold artifacts, guard handles.            | No                      |


### Supporting Components


| File                     | Purpose                                                                  |
| ------------------------ | ------------------------------------------------------------------------ |
| `schemas.py`             | Pydantic models: MemoryItem, Goal, Observation, ToolCall, DecisionOutput |
| `llm_gateway/gateway.py` | Multi-provider router with retry/backoff (Bedrock, NVIDIA, Gemini)       |
| `artifacts.py`           | Content-addressable store for large tool outputs (>4KB)                  |
| `config.py`              | Centralized settings via pydantic-settings                               |
| `mcp_server.py`          | 9 tools: web_search, fetch_url, get_time, currency_convert, file ops     |
| `chat.py`                | Interactive CLI REPL (like Claude Code)                                  |
| `chatbot.py`             | Web UI with WebSocket streaming                                          |


## Setup

```bash
# Install dependencies
uv sync

# Configure (copy and fill in)
cp .env.example .env

# For AWS Bedrock (recommended — no rate limits):
# Ensure `aws configure --profile bedrock` is set up

# Run interactive chat
uv run python chat.py

# Run single query
uv run python agent6.py "What time is it?"

# Run web chatbot
uv run python chatbot.py
# → Open http://localhost:8000

# Run tests
uv sync --extra dev
uv run pytest tests/ -v
```

## Target Queries (actual terminal output)

> The following is captured from `uv run python chat.py` on a clean state.

### Query A — Wikipedia Fetch + Extraction

![Query A Output](screenshots/Query-1.png)


```
> Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his
birth date, death date, and three key contributions to information
theory... .. 
.. 

─── iter 1 ───
[memory.read]   0 hits
[perception]    [ ] Fetch https://en.wikipedia.org/wiki/Claude_Shannon
                [ ] Extract birth date, death date, and three key contributions to information theory from the fetched page
[decision]      TOOL_CALL: fetch_url({"url": "https://en.wikipedia.org/wiki/Claude_Shannon"})
[action]        → [artifact art:067c3fd99a6a0ae8, 80201 bytes stored]

─── iter 2 ───
[memory.read]   1 hits
[perception]    [✓] Fetch https://en.wikipedia.org/wiki/Claude_Shannon
                [ ] Extract birth date, death date, and three key contributions to information theory from the fetched page
                      attach=art:067c3fd99a6a0ae8
[attach]        1 artifacts for synthesis
[decision]      ANSWER: **Birth date:** April 30, 1916
**Death date:** February 24, 2001

**Three key contributions to infor...

[done] all 2 goals satisfied

╭────────────────────────────────────────────────────────────────────────────────────────────────────── FINAL ───────────────────────────────────────────────────────────────────────────────────────────────────────╮
│                                                                                                                                                                                                                    │
│  Birth date: April 30, 1916 Death date: February 24, 2001                                                                                                                                                          │
│                                                                                                                                                                                                                    │
│  Three key contributions to information theory:                                                                                                                                                                    │
│                                                                                                                                                                                                                    │
│   1 "A Mathematical Theory of Communication" (1948) — Founded the field of information theory by developing information entropy as a measure of information content in a message, establishing the mathematical    │
│     framework for all digital communication.                                                                                                                                                                       │
│   2 Formal introduction of the "bit" — Coined and defined the bit as the fundamental unit of information, underpinning all digital data representation.                                                            │
│   3 Proof of the unbreakability of the one-time pad / "Communication Theory of Secrecy Systems" (1949) — Proved that perfectly secure encryption requires a truly random key as large as the plaintext, and that   │
│     any unbreakable cipher must share these properties, founding modern cryptography alongside information theory.                                                                                                 │
│                                                                                                                                                                                                                    │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
> 

```

### Query B — Multi-Goal + Weather Constraint

![Query B Output](screenshots/Query-2.png)

### Query C — Durable Memory Across Runs

![Query C Output](screenshots/Query-3.png)

### Query D — Multi-Source Synthesis

![Query D Output Part A](screenshots/Query-4-Part-A.png)

![Query D Output Part B](screenshots/Query-4-Part-B.png)

## Key Design Decisions

- **Typed boundaries**: Every role consumes/produces Pydantic models. No free-form dicts between roles.
- **Artifact store**: Tool results >4KB go to content-addressable storage. Memory holds only the handle.
- **Sticky-done**: Once Perception marks a goal done, it stays done forever.
- **Force-answer on synthesis**: When prior results exist and goal is synthesis, tools are disabled — model must answer from memory.
- **Multi-fetch enforcement**: "Read top N results" goals require N distinct fetch_url calls before marking done.
- **Readability extraction**: Uses `readability-lxml` (Firefox Reader View algorithm) for clean web content.



