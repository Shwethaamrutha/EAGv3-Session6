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

| Role | File | Responsibility | LLM Call? |
|------|------|---------------|-----------|
| **Memory** | `memory.py` | Persist facts, preferences, tool outcomes. Keyword-search reads. | Yes (classify on write) |
| **Perception** | `perception.py` | Decompose query into goals, track completion, decide attachments. | Yes (structured output) |
| **Decision** | `decision.py` | Pick next action for one goal: answer OR one tool call. | Yes (tool-calling) |
| **Action** | `action.py` | Dispatch MCP tool, threshold artifacts, guard handles. | No |

### Supporting Components

| File | Purpose |
|------|---------|
| `schemas.py` | Pydantic models: MemoryItem, Goal, Observation, ToolCall, DecisionOutput |
| `llm_gateway/gateway.py` | Multi-provider router with retry/backoff (Bedrock, NVIDIA, Gemini) |
| `artifacts.py` | Content-addressable store for large tool outputs (>4KB) |
| `config.py` | Centralized settings via pydantic-settings |
| `mcp_server.py` | 9 tools: web_search, fetch_url, get_time, currency_convert, file ops |
| `chat.py` | Interactive CLI REPL (like Claude Code) |
| `chatbot.py` | Web UI with WebSocket streaming |

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

## Target Queries

### Query A — Wikipedia Fetch + Extraction
```
Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his
birth date, death date, and three key contributions to information theory.
```

```
─── iter 1 ───
[memory.read]   0 hits
[perception]    [open] Fetch https://en.wikipedia.org/wiki/Claude_Shannon
                [open] tell me his birth date, death date, and three key contributions
[decision]      TOOL_CALL: fetch_url({"url": "https://en.wikipedia.org/wiki/Claude_Shannon"})
[action]        → [artifact art:067c3fd9, 80201 bytes] preview: # Claude Shannon - Wikipedia

─── iter 2 ───
[memory.read]   1 hits
[perception]    [done] Fetch https://en.wikipedia.org/wiki/Claude_Shannon
                [open] tell me his birth date, death date, and three key contributions
                  attach=art:067c3fd9
[attach]        art:067c3fd9 (80201 bytes)
[decision]      ANSWER: Birth date: April 30, 1916. Death date: February 24, 2001...

─── iter 3 ───
[perception]    [done] Fetch https://en.wikipedia.org/wiki/Claude_Shannon
                [done] tell me his birth date, death date, and three key contributions

[done] all 2 goals satisfied

FINAL: Birth date: April 30, 1916. Death date: February 24, 2001.
       Three contributions: (1) Developed the mathematical theory of communication,
       (2) Introduced the concept of the bit as a unit of information,
       (3) Established the field of information theory.
```

### Query B — Multi-Goal + Weather Constraint
```
Find 3 family-friendly things to do in Tokyo this weekend.
Check Saturday's weather forecast there and tell me which one is most appropriate.
```

```
─── iter 1 ───
[perception]    [open] Find 3 family-friendly things to do in Tokyo this weekend
                [open] Check Saturday's weather forecast in Tokyo
                [open] Tell me which one is most appropriate
[decision]      TOOL_CALL: web_search({"query": "top 3 family-friendly things to do in Tokyo 2026"})
[action]        → [3 results] THE 15 BEST Things to Do in Tokyo (2026); Tokyo for Families; ...

─── iter 2 ───
[perception]    [done] Find 3 family-friendly things to do in Tokyo this weekend
[decision]      TOOL_CALL: fetch_url({"url": "https://wttr.in/Tokyo?format=3"})
[action]        → tokyo: ☁️ +16°C

─── iter 3 ───
[perception]    [done] Find 3 family-friendly things to do in Tokyo this weekend
                [done] Check Saturday's weather forecast in Tokyo
[decision]      ANSWER: From the three options found (Senso-ji Temple, Shinjuku Gyoen,
                teamLab Planets), teamLab Planets is most appropriate because it is
                fully indoors given the cloudy 16°C weather.

[done] all 3 goals satisfied
```

### Query C — Durable Memory Across Runs
```
Run 1: My mom's birthday is 15 May 2026. Remember that and give me
       a calendar reminder for two weeks before and on the day.

Run 2: When is mom's birthday?
```

```
Run 1:
[memory.remember] stored [fact] User's mother's birthday is May 15, 2026

─── iter 1 ───
[perception]    [open] Create a reminder for 1 May 2026 (two weeks before)
                [open] Create a reminder for 15 May 2026 (mom's birthday)
[decision]      TOOL_CALL: create_file({"path": "reminders/mom_birthday.ics", ...})
[action]        → Created: reminders/mom_birthday.ics (842 bytes)

─── iter 2 ───
[perception]    [done] Create a reminder for 1 May 2026
                [done] Create a reminder for 15 May 2026

[done] all 2 goals satisfied

---

Run 2:
─── iter 1 ───
[memory.read]   1 hits
[perception]    [done] When is mom's birthday?
[decision]      ANSWER: Mom's birthday is May 15, 2026.

[done] all 1 goals satisfied
```

### Query D — Multi-Source Synthesis
```
Search for 'Python asyncio best practices', read the top 3 results,
and give me a short numbered list of the advice they agree on.
```

```
─── iter 1 ───
[perception]    [open] Search for 'Python asyncio best practices'
                [open] Fetch and read the top 3 search results
                [open] Extract and list the advice all 3 sources agree on
[decision]      TOOL_CALL: web_search({"query": "Python asyncio best practices"})
[action]        → [3 results] Asyncio best practices; How to Use asyncio; Asyncio Pitfalls

─── iter 2-4 ───
[decision]      TOOL_CALL: fetch_url({"url": "..."})  (×3 different URLs)

─── iter 5 ───
[perception]    [done] Search
                [done] Fetch and read the top 3 search results
                [open] Extract and list the advice all 3 sources agree on
[decision]      ANSWER:
  1. Use asyncio.run() as the main entry point
  2. Never block the event loop — use await asyncio.sleep() or run_in_executor()
  3. Use asyncio.gather() or create_task() for concurrent coroutines

[done] all 3 goals satisfied
```

## Key Design Decisions

- **Typed boundaries**: Every role consumes/produces Pydantic models. No free-form dicts between roles.
- **Artifact store**: Tool results >4KB go to content-addressable storage. Memory holds only the handle.
- **Sticky-done**: Once Perception marks a goal done, it stays done forever.
- **Force-answer on synthesis**: When prior results exist and goal is synthesis, tools are disabled — model must answer from memory.
- **Multi-fetch enforcement**: "Read top N results" goals require N distinct fetch_url calls before marking done.
- **Readability extraction**: Uses `readability-lxml` (Firefox Reader View algorithm) for clean web content.

## Production Features

- **Retry with backoff** (tenacity) on transient LLM errors
- **File-locked memory** (filelock) prevents concurrent corruption
- **Memory dedup + eviction** (max 500 items, scratchpad evicted first)
- **Artifact TTL cleanup** (72h default)
- **Structured logging** (structlog, JSON mode available)
- **29 unit tests** covering all modules
- **Session isolation** in web chatbot
- **Health check endpoint** (`GET /health`)

## License

MIT
