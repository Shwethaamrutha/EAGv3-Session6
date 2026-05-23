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

> The following is captured from `uv run python demo.py` on a clean state. See `demo-output.txt` for the full unedited run.

### Query A — Wikipedia Fetch + Extraction (4 iterations)

```
─── iter 1 ───
[memory.read]   0 hits
[perception]    [open] Fetch https://en.wikipedia.org/wiki/Claude_Shannon
                [open] Extract Claude Shannon's birth date from the Wikipedia page
                [open] Extract Claude Shannon's death date from the Wikipedia page
                [open] Extract three key contributions to information theory
[decision]      TOOL_CALL: fetch_url({"url": "https://en.wikipedia.org/wiki/Claude_Shannon"})
[action]        → [artifact art:067c3fd99a6a0ae8, 80201 bytes]

─── iter 2 ───
[perception]    [done] Fetch https://en.wikipedia.org/wiki/Claude_Shannon
                [open] Extract Claude Shannon's birth date
                  attach=art:067c3fd99a6a0ae8
[decision]      ANSWER: Claude Shannon's birth date was April 30, 1916.

─── iter 3 ───
[perception]    [done] Extract Claude Shannon's birth date
                [open] Extract Claude Shannon's death date
                  attach=art:067c3fd99a6a0ae8
[decision]      ANSWER: Claude Shannon died on February 24, 2001, at age 84.

─── iter 4 ───
[perception]    [done] Extract Claude Shannon's death date
                [open] Extract three key contributions to information theory
                  attach=art:067c3fd99a6a0ae8
[decision]      ANSWER: (1) "A Mathematical Theory of Communication" (1948)...

[done] all 4 goals satisfied

FINAL: Birth date: April 30, 1916. Death date: February 24, 2001.
       (1) "A Mathematical Theory of Communication" — founded information theory
       (2) Introduction of the "bit" as the fundamental unit of information
       (3) Nyquist–Shannon Sampling Theorem — enabled analog-to-digital transition
```

### Query B — Multi-Goal + Weather Constraint (9 iterations)

```
─── iter 1 ───
[perception]    [open] Find 3 family-friendly activities in Tokyo
                [open] Check Saturday's weather forecast for Tokyo (2026-05-24)
                [open] Evaluate which is most appropriate based on weather
[decision]      TOOL_CALL: web_search({"query": "top 3 family-friendly attractions Tokyo"})
[action]        → [3 results] The best Tokyo Family-friendly activities 2026; ...

─── iter 2 ───
[perception]    [done] Find 3 family-friendly activities in Tokyo
[decision]      TOOL_CALL: fetch_url({"url": "https://wttr.in/Tokyo?format=3"})
[action]        → tokyo: 🌤️ +15°C

─── iter 3-8 ───
(fetches activity details from multiple sources, some return 403)

─── iter 9 ───
[perception]    [done] Find 3 family-friendly activities
                [done] Check Saturday's weather forecast
                [done] Extract detailed activity information
[decision]      ANSWER: From the three options found (Shinjuku Sumo Show, Sushi Making
                in Ginza, Samurai Ninja Museum Asakusa), the Tokyo Sushi Making
                Experience in Ginza is most appropriate — fully indoors, weather-proof,
                hands-on and engaging for families at +15°C partly cloudy.

[done] all 4 goals satisfied
```

### Query C — Durable Memory Across Runs (Run 1: 3 iters, Run 2: 1 iter)

```
Run 1:
[memory.remember] stored [fact] Mom's birthday is May 15, 2026

─── iter 1 ───
[perception]    [done] Store mom's birthday in memory
                [open] Set calendar reminder for May 1, 2026 (two weeks before)
                [open] Set calendar reminder for May 15, 2026 (birthday day)
[decision]      TOOL_CALL: create_file({"path": "mom_birthday_reminder_may1.ics", ...})
[action]        → Created: mom_birthday_reminder_may1.ics (482 bytes)

─── iter 2 ───
[perception]    [done] Set calendar reminder for May 1, 2026
                [open] Set calendar reminder for May 15, 2026
[decision]      TOOL_CALL: create_file({"path": "mom_birthday_day_may15.ics", ...})
[action]        → Created: mom_birthday_day_may15.ics (398 bytes)

─── iter 3 ───
[perception]    [done] all reminders created
[decision]      ANSWER: Both .ics files created — importable to Google/Apple Calendar.

[done] all 3 goals satisfied

---

Run 2:
─── iter 1 ───
[memory.read]   1 hits
[perception]    [done] Find mom's birthday date
[decision]      ANSWER: Mom's birthday is May 15, 2026.

[done] all 1 goals satisfied
```

### Query D — Multi-Source Synthesis (4 iterations)

```
─── iter 1 ───
[perception]    [done] Search for 'Python asyncio best practices' (from prior context)
                [open] Extract and synthesize common advice into a numbered list
                  attach=art:2f0b354e6a8ca72e
[decision]      TOOL_CALL: fetch_url (×3 different URLs across iters 1-3)

─── iter 2-3 ───
[decision]      TOOL_CALL: fetch_url("https://discuss.python.org/...")
[decision]      TOOL_CALL: fetch_url("https://www.shanechang.com/...")

─── iter 4 ───
[perception]    [done] All 3 results fetched and read
                [open] Extract and synthesize common advice
[attach]        2 artifacts for synthesis
[decision]      ANSWER:
  1. Use asyncio.run() as the main entry point
  2. Never block the event loop — use await asyncio.sleep() or run_in_executor()
  3. Use async/await consistently — don't mix sync and async
  4. Run concurrent tasks with asyncio.gather() or create_task()
  5. Avoid fire-and-forget tasks without tracking references
  6. Handle cancellation and shutdown gracefully
  7. Use asyncio-compatible libraries for I/O

[done] all 5 goals satisfied
```

> Full unedited output: see `demo-output.txt`

## Key Design Decisions

- **Typed boundaries**: Every role consumes/produces Pydantic models. No free-form dicts between roles.
- **Artifact store**: Tool results >4KB go to content-addressable storage. Memory holds only the handle.
- **Sticky-done**: Once Perception marks a goal done, it stays done forever.
- **Force-answer on synthesis**: When prior results exist and goal is synthesis, tools are disabled — model must answer from memory.
- **Multi-fetch enforcement**: "Read top N results" goals require N distinct fetch_url calls before marking done.
- **Readability extraction**: Uses `readability-lxml` (Firefox Reader View algorithm) for clean web content.



