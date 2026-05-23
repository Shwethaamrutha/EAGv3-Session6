"""Chatbot server — Web UI for agent6 with session isolation.

Features:
- WebSocket-based real-time streaming
- Per-session state isolation
- Health check endpoint
- Graceful shutdown via lifespan
- Input validation
"""
from __future__ import annotations

import asyncio
import json
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import action
import decision
import perception
from artifacts import artifact_store
from config import settings
from logger import get_logger
from memory import memory
from schemas import Goal

log = get_logger("chatbot")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("chatbot_startup", port=settings.chatbot_port)
    artifact_store.cleanup(max_age_hours=settings.artifact_ttl_hours)
    yield
    log.info("chatbot_shutdown_begin")
    await asyncio.sleep(2)
    log.info("chatbot_shutdown_complete")


app = FastAPI(title="Agent6 Chatbot", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "memory_items": len(memory._items),
        "nvidia_configured": bool(settings.nvidia_api_key),
    }


@asynccontextmanager
async def mcp_session():
    server_params = StdioServerParameters(command="python", args=["mcp_server.py"])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def load_tools(session: ClientSession) -> list[dict]:
    result = await session.list_tools()
    tools = []
    for tool in result.tools:
        tool_def = {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema if tool.inputSchema else {"type": "object", "properties": {}},
        }
        tools.append(tool_def)
    return tools


async def run_agent_streaming(query: str, ws: WebSocket, session_id: str):
    run_id = uuid.uuid4().hex[:8]
    history: list[dict] = []
    prior_goals: list[Goal] = []

    log.info("query_start", session_id=session_id, run_id=run_id, query=query[:80])

    async def send(event_type: str, data: dict):
        await ws.send_json({"type": event_type, **data})

    await send("thinking", {"text": "Classifying query into memory..."})
    memory.remember(query, source="user_query", run_id=run_id)

    async with mcp_session() as session:
        mcp_tools = await load_tools(session)
        await send("status", {"text": f"Connected to {len(mcp_tools)} tools"})

        for it in range(1, settings.max_iterations + 1):
            hits = memory.read(query, history)
            obs = perception.observe(query, hits, history, prior_goals, run_id)
            prior_goals = obs.goals

            goals_info = [{"text": g.text, "status": "done" if g.done else "working"} for g in obs.goals]
            await send("goals", {"goals": goals_info, "iteration": it})

            if obs.all_done:
                has_answer = any(e.get("kind") == "answer" for e in history)
                if not has_answer and hits:
                    summary_goal = obs.goals[-1]
                    out = decision.next_step(summary_goal, hits, [], history, mcp_tools)
                    if out.is_answer:
                        history.append({"iter": it, "kind": "answer", "goal_id": summary_goal.id, "text": out.answer})
                break

            goal = obs.next_unfinished()
            if goal is None:
                break

            attached: list[tuple[str, bytes]] = []
            synthesis_keywords = {"synthesize", "synthesise", "extract", "list", "compare",
                                  "decide", "choose", "summarize", "common", "agree", "advice"}
            goal_tokens = set(goal.text.lower().split())
            is_synthesis = bool(goal_tokens & synthesis_keywords)

            if is_synthesis:
                seen_arts = set()
                for h in hits:
                    if h.artifact_id and h.artifact_id not in seen_arts and artifact_store.exists(h.artifact_id):
                        blob = artifact_store.get_bytes(h.artifact_id)
                        attached.append((h.artifact_id, blob))
                        seen_arts.add(h.artifact_id)
            elif goal.attach_artifact_id and artifact_store.exists(goal.attach_artifact_id):
                blob = artifact_store.get_bytes(goal.attach_artifact_id)
                attached.append((goal.attach_artifact_id, blob))
                await send("status", {"text": f"Reading artifact ({len(blob)} bytes)..."})

            await send("thinking", {"text": f"Deciding next step for: {goal.text[:60]}..."})
            out = decision.next_step(goal, hits, attached, history, mcp_tools)

            if out.is_error:
                await send("status", {"text": "Retrying..."})
                continue

            if out.is_answer:
                history.append({"iter": it, "kind": "answer", "goal_id": goal.id, "text": out.answer})
                await send("progress", {"text": f"Answered: {goal.text}"})
                continue

            await send("tool_call", {"tool": out.tool_call.name, "args": out.tool_call.arguments})
            result_text, art_id = await action.execute(session, out.tool_call)

            memory.record_outcome(
                tool_call=out.tool_call, result_text=result_text, artifact_id=art_id,
                run_id=run_id, goal_id=goal.id,
            )
            history.append({
                "iter": it, "kind": "action", "goal_id": goal.id,
                "tool": out.tool_call.name, "arguments": out.tool_call.arguments,
                "result_descriptor": result_text[:300], "artifact_id": art_id,
            })
            await send("tool_result", {
                "tool": out.tool_call.name, "result": result_text[:200],
                "has_artifact": art_id is not None,
            })

    answers = [e["text"] for e in history if e.get("kind") == "answer"]
    final = "\n\n".join(answers) if answers else "I completed the tasks but have no text answer to show."
    await send("answer", {"text": final})
    log.info("query_complete", session_id=session_id, run_id=run_id)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    session_id = uuid.uuid4().hex[:8]
    log.info("ws_connected", session_id=session_id)
    try:
        while True:
            data = await ws.receive_json()
            if data.get("type") == "query":
                query = data.get("text", "").strip()
                if not query:
                    continue
                if len(query) > settings.max_query_length:
                    await ws.send_json({"type": "error", "text": f"Query too long (max {settings.max_query_length} chars)"})
                    continue
                try:
                    await run_agent_streaming(query, ws, session_id=session_id)
                except Exception as e:
                    log.exception("agent_error", session_id=session_id)
                    await ws.send_json({"type": "error", "text": str(e)})
    except WebSocketDisconnect:
        log.info("ws_disconnected", session_id=session_id)


@app.get("/")
async def index():
    return HTMLResponse(CHAT_HTML)


CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Agent6 — Chat</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
:root {
    --bg: #1a1a2e;
    --surface: #16213e;
    --surface2: #0f3460;
    --accent: #e94560;
    --accent2: #533483;
    --text: #eaeaea;
    --text-dim: #a0a0b0;
    --border: #2a2a4a;
    --user-bg: #533483;
    --agent-bg: #16213e;
    --code-bg: #0d1b2a;
}
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    display: flex;
    flex-direction: column;
}
.header {
    padding: 16px 24px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 12px;
    background: var(--surface);
}
.header h1 {
    font-size: 18px;
    font-weight: 600;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.header .subtitle { font-size: 12px; color: var(--text-dim); }
.chat-container {
    flex: 1; overflow-y: auto; padding: 24px;
    display: flex; flex-direction: column; gap: 16px;
}
.message { max-width: 780px; width: 100%; margin: 0 auto; display: flex; gap: 12px; animation: fadeIn 0.3s ease; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
.message.user { justify-content: flex-end; }
.message.user .bubble { background: var(--user-bg); border-radius: 18px 18px 4px 18px; }
.message.agent .bubble { background: var(--agent-bg); border: 1px solid var(--border); border-radius: 18px 18px 18px 4px; }
.bubble { padding: 14px 18px; max-width: 85%; line-height: 1.6; font-size: 14px; white-space: pre-wrap; word-wrap: break-word; }
.bubble code { background: var(--code-bg); padding: 2px 6px; border-radius: 4px; font-size: 13px; font-family: monospace; }
.avatar { width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 14px; flex-shrink: 0; }
.message.user .avatar { background: var(--accent2); }
.message.agent .avatar { background: var(--accent); }
.status-bar { max-width: 780px; width: 100%; margin: 0 auto; padding: 8px 16px; font-size: 12px; color: var(--text-dim); display: flex; align-items: center; gap: 8px; }
.status-bar .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); animation: pulse 1.5s infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
.goals-panel { max-width: 780px; width: 100%; margin: 0 auto; padding: 10px 16px; background: var(--surface2); border-radius: 10px; font-size: 12px; border: 1px solid var(--border); }
.goals-panel .goal { padding: 4px 0; display: flex; align-items: center; gap: 8px; }
.goals-panel .goal .check { width: 14px; height: 14px; border-radius: 50%; border: 2px solid var(--text-dim); display: flex; align-items: center; justify-content: center; font-size: 9px; }
.goals-panel .goal.done .check { background: #4ecdc4; border-color: #4ecdc4; color: #000; }
.tool-badge { max-width: 780px; width: 100%; margin: 0 auto; padding: 6px 12px; font-size: 11px; color: var(--text-dim); background: var(--code-bg); border-radius: 6px; font-family: monospace; border-left: 3px solid var(--accent2); }
.input-area { padding: 16px 24px; border-top: 1px solid var(--border); background: var(--surface); }
.input-wrapper { max-width: 780px; margin: 0 auto; display: flex; gap: 12px; align-items: flex-end; }
.input-wrapper textarea { flex: 1; background: var(--bg); border: 1px solid var(--border); border-radius: 12px; padding: 12px 16px; color: var(--text); font-size: 14px; font-family: inherit; resize: none; min-height: 44px; max-height: 200px; outline: none; transition: border-color 0.2s; }
.input-wrapper textarea:focus { border-color: var(--accent); }
.input-wrapper button { background: var(--accent); border: none; border-radius: 10px; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; cursor: pointer; transition: transform 0.1s; color: white; font-size: 18px; }
.input-wrapper button:hover { transform: scale(1.05); }
.input-wrapper button:disabled { opacity: 0.4; cursor: not-allowed; }
</style>
</head>
<body>
<div class="header">
    <h1>Agent6</h1>
    <span class="subtitle">Memory &bull; Perception &bull; Decision &bull; Action</span>
</div>
<div class="chat-container" id="chat">
    <div class="message agent">
        <div class="avatar">A6</div>
        <div class="bubble">Hey! I'm Agent6 — an AI assistant with durable memory, web search, and multi-step reasoning. Ask me anything complex!</div>
    </div>
</div>
<div class="input-area">
    <div class="input-wrapper">
        <textarea id="input" placeholder="Ask something... (Shift+Enter for newline)" rows="1"></textarea>
        <button id="send" onclick="sendMessage()">&#x2191;</button>
    </div>
</div>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
let ws = null, isProcessing = false;

function connect() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${location.host}/ws`);
    ws.onmessage = (event) => handleMessage(JSON.parse(event.data));
    ws.onclose = () => setTimeout(connect, 2000);
    ws.onerror = () => {};
}

function handleMessage(msg) {
    switch(msg.type) {
        case 'status': case 'thinking': showStatus(msg.text); break;
        case 'goals': showGoals(msg.goals, msg.iteration); break;
        case 'tool_call': showToolCall(msg.tool, msg.args); break;
        case 'tool_result': showToolResult(msg.tool, msg.result); break;
        case 'progress': showStatus(msg.text); break;
        case 'answer': showAnswer(msg.text); isProcessing = false; sendBtn.disabled = false; removeStatus(); break;
        case 'error': showAnswer('Error: ' + msg.text); isProcessing = false; sendBtn.disabled = false; removeStatus(); break;
    }
}

function sendMessage() {
    const text = input.value.trim();
    if (!text || isProcessing || !ws) return;
    addMessage('user', text);
    input.value = ''; input.style.height = 'auto';
    isProcessing = true; sendBtn.disabled = true;
    ws.send(JSON.stringify({ type: 'query', text }));
}

function addMessage(role, text) {
    const div = document.createElement('div');
    div.className = `message ${role}`;
    const avatar = role === 'user' ? 'You' : 'A6';
    div.innerHTML = `<div class="avatar">${avatar}</div><div class="bubble">${escapeHtml(text)}</div>`;
    chat.appendChild(div); scrollBottom();
}

function showStatus(text) { removeStatus(); const div = document.createElement('div'); div.className = 'status-bar'; div.id = 'current-status'; div.innerHTML = `<span class="dot"></span>${escapeHtml(text)}`; chat.appendChild(div); scrollBottom(); }
function removeStatus() { const el = document.getElementById('current-status'); if (el) el.remove(); }

function showGoals(goals, iteration) {
    const prev = document.getElementById('goals-panel'); if (prev) prev.remove();
    const div = document.createElement('div'); div.className = 'goals-panel'; div.id = 'goals-panel';
    let html = `<div style="margin-bottom:6px;color:var(--text-dim)">Goals (iter ${iteration}):</div>`;
    for (const g of goals) { const cls = g.status === 'done' ? 'goal done' : 'goal'; const check = g.status === 'done' ? '&#10003;' : ''; html += `<div class="${cls}"><span class="check">${check}</span>${escapeHtml(g.text)}</div>`; }
    div.innerHTML = html; chat.appendChild(div); scrollBottom();
}

function showToolCall(tool, args) { const div = document.createElement('div'); div.className = 'tool-badge'; div.textContent = `> ${tool}(${JSON.stringify(args).slice(0, 100)})`; chat.appendChild(div); scrollBottom(); }
function showToolResult(tool, result) { const div = document.createElement('div'); div.className = 'tool-badge'; div.style.borderLeftColor = '#4ecdc4'; div.textContent = `< ${tool}: ${result.slice(0, 120)}`; chat.appendChild(div); scrollBottom(); }
function showAnswer(text) { const gp = document.getElementById('goals-panel'); if (gp) gp.remove(); addMessage('agent', text); }
function escapeHtml(str) { return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function scrollBottom() { chat.scrollTop = chat.scrollHeight; }

input.addEventListener('input', () => { input.style.height = 'auto'; input.style.height = Math.min(input.scrollHeight, 200) + 'px'; });
input.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } });
connect();
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    port = settings.chatbot_port
    log.info("starting_chatbot", port=port)
    uvicorn.run(app, host="0.0.0.0", port=port)
