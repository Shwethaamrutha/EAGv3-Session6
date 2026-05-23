"""MCP Server with 9 tools for agent6.

Tools: web_search, fetch_url, get_time, currency_convert,
       read_file, list_dir, create_file, update_file, edit_file

Heavy deps (crawl4ai, duckduckgo-search) are lazy-imported inside tool functions
to keep subprocess startup fast (~1s instead of ~8s).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

if os.getenv("MCP_LOG_LEVEL") == "error":
    logging.disable(logging.CRITICAL)

load_dotenv()

mcp = FastMCP("agent6-tools")

SANDBOX_DIR = Path("state/sandbox")
SANDBOX_DIR.mkdir(parents=True, exist_ok=True)


@mcp.tool()
async def web_search(query: str, max_results: int = 3) -> str:
    """Search the web. Returns top results with titles, URLs, and content snippets."""
    max_results = int(max_results)

    # Primary: Tavily (best snippets, AI-optimized)
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if tavily_key:
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=tavily_key)
            response = client.search(query, max_results=max_results)
            results = response.get("results", [])
            if results:
                lines = []
                for r in results:
                    lines.append(f"Title: {r.get('title', '')}")
                    lines.append(f"URL: {r.get('url', '')}")
                    lines.append(f"Snippet: {r.get('content', '')[:300]}")
                    lines.append("")
                return "\n".join(lines)
        except Exception:
            pass

    # Fallback: DuckDuckGo
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if results:
            lines = []
            for r in results:
                lines.append(f"Title: {r['title']}")
                lines.append(f"URL: {r['href']}")
                lines.append(f"Snippet: {r['body']}")
                lines.append("")
            return "\n".join(lines)
    except Exception:
        pass

    return "No results found. Search services may be rate-limited."


@mcp.tool()
async def fetch_url(url: str) -> str:
    """Fetch a URL and return its content as cleaned markdown using Crawl4AI."""
    import httpx
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; Agent6/1.0)"})
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")

            if "json" in content_type:
                return resp.text[:100000]

            if "text" in content_type or "html" in content_type:
                html = resp.text

                # Use readability to extract main article content (like Firefox Reader View)
                try:
                    from readability import Document
                    from markdownify import markdownify as md
                    doc = Document(html)
                    clean_html = doc.summary()
                    title = doc.title()
                    text = md(clean_html, heading_style="ATX", strip=["img", "svg"])
                    text = f"# {title}\n\n{text}" if title else text
                except ImportError:
                    import re
                    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
                    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
                    text = re.sub(r"<[^>]+>", " ", text)

                import re
                text = re.sub(r"\n{3,}", "\n\n", text)
                text = re.sub(r" {2,}", " ", text)
                return text.strip()[:80000]

            return f"Binary content ({content_type}), {len(resp.content)} bytes"
    except Exception as e:
        return f"Fetch error: {e}"


@mcp.tool()
async def get_time(timezone: str = "UTC") -> str:
    """Get the current date and time. Timezone can be 'UTC', 'local', or an IANA timezone name."""
    from datetime import timezone as tz, timedelta
    from zoneinfo import ZoneInfo

    if timezone.lower() == "local":
        now = datetime.now().astimezone()
        tz_name = str(now.tzinfo)
    elif timezone.upper() == "UTC":
        now = datetime.now(tz.utc)
        tz_name = "UTC"
    else:
        try:
            now = datetime.now(ZoneInfo(timezone))
            tz_name = timezone
        except Exception:
            now = datetime.now(tz.utc)
            tz_name = f"UTC ('{timezone}' not recognized)"

    return f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')} ({tz_name})"


@mcp.tool()
async def currency_convert(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert currency using a free exchange rate API."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            url = f"https://api.exchangerate-api.com/v4/latest/{from_currency.upper()}"
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            rate = data["rates"].get(to_currency.upper())
            if rate is None:
                return f"Currency {to_currency} not found."
            result = amount * rate
            return f"{amount} {from_currency.upper()} = {result:.2f} {to_currency.upper()} (rate: {rate})"
    except Exception as e:
        return f"Conversion error: {e}"


@mcp.tool()
async def read_file(path: str) -> str:
    """Read a file from the sandbox directory. Path is relative to state/sandbox/."""
    target = SANDBOX_DIR / path
    if not target.exists():
        return f"File not found: {path}"
    try:
        return target.read_text()
    except Exception as e:
        return f"Read error: {e}"


@mcp.tool()
async def list_dir(path: str = ".") -> str:
    """List files in a sandbox directory. Path is relative to state/sandbox/."""
    target = SANDBOX_DIR / path
    if not target.exists():
        return f"Directory not found: {path}"
    if not target.is_dir():
        return f"Not a directory: {path}"
    entries = []
    for item in sorted(target.iterdir()):
        kind = "dir" if item.is_dir() else "file"
        entries.append(f"  [{kind}] {item.name}")
    return "\n".join(entries) if entries else "(empty directory)"


@mcp.tool()
async def create_file(path: str, content: str) -> str:
    """Create a new file in the sandbox directory. Path is relative to state/sandbox/."""
    target = SANDBOX_DIR / path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return f"File already exists: {path}. Use update_file or edit_file instead."
    target.write_text(content)
    return f"Created: {path} ({len(content)} bytes)"


@mcp.tool()
async def update_file(path: str, content: str) -> str:
    """Overwrite an existing file in the sandbox. Path is relative to state/sandbox/."""
    target = SANDBOX_DIR / path
    if not target.exists():
        return f"File not found: {path}. Use create_file instead."
    target.write_text(content)
    return f"Updated: {path} ({len(content)} bytes)"


@mcp.tool()
async def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace old_text with new_text in a sandbox file. Path is relative to state/sandbox/."""
    target = SANDBOX_DIR / path
    if not target.exists():
        return f"File not found: {path}"
    current = target.read_text()
    if old_text not in current:
        return f"old_text not found in {path}"
    updated = current.replace(old_text, new_text, 1)
    target.write_text(updated)
    return f"Edited: {path}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
