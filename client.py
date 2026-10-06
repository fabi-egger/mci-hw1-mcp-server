"""ReAct agent loop for Homework 1.

An LLM (any OpenAI-compatible endpoint, e.g. Gemini or a LiteLLM proxy) discovers the tools of
server.py at runtime via MCP and decides on its own which ones to call:
Thought (model text) -> Action (tool call) -> Observation (tool result) -> repeat -> final answer.

Usage:
    uv run server.py                  # terminal 1
    uv run client.py "your prompt"    # terminal 2 (configuration in .env, see .env.example)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.exceptions import ToolError
from openai import APIConnectionError, APIStatusError, InternalServerError, OpenAI, RateLimitError

LOG_DIR = Path(__file__).resolve().parent / "execution_logs"
MAX_LLM_WAITS = 4
MAX_AUTO_WAIT_SECONDS = 120  # a quota that resets later than this (e.g. the daily free-tier limit) is not waited for

DEFAULT_PROMPT = (
    "Our climbing club wants to book 12 places in the beginner top-rope course (Basiskurs Toprope). "
    "Check that enough places are free, compute the price including the group discount "
    "and record the quote in the audit log."
)

SYSTEM_PROMPT = (
    "You are a booking assistant for the Bergstation Telfs, a bouldering and climbing centre in Tyrol. "
    "Work in a ReAct style. Whenever you call tools, the same message MUST also contain ONE short sentence "
    "of plain text saying what you need and why (your Thought) - never call a tool without it. "
    "Then call the tools you need (Action) and use their results (Observation) to decide the next step. "
    "Never guess free places, prices or discounts - always use the tools. "
    "The group discount is a demo rule of this application, not an offer of the gym; say so briefly when you "
    "quote it. When you have quoted a booking, record it with append_audit_log "
    "(event 'BOOKING_QUOTED' with the key facts as details). "
    "Finish with a short, clear answer for the customer."
)


class Transcript:
    """Prints every line and, if a path is given, also appends it to a log file."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.lines: list[str] = []
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, text: str) -> None:
        self.lines.append(text)
        print(text, flush=True)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(text + "\n")


def mcp_tools_to_openai(tools) -> list[dict]:
    """Convert the tool definitions discovered via MCP into the OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema or {"type": "object", "properties": {}},
            },
        }
        for tool in tools
    ]


def assistant_message_to_dict(message) -> dict:
    if hasattr(message, "model_dump"):
        # Keep only the fields every provider accepts back (Groq rejects e.g. a top-level "reasoning"), but keep
        # everything inside tool_calls: Gemini 3 puts its thought signature there (extra_content).
        data = message.model_dump(exclude_none=True)
        result: dict = {"role": "assistant", "content": data.get("content")}
        if data.get("tool_calls"):
            result["tool_calls"] = data["tool_calls"]
        return result
    data: dict = {"role": "assistant", "content": message.content}
    if message.tool_calls:
        data["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in message.tool_calls
        ]
    return data


async def execute_tool(mcp_client: Client, name: str, raw_arguments: str | None) -> tuple[dict, str]:
    """Run one tool via MCP and return (parsed arguments, observation text). Errors become observations."""
    try:
        arguments = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError as exc:
        return {}, f"ERROR: tool arguments are not valid JSON ({exc})"
    try:
        result = await mcp_client.call_tool(name, arguments)
    except ToolError as exc:
        return arguments, f"ERROR: {exc}"
    if result.data is not None:
        return arguments, json.dumps(result.data, ensure_ascii=False)
    text = " ".join(getattr(block, "text", "") for block in result.content)
    return arguments, text


def parse_retry_seconds(text: str) -> float | None:
    """Read the delay a provider asks for, e.g. 'retry in 37.0s' (Gemini) or 'try again in 14m22.5s' (Groq)."""
    match = re.search(r"(?:retry|try again) in ((?:\d+(?:\.\d+)?(?:ms|[hms]))+)", text, re.IGNORECASE)
    if not match:
        return None
    units = {"h": 3600, "m": 60, "s": 1, "ms": 0.001}
    parts = re.findall(r"(\d+(?:\.\d+)?)(ms|[hms])", match.group(1), re.IGNORECASE)
    return sum(float(number) * units[unit.lower()] for number, unit in parts)


def format_duration(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h" if seconds >= 3600 else f"{seconds / 60:.0f} min"


def call_llm(llm, transcript: Transcript, sleep, **request):
    """Call the LLM and wait out short hiccups: rate limits (free tiers allow only a few requests per minute),
    503 'high demand' errors and dropped connections. A quota that resets only after hours is not waited for."""
    for attempt in range(MAX_LLM_WAITS + 1):
        try:
            return llm.chat.completions.create(**request)
        except RateLimitError as exc:
            advised = parse_retry_seconds(str(exc))
            if advised is not None and advised > MAX_AUTO_WAIT_SECONDS:
                raise RuntimeError(
                    f"The quota for model '{request['model']}' is used up; the provider asks to retry in "
                    f"{format_duration(advised)}. Use another LLM_MODEL in .env, wait, or enable billing for the key."
                ) from exc
            error, reason = exc, "Rate limit reached"
            wait = advised + 2 if advised is not None else 30.0
        except (InternalServerError, APIConnectionError) as exc:
            error, reason = exc, f"LLM temporarily unavailable ({type(exc).__name__})"
            wait = 10.0 * (attempt + 1)
        if attempt == MAX_LLM_WAITS:
            raise error
        transcript.write(f"[Wait] {reason}. Waiting {wait:.0f} s, then retrying the same request.")
        sleep(wait)


async def run_react(llm, model: str, mcp_client: Client, user_prompt: str,
                    transcript: Transcript, max_iterations: int = 8, sleep=time.sleep) -> str:
    tools = await mcp_client.list_tools()
    openai_tools = mcp_tools_to_openai(tools)
    transcript.write(f"Discovered MCP tools: {[tool.name for tool in tools]}")

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    for step in range(1, max_iterations + 1):
        transcript.write(f"\n=== Step {step}: querying the LLM ===")
        response = call_llm(
            llm,
            transcript,
            sleep,
            model=model,
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
            temperature=0,
        )
        message = response.choices[0].message

        if not message.tool_calls:
            answer = (message.content or "").strip()
            transcript.write(f"\n--- Final answer ---\n{answer}")
            return answer

        if message.content and message.content.strip():
            transcript.write(f"[Thought] {message.content.strip()}")
        messages.append(assistant_message_to_dict(message))

        for call in message.tool_calls:
            arguments, observation = await execute_tool(mcp_client, call.function.name, call.function.arguments)
            transcript.write(f"[Action] {call.function.name}({json.dumps(arguments, ensure_ascii=False)})")
            transcript.write(f"[Observation] {observation}")
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "name": call.function.name, "content": observation}
            )

    raise RuntimeError(f"Stopped: no final answer after {max_iterations} steps.")


def describe_llm_error(exc: Exception) -> str:
    """One readable line instead of a traceback (wrong model name, bad key, endpoint unreachable, ...)."""
    if isinstance(exc, APIStatusError):
        # The SDK keeps the provider's own sentence in exc.body (a dict for Groq/OpenAI, a list for Gemini).
        body = exc.body
        detail = body["message"] if isinstance(body, dict) and body.get("message") else exc.message
        return f"{type(exc).__name__} (HTTP {exc.status_code}): {detail}"
    if isinstance(exc, APIConnectionError):
        return f"{type(exc).__name__}: could not reach the LLM endpoint ({exc})"
    return str(exc)


def log_path(stamp: str, model: str) -> Path:
    """execution_logs/run-<timestamp>-<model>.txt, so the model is visible from the file name."""
    return LOG_DIR / f"run-{stamp}-{re.sub(r'[^A-Za-z0-9._-]+', '-', model)}.txt"


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value or value.startswith("<"):
        sys.exit(f"Missing configuration: set {name} in .env (see .env.example).")
    return value


async def main() -> None:
    parser = argparse.ArgumentParser(description="ReAct agent that uses the MCP tools of server.py")
    parser.add_argument("prompt", nargs="*", help="request for the agent (default: example order)")
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--no-log", action="store_true", help="do not write a file to execution_logs/")
    args = parser.parse_args()

    load_dotenv()
    base_url = require_env("LLM_BASE_URL")
    api_key = require_env("LLM_API_KEY")
    model = require_env("LLM_MODEL")
    mcp_url = os.environ.get("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp")
    prompt = " ".join(args.prompt) or DEFAULT_PROMPT

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    transcript = Transcript(None if args.no_log else log_path(stamp, model))
    transcript.write(f"Run {stamp} | model: {model} | LLM host: {urlparse(base_url).netloc} | MCP server: {mcp_url}")
    transcript.write(f"User request: {prompt}")

    # The SDK's own retries are off: call_llm handles every retry, so each wait is visible in the log.
    llm = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)
    try:
        async with Client(mcp_url) as mcp_client:
            await run_react(llm, model, mcp_client, prompt, transcript, args.max_iterations)
    except (RuntimeError, APIStatusError, APIConnectionError) as exc:
        transcript.write(f"\n[Error] {describe_llm_error(exc)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
