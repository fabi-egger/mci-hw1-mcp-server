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
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.exceptions import ToolError
from openai import OpenAI

LOG_DIR = Path(__file__).resolve().parent / "execution_logs"

DEFAULT_PROMPT = (
    "I want to order 120 inductive proximity sensors. Check that they are in stock, "
    "compute the price including the volume discount and record the quote in the audit log."
)

SYSTEM_PROMPT = (
    "You are a procurement assistant for a shop that sells industrial components. "
    "Work in a ReAct style: before every tool call write ONE short sentence explaining what you need "
    "and why (Thought), then call the tools you need (Action), and use their results (Observation) "
    "to decide the next step. Never guess stock levels, prices or discounts - always use the tools. "
    "When you have quoted or confirmed an order, record it with append_audit_log "
    "(event 'ORDER_QUOTED' with the key facts as details). "
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


async def run_react(llm, model: str, mcp_client: Client, user_prompt: str,
                    transcript: Transcript, max_iterations: int = 8) -> str:
    tools = await mcp_client.list_tools()
    openai_tools = mcp_tools_to_openai(tools)
    transcript.write(f"Discovered MCP tools: {[tool.name for tool in tools]}")

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    for step in range(1, max_iterations + 1):
        transcript.write(f"\n=== Step {step}: querying the LLM ===")
        response = llm.chat.completions.create(
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
    transcript = Transcript(None if args.no_log else LOG_DIR / f"run-{stamp}.txt")
    transcript.write(f"Run {stamp} | model: {model} | LLM host: {urlparse(base_url).netloc} | MCP server: {mcp_url}")
    transcript.write(f"User request: {prompt}")

    llm = OpenAI(base_url=base_url, api_key=api_key, max_retries=5)
    async with Client(mcp_url) as mcp_client:
        await run_react(llm, model, mcp_client, prompt, transcript, args.max_iterations)


if __name__ == "__main__":
    asyncio.run(main())
