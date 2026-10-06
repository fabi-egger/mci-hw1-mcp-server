# Homework 1 – Custom MCP Math & Database Tool Server

Industrial Computing (MCI, DiBSE, WS2026), Option A (technical track).

Domain: group bookings for the courses of the climbing gym **Bergstation Telfs** (Tyrol).

A Python **MCP server** exposes three tools, and a **ReAct agent loop** lets an LLM discover and use them at runtime:

| Tool | What it does |
|---|---|
| `lookup_inventory(query, limit)` | **SQL lookup** (sqlite3) on the gym's courses and offers: by SKU or by words from the German/English name or category; returns schedule, price, what is included, free places and requirements |
| `compute_tiered_discount(quantity, unit_price_eur)` | **Formula engine**: graduated group discount with a per-tier breakdown |
| `append_audit_log(event, details)` | **Log audit**: appends one JSON line per event to `data/audit.log`; the same file is exposed read-only as the MCP **resource** `audit://log` |

```
 user request ──► client.py (ReAct loop) ──► LLM (any OpenAI-compatible API)
                        │  Thought → Action → Observation → … → answer
                        ▼  MCP over HTTP (http://127.0.0.1:8000/mcp)
                  server.py (FastMCP) ──► data/inventory.db (SQLite)
                                      └─► data/audit.log   (JSON Lines)
```

The client does not know the tools in advance: it lists them via MCP (`list_tools`), passes their JSON schemas to the LLM and executes whatever tool calls the model returns.

## About the data

The offers in the database are **real**: courses and packages of the bouldering and climbing centre [Bergstation Telfs](https://bergstation.tirol/), copied from the public page <https://bergstation.tirol/kurse> on **2026-10-06** (name, schedule, price, what the price includes, free places, requirements). Every row names its source.

- **Snapshot:** free places change daily; prices and dates may be outdated. Check bergstation.tirol before relying on them.
- **Not included:** entry and rental prices (only visible in the ticket shop after login), trainer names and contact details.
- **Fictional:** the group discount below (our own demo rule, not an offer of the gym) and the audit-log entries.
- No booking is made; the tools only produce quotes. This is a student project and is **not affiliated with, endorsed or authorised by** Bergstation Telfs.

## Discount rule (demo rule)

Every unit (place, hour) gets the discount of the tier it falls into (like tax brackets). Amounts are rounded half-up to cents per tier; totals are the sum of the tier values.

| Units | Discount |
|---|---|
| 1–9 | 0 % |
| 10–49 | 5 % |
| 50–99 | 10 % |
| 100–499 | 15 % |
| 500 and more | 20 % |

Example: a club books 12 places of the top-rope basic course at 35.00 EUR → 9×0 % + 3×5 % → gross 420.00, discount 5.25, **net 414.75 EUR** (effective 1.25 %).

## Setup

Requirements: [uv](https://docs.astral.sh/uv/) (it installs a matching Python ≥ 3.11 automatically) and an API key for any OpenAI-compatible LLM endpoint (e.g. a free Groq or Google AI Studio key; the examples in `execution_logs/` were produced with Groq / `qwen/qwen3.8-27b` and, earlier, Google's `gemini-3.5-flash`).

```bash
uv sync
cp .env.example .env      # then edit .env: set LLM_API_KEY (and check LLM_MODEL)
```

### Notes on free LLM tiers

A single agent run needs 3–5 LLM requests (one per ReAct step, roughly 5,000 input tokens in total). Free tiers are tight (October 2026):

- **Groq** (recommended, `.env.example`): about 30 requests/min, 1,000 requests/day and 8,000 tokens/min, 200,000 tokens/day per model. Model choice matters: `openai/gpt-oss-120b` wrote its tool call into the reasoning field and then stopped with an empty answer, while `qwen/qwen3.8-27b` makes proper tool calls.
- **Google AI Studio** (Gemini): about 5 requests/min and only 20 requests **per day and model**.

`client.py` therefore:

- waits and retries on per-minute limits, 503 "high demand" errors and dropped connections (each wait is printed as `[Wait]` and stored in the log);
- stops immediately with a clear `[Error]` when the quota resets only after hours – switch `LLM_MODEL` in `.env`, wait, or use another provider.

## Run

Terminal 1 – start the MCP server:

```bash
uv run server.py
```

Terminal 2 – run the agent (without arguments it uses an example booking):

```bash
uv run client.py
uv run client.py "Is the slightly advanced boulder training group still free? If not, what would 4 places of the beginners group cost?"
```

Every run is also written to `execution_logs/run-<timestamp>.txt` (Thought / Action / Observation / final answer).

Example prompts that exercise the tools:

- `Is the slightly advanced boulder training group still free?` → lookup, fully booked (0/6) case
- `We want to register 6 children for the ÖAV children's group - is there room, and what are the requirements?` → lookup, only 4 of 8 places free, requirement "Topropeschein"
- `Our club wants 12 places in the beginner top-rope course. Quote it and record the quote in the audit log.` → all three tools in sequence

Read the audit log resource with any MCP client, or just `cat data/audit.log`.

## Tests (no LLM, no API key needed)

```bash
uv run pytest
```

The tests call the tools through an in-process MCP client (discount math against hand calculations, SQL-injection attempts, audit-log sanitising, the resource) and run the ReAct loop against a scripted fake LLM.

## Design and security notes

- **SQL:** parameterised queries only (the SQL text consists of constant fragments, the number of search words is capped at 6), `LIKE` wildcards are escaped, the lookup uses a **read-only** database connection.
- **Audit log:** fixed file path (the model cannot choose a path), event names restricted to `[A-Za-z0-9_.:-]`, one JSON object per line (newlines are escaped, so an entry cannot forge further entries).
- **Money:** `Decimal` instead of floats; prices are stored as integer cents.
- **Server** binds to `127.0.0.1` only. Inputs are validated and errors are returned to the model as observations so it can correct itself.
- The ReAct loop is capped at 8 steps by default (`--max-iterations`).

## Project structure

```
server.py            MCP server: tools, resource, SQLite seed data (Bergstation offers)
client.py            ReAct agent loop (LLM + MCP client), writes execution logs
tests/test_server.py tool, resource and agent-loop tests
execution_logs/      transcripts of real runs
data/                inventory.db (generated), audit.log
.env.example         configuration template
```
