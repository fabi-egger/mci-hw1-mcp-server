# Homework 1 – Custom MCP Math & Database Tool Server

Industrial Computing (MCI, DiBSE, WS2026), Option A (technical track).

A Python **MCP server** exposes three tools, and a **ReAct agent loop** lets an LLM discover and use them at runtime:

| Tool | What it does |
|---|---|
| `lookup_inventory(query, limit)` | **SQL lookup** (sqlite3) on a product inventory: by SKU or name/category substring, returns stock and unit price |
| `compute_tiered_discount(quantity, unit_price_eur)` | **Formula engine**: graduated volume discount with a per-tier breakdown |
| `append_audit_log(event, details)` | **Log audit**: appends one JSON line per event to `data/audit.log`; the same file is exposed read-only as the MCP **resource** `audit://log` |

```
 user request ──► client.py (ReAct loop) ──► LLM (any OpenAI-compatible API)
                        │  Thought → Action → Observation → … → answer
                        ▼  MCP over HTTP (http://127.0.0.1:8000/mcp)
                  server.py (FastMCP) ──► data/inventory.db (SQLite)
                                      └─► data/audit.log   (JSON Lines)
```

The client does not know the tools in advance: it lists them via MCP (`list_tools`), passes their JSON schemas to the LLM and executes whatever tool calls the model returns.

## Discount rule

Every unit gets the discount of the tier it falls into (like tax brackets). Amounts are rounded half-up to cents per tier; totals are the sum of the tier values.

| Units | Discount |
|---|---|
| 1–9 | 0 % |
| 10–49 | 5 % |
| 50–99 | 10 % |
| 100–499 | 15 % |
| 500 and more | 20 % |

Example: 120 units at 10.00 EUR → 9×0 % + 40×5 % + 50×10 % + 21×15 % → gross 1200.00, discount 101.50, **net 1098.50 EUR** (effective 8.46 %).

## Setup

Requirements: [uv](https://docs.astral.sh/uv/) (it installs a matching Python ≥ 3.11 automatically) and an API key for any OpenAI-compatible LLM endpoint (e.g. a free Google AI Studio key).

```bash
uv sync
cp .env.example .env      # then edit .env: set LLM_API_KEY (and check LLM_MODEL)
```

## Run

Terminal 1 – start the MCP server:

```bash
uv run server.py
```

Terminal 2 – run the agent (without arguments it uses an example order):

```bash
uv run client.py
uv run client.py "Which bearings do you have in stock? I need 600 of the cheaper one - what is the total?"
```

Every run is also written to `execution_logs/run-<timestamp>.txt` (Thought / Action / Observation / final answer).

Example prompts that exercise the tools:

- `Do you have SKU-2002 in stock?` → inventory lookup, out-of-stock case
- `What does an order of 600 units at 0.33 EUR cost?` → discount engine, all tiers
- `Quote 120 inductive proximity sensors and record the quote in the audit log.` → all three tools in sequence

Read the audit log resource with any MCP client, or just `cat data/audit.log`.

## Tests (no LLM, no API key needed)

```bash
uv run pytest
```

The tests call the tools through an in-process MCP client (discount math against hand calculations, SQL-injection attempts, audit-log sanitising, the resource) and run the ReAct loop against a scripted fake LLM.

## Design and security notes

- **SQL:** parameterised queries only, `LIKE` wildcards are escaped, the lookup uses a **read-only** database connection.
- **Audit log:** fixed file path (the model cannot choose a path), event names restricted to `[A-Za-z0-9_.:-]`, one JSON object per line (newlines are escaped, so an entry cannot forge further entries).
- **Money:** `Decimal` instead of floats; prices are stored as integer cents.
- **Server** binds to `127.0.0.1` only. Inputs are validated and errors are returned to the model as observations so it can correct itself.
- The ReAct loop is capped at 8 steps by default (`--max-iterations`).

## Project structure

```
server.py            MCP server: tools, resource, SQLite seed data
client.py            ReAct agent loop (LLM + MCP client), writes execution logs
tests/test_server.py tool, resource and agent-loop tests
execution_logs/      transcripts of real runs
data/                inventory.db (generated), audit.log
.env.example         configuration template
```
