"""MCP server for Homework 1: SQL inventory lookup, tiered discount, audit log.

Run:  uv run server.py   (serves MCP over HTTP at http://127.0.0.1:8000/mcp)
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from fastmcp import FastMCP

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "inventory.db"
AUDIT_LOG_PATH = DATA_DIR / "audit.log"

CENT = Decimal("0.01")
MAX_QUERY_LEN = 100
MAX_EVENT_LEN = 64
MAX_DETAILS_LEN = 500
AUDIT_RESOURCE_LINES = 200

# Graduated (marginal) volume discount: every unit gets the discount of the tier it falls into.
# (first_unit, last_unit or None for open end, discount in percent)
DISCOUNT_TIERS: list[tuple[int, int | None, Decimal]] = [
    (1, 9, Decimal("0")),
    (10, 49, Decimal("5")),
    (50, 99, Decimal("10")),
    (100, 499, Decimal("15")),
    (500, None, Decimal("20")),
]

SEED_PRODUCTS = [
    ("SKU-1001", "Hex Bolt M8x40", "Fasteners", 1200, 18),
    ("SKU-1002", "Hex Nut M8", "Fasteners", 3400, 7),
    ("SKU-2001", "Deep Groove Ball Bearing 6204", "Bearings", 85, 690),
    ("SKU-2002", "Tapered Roller Bearing 30205", "Bearings", 0, 1150),
    ("SKU-3001", "Stepper Motor NEMA 17", "Motors", 42, 1890),
    ("SKU-3002", "Servo Motor 400W", "Motors", 7, 12900),
    ("SKU-4001", "Inductive Proximity Sensor M12", "Sensors", 160, 1420),
    ("SKU-4002", "Temperature Sensor PT100", "Sensors", 95, 980),
    ("SKU-5001", "Industrial Ethernet Cable 5m", "Cabling", 300, 850),
    ("SKU-5002", "24V DC Power Supply 10A", "Power", 28, 6400),
]

mcp = FastMCP("MCI-HW1-Inventory-Server")
_log_lock = threading.Lock()


def init_db() -> None:
    """Create the inventory table and seed it once (prices are stored as integer cents)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS products (
                sku TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                stock INTEGER NOT NULL CHECK (stock >= 0),
                unit_price_cents INTEGER NOT NULL CHECK (unit_price_cents > 0)
            )
            """
        )
        if conn.execute("SELECT COUNT(*) FROM products").fetchone()[0] == 0:
            conn.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?)", SEED_PRODUCTS)


def _like_pattern(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _money(value: Decimal) -> float:
    return float(value.quantize(CENT, rounding=ROUND_HALF_UP))


@mcp.tool()
def lookup_inventory(query: str, limit: int = 5) -> dict:
    """Look up products in the inventory database (SQLite).

    Matches the query against the exact SKU or as a case-insensitive substring of the
    product name or category. Returns stock level and unit price in EUR.

    Args:
        query: SKU (e.g. 'SKU-2001') or a word from the name/category (e.g. 'bearing').
        limit: Maximum number of products to return (1-20).
    """
    query = query.strip()
    if not query or len(query) > MAX_QUERY_LEN:
        raise ValueError(f"query must be 1-{MAX_QUERY_LEN} characters")
    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20")

    pattern = _like_pattern(query)
    # Read-only connection: even a successful injection could not modify the data.
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            """
            SELECT sku, name, category, stock, unit_price_cents
            FROM products
            WHERE sku = ? COLLATE NOCASE
               OR name LIKE ? ESCAPE '\\'
               OR category LIKE ? ESCAPE '\\'
            ORDER BY sku
            LIMIT ?
            """,
            (query, pattern, pattern, limit),
        ).fetchall()

    products = [
        {
            "sku": sku,
            "name": name,
            "category": category,
            "stock": stock,
            "in_stock": stock > 0,
            "unit_price_eur": _money(Decimal(cents) / 100),
        }
        for sku, name, category, stock, cents in rows
    ]
    return {"query": query, "count": len(products), "products": products}


@mcp.tool()
def compute_tiered_discount(quantity: int, unit_price_eur: float) -> dict:
    """Compute the price of an order with a graduated volume discount.

    Every unit receives the discount of the tier it falls into (like tax brackets):
    units 1-9: 0%, 10-49: 5%, 50-99: 10%, 100-499: 15%, 500 and more: 20%.
    Amounts are rounded half-up to cents per tier; totals are the sum of the tier values.

    Args:
        quantity: Number of units ordered (1 to 1,000,000).
        unit_price_eur: List price per unit in EUR (greater than 0).
    """
    if not 1 <= quantity <= 1_000_000:
        raise ValueError("quantity must be between 1 and 1,000,000")
    if not math.isfinite(unit_price_eur) or not 0 < unit_price_eur <= 1_000_000:
        raise ValueError("unit_price_eur must be a finite number greater than 0 and at most 1,000,000")

    price = Decimal(str(unit_price_eur))
    breakdown = []
    gross_total = discount_total = net_total = Decimal("0")

    for first, last, percent in DISCOUNT_TIERS:
        if quantity < first:
            break
        upper = quantity if last is None else min(quantity, last)
        units = upper - first + 1
        gross = (price * units).quantize(CENT, rounding=ROUND_HALF_UP)
        discount = (gross * percent / 100).quantize(CENT, rounding=ROUND_HALF_UP)
        net = gross - discount
        gross_total += gross
        discount_total += discount
        net_total += net
        breakdown.append(
            {
                "tier": f"{first}-{last}" if last is not None else f"{first}+",
                "units": units,
                "discount_percent": float(percent),
                "gross_eur": _money(gross),
                "discount_eur": _money(discount),
                "net_eur": _money(net),
            }
        )

    effective = (discount_total / gross_total * 100) if gross_total else Decimal("0")
    return {
        "quantity": quantity,
        "unit_price_eur": float(price),
        "breakdown": breakdown,
        "gross_total_eur": _money(gross_total),
        "discount_total_eur": _money(discount_total),
        "net_total_eur": _money(net_total),
        "effective_discount_percent": float(effective.quantize(CENT, rounding=ROUND_HALF_UP)),
    }


@mcp.tool()
def append_audit_log(event: str, details: str = "") -> dict:
    """Append one event to the audit log file (JSON Lines, append-only).

    Use it to record business events such as quotes or orders,
    e.g. event='ORDER_QUOTED', details='120 x SKU-4001, net 1098.50 EUR'.

    Args:
        event: Short event type (letters, digits, '_', '-', '.', ':'; max 64 characters).
        details: Free-text details (max 500 characters).
    """
    event = event.strip()
    if not event or len(event) > MAX_EVENT_LEN or not all(c.isalnum() or c in "_-.:" for c in event):
        raise ValueError("event must be 1-64 characters: letters, digits, '_', '-', '.', ':'")
    if len(details) > MAX_DETAILS_LEN:
        raise ValueError(f"details must not exceed {MAX_DETAILS_LEN} characters")

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": event,
        "details": details,
    }
    # json.dumps escapes newlines, so every entry stays on exactly one line (no log injection).
    line = json.dumps(entry, ensure_ascii=False)
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _log_lock, AUDIT_LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return {"status": "logged", "entry": entry}


@mcp.resource("audit://log", mime_type="text/plain")
def read_audit_log() -> str:
    """Read-only view of the audit log file (the most recent 200 entries)."""
    if not AUDIT_LOG_PATH.exists():
        return "(audit log is empty)"
    lines = AUDIT_LOG_PATH.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[-AUDIT_RESOURCE_LINES:]) or "(audit log is empty)"


def main() -> None:
    init_db()
    port = int(os.environ.get("MCP_PORT", "8000"))
    mcp.run(transport="http", host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
