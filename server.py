"""MCP server for Homework 1: SQL lookup of the courses/offers of the climbing gym Bergstation Telfs,
tiered group discount, audit log.

The offers are real (published on bergstation.tirol/kurse, snapshot 2026-10-06). The group discount is our
own demo rule and not an offer of the gym.

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
MAX_QUERY_WORDS = 6
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

SOURCE = "bergstation.tirol/kurse, 2026-10-06"

# Courses and offers of the Bergstation Telfs as published on the course page (snapshot 2026-10-06).
# Prices in integer cents; free_places/max_places are None where the page names no fixed capacity.
# Entry and rental prices are not part of the data: they are only visible inside the ticket shop (login).
# (sku, name, name_en, category, schedule, price_cents, price_unit, price_note,
#  free_places, max_places, requirement, source)
SEED_OFFERS = [
    ("BST-AUF-2026-10", "Aufbaukurs Seilklettern - Vorstieg", "Lead climbing advanced course", "Rope course",
     "Wednesdays 19:00-21:00, 14.10. and 21.10.2026", 7000, "per place", "incl. entry and rental",
     0, 4, None, SOURCE),
    ("BST-AUF-2027-01", "Aufbaukurs Seilklettern - Vorstieg", "Lead climbing advanced course", "Rope course",
     "Wednesdays 18:00-20:00, 13.01. and 20.01.2027", 7000, "per place", "incl. entry and rental",
     6, 6, None, SOURCE),
    ("BST-BAS-2026-11-1", "Basiskurs Seilklettern - Toprope", "Top-rope basic course for beginners", "Rope course",
     "Friday 18:30-20:30, 13.11.2026 (listed twice, booked separately)", 3500, "per place",
     "incl. entry and rental", 6, 6, None, SOURCE),
    ("BST-BAS-2026-11-2", "Basiskurs Seilklettern - Toprope", "Top-rope basic course for beginners", "Rope course",
     "Friday 18:30-20:30, 13.11.2026 (listed twice, booked separately)", 3500, "per place",
     "incl. entry and rental", 6, 6, None, SOURCE),
    ("BST-BAS-2027-01", "Basiskurs Seilklettern - Toprope", "Top-rope basic course for beginners", "Rope course",
     "Friday 18:30-20:30, 29.01.2027", 3500, "per place", "incl. entry and rental", 6, 6, None, SOURCE),
    ("BST-BOU-ANF-2026-11", "Trainingsgruppe Erwachsene Anfänger", "Adult boulder training group beginners",
     "Boulder course", "Thursdays 16:30-18:00, 26.11.-17.12.2026", 10500, "per place",
     "excl. entry, incl. rental", 6, 6, "For participants who currently boulder at V2-3", SOURCE),
    ("BST-BOU-FOR-2026-11", "Trainingsgruppe Erwachsene leicht Fortgeschrittene",
     "Adult boulder training group slightly advanced", "Boulder course", "Wednesdays, 04.11.-25.11.2026",
     10500, "per place", "excl. entry, incl. rental", 0, 6, "For participants who currently boulder at V2-3", SOURCE),
    ("BST-OAV-KIDS-2026", "ÖAV Betreute Gruppe Kinder 7-10 Jahre", "ÖAV supervised children's group age 7-10",
     "Children's group", "Wednesdays 14:30-16:00, 25.11.2026-17.03.2027", 15000, "per place",
     "excl. entry and rental", 4, 8, "Completed top-rope course (Topropeschein)", SOURCE),
    ("BST-PT", "Personal Training", "Personal training private coaching", "Individual",
     "On request", 6000, "per hour", "excl. entry", None, None, None, SOURCE),
    ("BST-GEB-KIDS", "Kindergeburtstag ohne Trainer Kinderbereich",
     "Children's birthday party kids area without trainer", "Birthday party", "2.5 hours, on request",
     9900, "flat fee for up to 8 children, 7.00 EUR per additional child",
     "incl. rental and entry to the kids area", None, None, None, SOURCE),
    ("BST-GEB-HALL", "Kindergeburtstag ohne Trainer ganze Halle",
     "Children's birthday party whole hall without trainer", "Birthday party", "2.5 hours, on request",
     11900, "flat fee for up to 8 children, 7.00 EUR per additional child",
     "incl. rental and entry to the whole hall", None, None, None, SOURCE),
]

mcp = FastMCP("Bergstation-Telfs-MCP-Server")
_log_lock = threading.Lock()


def init_db() -> None:
    """Create the offers table and seed it once (prices are stored as integer cents)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS offers (
                sku TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                name_en TEXT NOT NULL,
                category TEXT NOT NULL,
                schedule TEXT NOT NULL,
                unit_price_cents INTEGER NOT NULL CHECK (unit_price_cents > 0),
                price_unit TEXT NOT NULL,
                price_note TEXT NOT NULL,
                free_places INTEGER CHECK (free_places IS NULL OR free_places >= 0),
                max_places INTEGER CHECK (max_places IS NULL OR max_places > 0),
                requirement TEXT,
                source TEXT NOT NULL
            )
            """
        )
        if conn.execute("SELECT COUNT(*) FROM offers").fetchone()[0] == 0:
            conn.executemany("INSERT INTO offers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", SEED_OFFERS)


def _like_pattern(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _money(value: Decimal) -> float:
    return float(value.quantize(CENT, rounding=ROUND_HALF_UP))


@mcp.tool()
def lookup_inventory(query: str, limit: int = 10) -> dict:
    """Look up courses and offers of the climbing gym Bergstation Telfs in the database (SQLite).

    Matches the query against the exact SKU or against the German name, English name and category:
    every word of the query must appear in one of them (case-insensitive, e.g. 'basiskurs' or
    'top-rope beginners'). Returns schedule, price in EUR with its unit and note, the free places
    (null = no fixed capacity, on request) and requirements. Prices are the published ones; free
    places are a snapshot and can change.

    Args:
        query: SKU (e.g. 'BST-PT') or one or more words from the name/category.
        limit: Maximum number of offers to return (1-20).
    """
    query = query.strip()
    if not query or len(query) > MAX_QUERY_LEN:
        raise ValueError(f"query must be 1-{MAX_QUERY_LEN} characters")
    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20")

    words = query.split()[:MAX_QUERY_WORDS]
    # The SQL text is assembled from constant fragments only; every user-supplied value is a bound parameter.
    word_match = "(name LIKE ? ESCAPE '\\' OR name_en LIKE ? ESCAPE '\\' OR category LIKE ? ESCAPE '\\')"
    sql = f"""
        SELECT sku, name, category, schedule, unit_price_cents, price_unit, price_note,
               free_places, max_places, requirement, source
        FROM offers
        WHERE sku = ? COLLATE NOCASE
           OR ({" AND ".join([word_match] * len(words))})
        ORDER BY sku
        LIMIT ?
    """
    params = [query] + [_like_pattern(word) for word in words for _ in range(3)] + [limit]
    # Read-only connection: even a successful injection could not modify the data.
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
        rows = conn.execute(sql, params).fetchall()

    offers = []
    for sku, name, category, schedule, cents, price_unit, price_note, free, maximum, requirement, source in rows:
        offer = {
            "sku": sku,
            "name": name,
            "category": category,
            "schedule": schedule,
            "unit_price_eur": _money(Decimal(cents) / 100),
            "price_unit": price_unit,
            "price_note": price_note,
            "free_places": free,
            "max_places": maximum,
            "available": free is None or free > 0,
            "source": source,
        }
        if requirement:
            offer["requirement"] = requirement
        offers.append(offer)
    return {"query": query, "count": len(offers), "offers": offers}


@mcp.tool()
def compute_tiered_discount(quantity: int, unit_price_eur: float) -> dict:
    """Compute the price of a group booking with a graduated volume discount.

    This is a demo rule of this application, not an offer of the gym. Every unit (place, hour, ...)
    receives the discount of the tier it falls into (like tax brackets):
    units 1-9: 0%, 10-49: 5%, 50-99: 10%, 100-499: 15%, 500 and more: 20%.
    Amounts are rounded half-up to cents per tier; totals are the sum of the tier values.

    Args:
        quantity: Number of units booked (1 to 1,000,000).
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

    Use it to record business events such as quotes or bookings,
    e.g. event='BOOKING_QUOTED', details='12 x BST-BAS-2026-11-1, net 414.75 EUR'.

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
