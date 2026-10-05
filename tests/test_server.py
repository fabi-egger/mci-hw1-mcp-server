"""Tests for the MCP tools and the ReAct loop. No LLM and no API key needed."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

import client as agent_client
import server


@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server, "DB_PATH", tmp_path / "inventory.db")
    monkeypatch.setattr(server, "AUDIT_LOG_PATH", tmp_path / "audit.log")
    server.init_db()
    return tmp_path


def call(name: str, arguments: dict):
    async def run():
        async with Client(server.mcp) as mcp_client:
            return await mcp_client.call_tool(name, arguments)

    return asyncio.run(run()).data


def call_error(name: str, arguments: dict) -> str:
    with pytest.raises(ToolError) as exc_info:
        call(name, arguments)
    return str(exc_info.value)


# --- tool 1: inventory lookup -------------------------------------------------------------

def test_lookup_by_sku_and_by_name(data_dir):
    by_sku = call("lookup_inventory", {"query": "sku-4001"})
    assert by_sku["count"] == 1
    assert by_sku["products"][0]["name"] == "Inductive Proximity Sensor M12"
    assert by_sku["products"][0]["unit_price_eur"] == 14.20

    by_word = call("lookup_inventory", {"query": "bearing"})
    assert {p["sku"] for p in by_word["products"]} == {"SKU-2001", "SKU-2002"}


def test_out_of_stock_is_flagged(data_dir):
    result = call("lookup_inventory", {"query": "SKU-2002"})
    assert result["products"][0]["stock"] == 0
    assert result["products"][0]["in_stock"] is False


def test_sql_injection_is_harmless(data_dir):
    assert call("lookup_inventory", {"query": "'; DROP TABLE products; --"})["count"] == 0
    assert call("lookup_inventory", {"query": "' OR '1'='1"})["count"] == 0
    assert call("lookup_inventory", {"query": "%"})["count"] == 0  # wildcard is escaped
    assert call("lookup_inventory", {"query": "bearing"})["count"] == 2  # table still intact


def test_lookup_validates_input(data_dir):
    assert "query must be" in call_error("lookup_inventory", {"query": "   "})
    assert "limit must be" in call_error("lookup_inventory", {"query": "bolt", "limit": 0})


# --- tool 2: tiered discount ---------------------------------------------------------------

def test_discount_breakdown_matches_hand_calculation(data_dir):
    # 120 units at 10.00 EUR: 9 x 0%, 40 x 5%, 50 x 10%, 21 x 15%
    result = call("compute_tiered_discount", {"quantity": 120, "unit_price_eur": 10.0})
    assert [t["units"] for t in result["breakdown"]] == [9, 40, 50, 21]
    assert [t["net_eur"] for t in result["breakdown"]] == [90.0, 380.0, 450.0, 178.5]
    assert result["gross_total_eur"] == 1200.0
    assert result["discount_total_eur"] == 101.5
    assert result["net_total_eur"] == 1098.5
    assert result["effective_discount_percent"] == 8.46


def test_small_order_gets_no_discount(data_dir):
    result = call("compute_tiered_discount", {"quantity": 5, "unit_price_eur": 2.5})
    assert result["discount_total_eur"] == 0.0
    assert result["net_total_eur"] == 12.5
    assert len(result["breakdown"]) == 1


def test_open_ended_top_tier(data_dir):
    result = call("compute_tiered_discount", {"quantity": 600, "unit_price_eur": 1.0})
    assert result["breakdown"][-1]["tier"] == "500+"
    assert result["breakdown"][-1]["units"] == 101
    assert result["breakdown"][-1]["discount_percent"] == 20.0


def test_discount_totals_equal_sum_of_rounded_tiers(data_dir):
    # Odd prices force rounding in every tier; the totals must still equal the sum of the tier values.
    result = call("compute_tiered_discount", {"quantity": 77, "unit_price_eur": 0.33})
    assert round(result["gross_total_eur"] - result["discount_total_eur"], 2) == result["net_total_eur"]
    assert round(sum(t["net_eur"] for t in result["breakdown"]), 2) == result["net_total_eur"]


def test_discount_rejects_invalid_input(data_dir):
    assert "quantity must be" in call_error("compute_tiered_discount", {"quantity": 0, "unit_price_eur": 1.0})
    assert "unit_price_eur must be" in call_error("compute_tiered_discount", {"quantity": 1, "unit_price_eur": -3})
    assert "unit_price_eur must be" in call_error("compute_tiered_discount", {"quantity": 1, "unit_price_eur": 0})


# --- tool 3: audit log + resource ----------------------------------------------------------

def test_audit_log_appends_exactly_one_line_per_event(data_dir):
    call("append_audit_log", {"event": "ORDER_QUOTED", "details": "first"})
    call("append_audit_log", {"event": "ORDER_QUOTED", "details": "line1\nline2\n" + json.dumps({"event": "FAKE"})})
    lines = (data_dir / "audit.log").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    entries = [json.loads(line) for line in lines]
    assert entries[1]["event"] == "ORDER_QUOTED"
    assert "line2" in entries[1]["details"]  # content preserved, but no forged third entry


def test_audit_log_rejects_bad_event_names(data_dir):
    assert "event must be" in call_error("append_audit_log", {"event": "bad event"})
    assert "event must be" in call_error("append_audit_log", {"event": "x\n{\"event\":\"FAKE\"}"})
    assert "event must be" in call_error("append_audit_log", {"event": ""})
    assert not (data_dir / "audit.log").exists()


def test_audit_resource_exposes_the_same_file(data_dir):
    async def run():
        async with Client(server.mcp) as mcp_client:
            empty = await mcp_client.read_resource("audit://log")
            await mcp_client.call_tool("append_audit_log", {"event": "TEST", "details": "hello"})
            filled = await mcp_client.read_resource("audit://log")
            return empty[0].text, filled[0].text

    empty_text, filled_text = asyncio.run(run())
    assert "empty" in empty_text
    assert json.loads(filled_text.splitlines()[-1])["details"] == "hello"


# --- ReAct loop with a scripted fake LLM ---------------------------------------------------

def _tool_call(call_id: str, name: str, arguments: dict):
    return SimpleNamespace(id=call_id, type="function",
                           function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


def _response(content=None, tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))])


class FakeLLM:
    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs["messages"], default=str)))
        return self.scripted.pop(0) if self.scripted else self.scripted_default()

    def scripted_default(self):
        return _response(tool_calls=[_tool_call("loop", "lookup_inventory", {"query": "bolt"})])


def run_loop(llm, max_iterations=8):
    transcript = agent_client.Transcript(None)

    async def run():
        async with Client(server.mcp) as mcp_client:
            return await agent_client.run_react(llm, "fake-model", mcp_client, "order 120 sensors",
                                                transcript, max_iterations)

    return asyncio.run(run()), transcript


def test_react_loop_executes_tools_and_feeds_observations_back(data_dir):
    llm = FakeLLM([
        _response("I need the stock level first.", [_tool_call("c1", "lookup_inventory", {"query": "SKU-4001"})]),
        _response(None, [
            _tool_call("c2", "compute_tiered_discount", {"quantity": 120, "unit_price_eur": 14.2}),
            _tool_call("c3", "append_audit_log", {"event": "ORDER_QUOTED", "details": "120 x SKU-4001"}),
        ]),
        _response("Quote ready: 120 sensors."),
    ])
    answer, transcript = run_loop(llm)

    assert answer == "Quote ready: 120 sensors."
    text = "\n".join(transcript.lines)
    assert "[Thought] I need the stock level first." in text
    assert "[Action] lookup_inventory" in text and "[Observation]" in text
    last_request = llm.requests[-1]
    tool_messages = [m for m in last_request if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["c1", "c2", "c3"]
    assert (data_dir / "audit.log").read_text(encoding="utf-8").count("\n") == 1


def test_react_loop_reports_tool_errors_as_observations(data_dir):
    llm = FakeLLM([
        _response(None, [_tool_call("c1", "compute_tiered_discount", {"quantity": 0, "unit_price_eur": 5})]),
        _response("Sorry, the quantity was invalid."),
    ])
    answer, transcript = run_loop(llm)
    assert answer.startswith("Sorry")
    assert any(line.startswith("[Observation] ERROR:") for line in transcript.lines)


def test_react_loop_stops_after_max_iterations(data_dir):
    llm = FakeLLM([])  # always asks for another tool call
    with pytest.raises(RuntimeError, match="no final answer"):
        run_loop(llm, max_iterations=3)
    assert len(llm.requests) == 3
