"""Fixes found in the T38 evaluation: boots named instead of coded, order questions on the ship date,
RVC on cost-breakdown questions, limits on every agent run, and tier C seeing only its own orders.
Offline: stand-in models only, no API key."""
import json
import re
from types import SimpleNamespace

import pytest

from agents import base, certification_agent, duty_agent, qualification_agent, sourcing_agent
from agents.base import COMMON_INSTRUCTIONS, _parse_arguments
from agents.pipeline import answer
from mcp_servers import cusma_tools as tools
from mcp_servers.cusma_tools import ORDERS, call_tool, demo_user, order_scope
from testing.test_pipeline import FakeModel, fixed


def best(name):
    r = call_tool("find_boot", {"name": name}, "C")
    return r["data"]["best"] if r["ok"] else r["error"]["code"]


# ---------------------------------------------------------------- 1. boots by name
@pytest.mark.parametrize("name,code", [
    ("Does the Glacier Pro (variant 3) qualify?", "NB-0412-V3"),
    ("the Glacier Pro V, variant 1", "NB-0413-V1"),       # 'Glacier Pro V' is its own style
    ("Glacier Pro V", "NB-0413"),
    ("Is the Glacier Lite v2 ok", "NB-0414-V2"),
    ("Tundra Pac Leather (women's) variant 2", "NB-0106-V2"),
    ("the Tundra Pac", "NB-0105"),
    ("Marsh Neo Pro", "NB-0109"),
    ("Steelguard", "NB-0102"),
    ("Urgent: Would NB0413V3 be treated as originating?", "NB-0413-V3"),   # code written another way
    ("Is the 0412-V2 still qualifying?", "NB-0412-V2"),
    ("hey what's the cusma status of nb-0413-v3?", "NB-0413-V3"),
])
def test_find_boot_resolves_names_and_odd_codes(name, code):
    assert best(name) == code


def test_find_boot_never_guesses():
    two = call_tool("find_boot", {"name": "Glacier Pro and the Steelguard"}, "C")
    assert two["ok"] and two["data"]["ambiguous"] and two["data"]["best"] is None
    assert best("the TPU safety boot") == "NOT_FOUND"           # a description, not a name: ask which boot
    assert best("Garden Ankle variant 9") == "NOT_FOUND"
    assert best("PO-8013 duty") == "NOT_FOUND"                  # order numbers and years aren't boots
    assert "Glacier Pro" in call_tool("find_boot", {"name": "Moonwalker"}, "C")["error"]["message"]


def test_every_name_only_test_question_resolves():
    rows = [json.loads(l) for l in open("data/test_holdout_human_reviewed.jsonl", encoding="utf-8") if l.strip()]
    named = [r for r in rows if r["answer_type"] == "sku_verdict" and r["sku"] and not re.search(r"NB-\d{4}", r["query"])]
    assert len(named) >= 40 and all(best(r["query"]) == r["sku"] for r in named)


def test_agents_have_find_boot_and_are_told_to_use_it():
    for agent in (qualification_agent.AGENT, duty_agent.AGENT, sourcing_agent.AGENT, certification_agent.AGENT):
        assert "find_boot" in agent.tools, agent.name
    assert "call find_boot first" in COMMON_INSTRUCTIONS and "Never make up or guess a code" in COMMON_INSTRUCTIONS


def test_named_boot_question_reaches_a_verdict():
    model = FakeModel([[("find_boot", {"name": "Glacier Pro variant 3"})],
                       [("check_origin", {"sku": "NB-0412-V3"})],
                       "Based on the rules engine (rules version 2026-10-19)."])
    out = answer("Does the Glacier Pro (variant 3) qualify?", "A", router=fixed("qualification_check"), model=model)
    assert [s["tool"] for s in out["trace"]] == ["find_boot", "check_origin"]
    assert out["result"] and out["result"]["sku"] == "NB-0412-V3"


# ---------------------------------------------------------------- 2. ship date and RVC
def test_order_questions_use_the_ship_date_and_rvc_uses_check_origin():
    assert "pass its ship_date as as_of" in COMMON_INSTRUCTIONS
    assert "get_order" in qualification_agent.AGENT.tools
    assert "check_origin is the tool that calculates the RVC" in qualification_agent.INSTRUCTIONS


# ---------------------------------------------------------------- 3. every run is bounded
def test_model_calls_have_a_timeout_one_retry_and_an_output_limit(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    m = base.OpenRouterModel(model="openai/gpt-4o-mini")
    assert m.client.timeout == base.MODEL_TIMEOUT_SECONDS <= 30 and m.client.max_retries == 1
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        msg = SimpleNamespace(content="ok", tool_calls=[])
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)
    m.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    m([{"role": "user", "content": "hi"}], [])
    assert seen["max_tokens"] == base.MAX_OUTPUT_TOKENS


def test_too_many_tool_calls_in_one_step_are_refused():
    six = [("get_boot", {"sku": "NB-0101"})] * 6
    out = answer("Tell me about NB-0101", "A", router=fixed("qualification_check"), model=FakeModel([six, "Done."]))
    codes = [(s["result"].get("error") or {}).get("code") for s in out["trace"]]
    assert codes.count("TOO_MANY_CALLS") == 6 - base.MAX_TOOL_CALLS_PER_STEP


def test_time_limit_hands_off(monkeypatch):
    monkeypatch.setattr(base, "QUESTION_TIME_LIMIT_SECONDS", -1)
    out = answer("Does NB-0101 qualify?", "A", router=fixed("qualification_check"), model=FakeModel(["never used"]))
    assert out["escalation"] and out["trace"][-1]["args"]["reason"] == "agent_time_limit"


def test_broken_tool_arguments_are_sent_back_not_crashed():
    assert _parse_arguments("{not json") == {"_unreadable_arguments": "{not json"}
    assert call_tool("get_boot", {"_unreadable_arguments": "{"}, "A")["error"]["code"] == "BAD_INPUT"


def test_a_model_that_times_out_is_handed_to_a_person():
    def slow(messages, tools_):
        raise TimeoutError("model took too long")
    out = answer("Does NB-0101 qualify?", "A", router=fixed("qualification_check"), model=slow)
    assert out["decision"] == "unavailable" and out["escalation"]


# ---------------------------------------------------------------- 4. tier C sees only its own orders
OWN = next(p for p, o in ORDERS.items() if o["customer_id"] == "CUS-201")
OTHER = next(p for p, o in ORDERS.items() if o["customer_id"] == "CUS-205")


@pytest.mark.parametrize("tool", ["get_order", "order_duty"])
def test_external_user_sees_own_orders_only(tool):
    with order_scope(["CUS-201"]):
        assert call_tool(tool, {"po_number": OWN}, "C")["ok"]
        hidden = call_tool(tool, {"po_number": OTHER}, "C")
        missing = call_tool(tool, {"po_number": "PO-9999"}, "C")
        assert not hidden["ok"] and hidden["error"]["code"] == missing["error"]["code"] == "NOT_FOUND"
        assert "PO-9999" not in hidden["error"]["message"] and OTHER in hidden["error"]["message"]   # same wording
        for tier in ("A", "B"):                                    # staff see every order
            assert call_tool(tool, {"po_number": OTHER}, tier)["ok"]


def test_broker_sees_both_customers_and_no_one_else():
    tier, customers = demo_user("broker_two_customers")
    mine = [p for p, o in ORDERS.items() if o["customer_id"] in customers]
    with order_scope(customers):
        assert tier == "C" and all(call_tool("get_order", {"po_number": p}, tier)["ok"] for p in mine)
        assert not call_tool("get_order", {"po_number": OTHER}, tier)["ok"]


def test_demo_users_and_no_customer_setting(monkeypatch):
    assert demo_user("staff_compliance") == ("A", None) and demo_user("buyer_lakehead") == ("C", ["CUS-201"])
    assert call_tool("get_order", {"po_number": OTHER}, "C")["ok"]          # no customer known: 'allow' for now
    monkeypatch.setattr(tools, "TIER_C_WITHOUT_CUSTOMER", "deny")
    assert not call_tool("get_order", {"po_number": OTHER}, "C")["ok"]


def test_signed_in_buyer_asking_about_another_customers_order_gets_not_found():
    model = FakeModel([[("order_duty", {"po_number": OTHER})], "I can't find that order."])
    out = answer(f"What's the duty on {OTHER}?", None, user="buyer_lakehead", router=fixed("duty_cost"), model=model)
    assert out["tier"] == "C" and out["trace"][0]["result"]["error"]["code"] == "NOT_FOUND"
    model = FakeModel([[("order_duty", {"po_number": OWN})], "Based on the rules engine."])
    out = answer(f"What's the duty on {OWN}?", None, user="buyer_lakehead", router=fixed("duty_cost"), model=model)
    assert out["trace"][0]["result"]["ok"]
