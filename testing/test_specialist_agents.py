"""T27 tests: CUSMA tools and specialist agents. Offline: a scripted fake model, no API key needed."""
import json

import pytest

from agents.base import Agent
from agents.specialists import AGENTS, keyword_route, run_specialist
from mcp_servers.cusma_tools import TOOLS, call_tool, normalise_tier


class FakeModel:
    """Plays back a list of replies. A reply is a str (final answer) or a list of (tool, args) calls."""

    def __init__(self, script):
        self.script, self.seen_tools = list(script), []

    def __call__(self, messages, tools):
        self.seen_tools.append([t["function"]["name"] for t in tools])
        step = self.script.pop(0) if self.script else "done"
        if isinstance(step, str):
            return {"content": step, "tool_calls": []}
        return {"content": None, "tool_calls": [{"id": f"c{i}", "name": n, "arguments": a} for i, (n, a) in enumerate(step)]}


def text(obj):
    return json.dumps(obj, default=str)


# ------------------------------------------------------------------ tools
def test_tiers_accept_dataset_names():
    assert [normalise_tier(t) for t in ("A", "b", "T1", "T2", "T3", None, "admin")] == ["A", "B", "A", "B", "C", "C", "C"]


def test_check_origin_shows_costs_only_to_tier_a():
    a = call_tool("check_origin", {"sku": "NB-0101"}, "A")
    c = call_tool("check_origin", {"sku": "NB-0101"}, "C")
    assert a["ok"] and c["ok"] and a["data"]["verdict"] == c["data"]["verdict"]
    assert "Net cost =" in text(a["trace"]) and "CAD" in text(a["trace"])
    assert "[confidential]" in text(c["trace"]) and " CAD" not in text(c["trace"])


@pytest.mark.parametrize("tier", ["B", "C"])
def test_no_cost_fields_for_b_and_c(tier):
    for name, args in [("get_materials", {"sku": "NB-0101"}), ("what_if", {"sku": "NB-0101", "changes": [{"line": 5, "origin_country": "MX"}]})]:
        r = call_tool(name, args, tier)
        assert r["ok"], r
        for key in ("cost_cad", "net_cost_cad", "vnm_cad", "labour_cad"):
            assert f'"{key}"' not in text(r)


def test_tier_b_sees_supplier_names_but_no_costs():
    lines = call_tool("get_materials", {"sku": "NB-0101"}, "B")["data"]
    assert all(l.get("supplier_name") and l.get("supplier_id") for l in lines)
    assert not any("cost_cad" in l for l in lines)
    assert call_tool("find_alternative_suppliers", {"sku": "NB-0101", "line_no": 5}, "B")["ok"]
    assert "Net cost = [confidential]" in text(call_tool("check_origin", {"sku": "NB-0101"}, "B")["trace"])


def test_tier_c_sees_no_supplier_names_or_alternatives():
    r = call_tool("get_materials", {"sku": "NB-0101"}, "C")
    assert "SUP-" not in text(r["data"]) and "supplier_name" not in text(r)
    assert call_tool("find_alternative_suppliers", {"sku": "NB-0101", "line_no": 5}, "C")["error"]["code"] == "NOT_ALLOWED"
    assert call_tool("find_alternative_suppliers", {"sku": "NB-0101", "line_no": 5}, "A")["ok"]


def test_unknown_boot_and_order():
    assert call_tool("check_origin", {"sku": "NB-0199"}, "A")["error"]["code"] == "NOT_FOUND"
    assert not call_tool("get_order", {"po_number": "PO-9999"}, "A")["ok"]


def test_order_duty_and_section_338():
    assert call_tool("order_duty", {"po_number": "PO-8017"}, "A")["data"]["duty_payable_usd"] == pytest.approx(7410)
    d = call_tool("import_duty", {"sku": "NB-0106", "quantity": 1000, "unit_price_usd": 12}, "A")["data"]
    assert any(t["name"] == "Section 338" for t in d.get("special_tariffs", []))


def test_section_338_rule_comes_from_the_rules_file():
    text = call_tool("lookup_rule", {"topic": "section_338"}, "C")["data"]["section_338"]
    assert "6403.91.90" in text and "50%" in text
    from mcp_servers.cusma_tools import eng
    if "6403.91.60" in eng.RULES["us_duty"]:                    # men's leather code (rules update, PR #30)
        assert "Not on the Section 338 list" in text and "6403.91.60" in text


def test_certification_draft_rules():
    a = call_tool("draft_certification", {"po_number": "PO-8005"}, "A")
    assert a["ok"] and "DRAFT" in text(a["data"])
    assert call_tool("draft_certification", {"po_number": "PO-8005"}, "C")["error"]["code"] == "NOT_ALLOWED"
    assert call_tool("draft_certification", {"po_number": "PO-8017"}, "A")["error"]["code"] == "NOT_SUPPORTED"


def test_model_cannot_set_tier_or_bad_args():
    r = call_tool("get_materials", {"sku": "NB-0101", "tier": "A"}, "C")
    assert "cost_cad" not in text(r)
    assert call_tool("check_origin", {"boot": "NB-0101"}, "A")["error"]["code"] == "BAD_INPUT"


# ------------------------------------------------------------------ agents
def test_every_agent_tool_exists():
    for agent in AGENTS.values():
        assert agent.tools and all(t in TOOLS for t in agent.tools)


def test_agent_answers_from_tool_and_returns_trace():
    model = FakeModel([[("check_origin", {"sku": "NB-0101"})], "NB-0101 qualifies."])
    out = AGENTS["qualification_check"].run("Does NB-0101 qualify?", "B", model=model)
    assert out["answer"] == "NB-0101 qualifies." and out["agent"] == "qualification_agent"
    assert [s["tool"] for s in out["trace"]] == ["check_origin"]
    assert set(out["trace"][0]) == {"tool", "args", "result"}
    assert model.seen_tools[0] == AGENTS["qualification_check"].tools


def test_agent_cannot_use_another_agents_tool():
    model = FakeModel([[("draft_certification", {"po_number": "PO-8005"})], "Sorry."])
    out = AGENTS["duty_cost"].run("Draft a certificate for PO-8005", "A", model=model)
    assert out["trace"][0]["result"]["error"]["code"] == "NOT_ALLOWED"


def test_model_supplied_tier_is_ignored():
    model = FakeModel([[("get_materials", {"sku": "NB-0101", "tier": "A"})], "ok"])
    out = AGENTS["sourcing_whatif"].run("What are the parts of NB-0101?", "T3", model=model)
    assert out["tier"] == "C" and "tier" not in out["trace"][0]["args"] and "cost_cad" not in text(out["trace"])


def test_step_limit_hands_off():
    model = FakeModel([[("get_boot", {"sku": "NB-0101"})]] * 10)
    out = Agent("loop", "x", ["get_boot", "create_escalation"]).run("?", "A", model=model, max_steps=3)
    assert out["escalation"] and out["trace"][-1]["tool"] == "create_escalation"


def test_certification_flow_hands_to_signer():
    model = FakeModel([[("draft_certification", {"po_number": "PO-8005"})],
                       [("create_escalation", {"reason": "certification_drafted", "summary": "PO-8005",
                                               "handoff_to": "authorized_signer"})],
                       "Here is a DRAFT for an authorized person to sign."])
    out = run_specialist("certification_draft", "Draft a certificate for PO-8005", "A", model=model)
    assert out["escalation"] and out["escalation"]["handoff_to"] == "authorized_signer"


def test_escalate_human_uses_escalation_agent():
    out = run_specialist(
        "escalate_human",
        "Customs is auditing us",
        "B",
    )

    assert out["agent"] == "escalation_agent"
    assert out["escalation"]

    assert (
        out["escalation"]["handoff_to"]
        == "compliance_contact"
    )

    assert (
        out["trace"][0]["tool"]
        == "create_escalation"
    )

    assert (
        "northfield.compliance@outlook.com"
        in out["answer"]
    )


def test_out_of_scope_does_not_show_compliance_email():
    out = run_specialist(
        "out_of_scope",
        "What's the weather tomorrow?",
        "C",
    )

    assert out["agent"] is None
    assert out["escalation"] is None

    assert (
        "northfield.compliance@outlook.com"
        not in out["answer"]
    )


def test_keyword_route_stand_in():
    assert keyword_route("How much duty for order PO-8036?") == "duty_cost"
    assert keyword_route("Does sku NB-0413-V1 qualify under CUSMA?") == "qualification_check"
    assert keyword_route("Please sign the certificate") == "escalate_human"
    assert keyword_route("What's the weather?") == "out_of_scope"


# ------------------------------------------------------------------ hand-off to the reviewer (T28)
def test_trace_passes_the_reviewer():
    guard = pytest.importorskip("security.output_guardrails")
    if not hasattr(guard, "review"):
        pytest.skip("reviewer (T28) not merged yet")
    model = FakeModel([[("check_origin", {"sku": "NB-0101"})], "x"])
    out = AGENTS["qualification_check"].run("Does NB-0101 qualify?", "C", model=model)
    rvc = out["trace"][0]["result"]["data"]["rvc"]["value"]
    answer = f"Yes, NB-0101 qualifies under CUSMA (RVC {rvc:.1%} vs 55%). Based on the rules engine."
    res = guard.review("Does NB-0101 qualify?", answer, out["trace"], "C", language="en")
    assert res.decision == "pass", res
    wrong = guard.review("Does NB-0101 qualify?", answer.replace(f"{rvc:.1%}", "62.0%"), out["trace"], "C", language="en")
    assert wrong.decision != "pass", wrong
