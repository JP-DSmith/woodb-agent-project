"""Rule topics for lookup_rule (config/rule_topics.json). Offline: no model, no network."""
import json
import re

from agents import qualification_agent
from agents.pipeline import answer
from mcp_servers.cusma_tools import RULE_TOPICS_FILE, TOOLS, call_tool
from security.output_guardrails import review
from testing.test_pipeline import FakeModel, fixed

NEW_TOPICS = ["certification_format", "certification_contents", "blanket_certification", "record_keeping",
              "correcting_errors", "low_value_shipments", "packaging_and_relabelling", "confidential_information",
              "proving_origin", "cost_changes"]


def rule(topic, tier="C"):
    return [{"tool": "lookup_rule", "args": {"topic": topic}, "result": call_tool("lookup_rule", {"topic": topic}, tier)}]


def test_every_topic_has_summary_text_and_source():
    topics = json.loads(RULE_TOPICS_FILE.read_text(encoding="utf-8"))["topics"]
    for name in NEW_TOPICS + ["rvc", "tariff_shift", "de_minimis", "declarations", "review_band", "section_338"]:
        assert name in topics, name
        assert topics[name]["summary"] and topics[name]["text"] and topics[name]["source"], name
    for name in NEW_TOPICS:
        assert re.search(r"CUSMA (Article|Annex)", topics[name]["source"]), f"{name}: cite the CUSMA article"


def test_every_topic_loads_and_shows_its_source():
    for name in json.loads(RULE_TOPICS_FILE.read_text(encoding="utf-8"))["topics"]:
        r = call_tool("lookup_rule", {"topic": name}, "C")
        assert r["ok"] and "(Source: " in r["data"][name] and "{" not in r["data"][name], name
    everything = call_tool("lookup_rule", {"topic": "all"}, "C")["data"]
    assert set(NEW_TOPICS) <= set(everything)


def test_the_agent_sees_every_topic_in_the_tool_description():
    description = TOOLS["lookup_rule"][1]
    assert all(t in description for t in NEW_TOPICS) and "do not answer from memory" in description


def test_known_facts_are_in_the_texts():
    text = lambda t: call_tool("lookup_rule", {"topic": t}, "C")["data"][t]
    assert "12 months" in text("blanket_certification") and "4 years" in text("blanket_certification")
    assert "5 years" in text("record_keeping") and "6 years" in text("record_keeping")
    assert "electronic or digital signature" in text("certification_format")
    assert "US$1,000" in text("low_value_shipments")
    assert "Article 5.8" in text("record_keeping")


def test_unknown_topic_tells_the_agent_to_hand_off():
    r = call_tool("lookup_rule", {"topic": "anti_dumping"}, "A")
    assert r["error"]["code"] == "NOT_FOUND" and "create_escalation" in r["error"]["message"]


def test_reviewer_accepts_published_figures_for_tier_c():
    # Before rule_figures, "12 months" and "30 days" were blocked for tiers that can't see costs.
    for topic, reply in [("blanket_certification", "A blanket period can be up to 12 months; a certification is valid for 4 years."),
                         ("proving_origin", "You get at least 30 days to answer a written request from customs.")]:
        assert review("q", reply, rule(topic), "C", language="en").decision == "pass", topic


def test_reviewer_still_catches_an_invented_figure():
    out = review("q", "No certification is needed for imports of US$2,500 or less.", rule("low_value_shipments"), "C",
                 language="en")
    assert out.decision == "hand_to_person"


def test_agent_is_told_to_hand_off_uncovered_rules():
    assert "never answer a rule question from memory" in qualification_agent.INSTRUCTIONS
    assert "rule_not_covered" in qualification_agent.INSTRUCTIONS


def test_uncovered_rule_question_is_handed_to_a_person():
    model = FakeModel([[("lookup_rule", {"topic": "anti_dumping"})],
                       [("create_escalation", {"reason": "rule_not_covered", "summary": "anti-dumping duties"})],
                       "I don't have that rule, so I've passed your question to a colleague."])
    out = answer("Do anti-dumping duties apply to our boots?", "B", router=fixed("rules_info"), model=model)
    assert out["escalation"] and out["escalation"]["reason"] == "rule_not_covered"
