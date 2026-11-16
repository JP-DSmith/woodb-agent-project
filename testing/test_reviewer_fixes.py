"""Tests for the two reviewer fixes (figures such as 'TSR 20', and verdicts about a part)."""
import pytest

from security.output_guardrails import _asserted_verdicts, review

TRACE = [{
    "tool": "check_origin",
    "args": {"boot_id": "NB-0105"},
    "result": {
        "ok": True,
        "data": {"boot_id": "NB-0105", "verdict": "needs_review", "rvc": {"value": 0.7074, "threshold": 0.55}},
        "trace": [
            "Upper: TSR 20 natural rubber from Thailand is non-originating",
            "RVC = 70.74%, above 55%",
            "Net cost 38.50 for the outsole",
        ],
        "warnings": [],
        "rules_version": "2026-10-24",
    },
}]
Q = "Does NB-0105 qualify under CUSMA?"


@pytest.mark.parametrize("tier", ["A", "B", "C"])
def test_grade_in_tool_text_is_not_a_price(tier):
    r = review(Q, "NB-0105 needs review. The upper uses TSR 20 natural rubber. The RVC is 70.74%, above 55%.",
               TRACE, tier, language="en")
    assert "figures_match_trace" not in r.rule_ids
    assert r.decision != "hand_to_person"


def test_invented_plain_figure_is_still_blocked():
    r = review(Q, "NB-0105 needs review. The RVC score is 65, above the rule.", TRACE, "C", language="en")
    assert "figures_match_trace" in r.rule_ids


def test_invented_money_is_still_blocked():
    r = review(Q, "NB-0105 needs review. The outsole costs $12.40.", TRACE, "A", language="en")
    assert r.decision == "hand_to_person"


def test_cost_step_number_stays_hidden_from_tier_c():
    r = review(Q, "NB-0105 needs review. The outsole net cost is 38.50.", TRACE, "C", language="en")
    assert r.decision == "hand_to_person" or "masking_costs" in r.rule_ids


@pytest.mark.parametrize("text", [
    "The outsole compound qualifies as originating.",
    "The upper is originating under the tariff shift rule.",
    "The rubber does not qualify as originating.",
])
def test_verdict_about_a_part_is_not_the_boot_verdict(text):
    assert _asserted_verdicts(text) == set()


@pytest.mark.parametrize("text,expected", [
    ("NB-0105 qualifies under CUSMA.", {"qualifies"}),
    ("This boot does not qualify.", {"does_not_qualify"}),
    ("NB-0105 qualifies because the outsole is originating.", {"qualifies"}),
    ("NB-0105 needs review. The rubber sole is originating.", {"needs_review"}),
])
def test_verdict_about_the_boot_still_counts(text, expected):
    assert _asserted_verdicts(text) == expected


def test_wrong_boot_verdict_is_still_blocked():
    r = review(Q, "NB-0105 qualifies under CUSMA. The RVC is 70.74%.", TRACE, "C", language="en")
    assert "verdict_matches_trace" in r.rule_ids


def test_formula_figures_stay_hidden_from_restricted_tiers():
    trace = [{"tool": "check_origin", "args": {"boot_id": "NB-0412"},
              "result": {"ok": True, "data": {"boot_id": "NB-0412", "verdict": "qualifies"},
                         "trace": ["RVC = (50.55 - 17.55) / 50.55 = 65.3%"], "warnings": [], "rules_version": "2026-10-24"}}]
    r = review("Does NB-0412 qualify?", "Based on the rules, NB-0412 qualifies: RVC = (50.55 - 17.55) / 50.55 = 65.3%.", trace, "B", language="en")
    assert r.decision == "hand_to_person" and "figures_match_trace" in r.rule_ids
