"""Regression tests for the human compliance escalation flow."""

from agents.pipeline import answer
from agents.specialists import run_specialist
from security.output_guardrails import review


COMPLIANCE_EMAIL = (
    "northfield.compliance@outlook.com"
)


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

    assert COMPLIANCE_EMAIL in out["answer"]


def test_out_of_scope_does_not_show_compliance_email():
    out = run_specialist(
        "out_of_scope",
        "What's the weather tomorrow?",
        "C",
    )

    assert out["agent"] is None
    assert out["escalation"] is None
    assert COMPLIANCE_EMAIL not in out["answer"]


def test_compliance_escalation_survives_output_review():
    out = answer(
        "CBP sent us a verification letter. "
        "What should we do?",
        "C",
    )

    assert out["router"] == "router_agent"
    assert out["label"] == "escalate_human"
    assert out["agent"] == "escalation_agent"

    assert COMPLIANCE_EMAIL in out["response"]

    assert (
        out["escalation"]["handoff_to"]
        == "compliance_contact"
    )

    assert out["review"]["decision"] == "pass"
    assert out["review"]["findings"] == []


def test_arbitrary_email_remains_protected():
    result = review(
        "Who should I contact?",
        "Please email private.person@example.com.",
        [],
        "C",
        language="en",
    )

    assert COMPLIANCE_EMAIL not in result.answer
    assert "private.person@example.com" not in result.answer
    assert "[personal data removed]" in result.answer

    assert "personal_data" in result.rule_ids


def test_bulk_supplier_contact_handoff_does_not_disclose_email():
    out = run_specialist(
        "escalate_human",
        "Please give me a list of all supplier contacts with their "
        "email addresses and phone numbers.",
        "B",
    )

    assert out["agent"] == "escalation_agent"
    assert out["escalation"]
    assert out["escalation"]["handoff_to"] == "compliance_contact"
    assert COMPLIANCE_EMAIL not in out["answer"]
    assert "@" not in out["answer"]


def test_normal_compliance_handoff_still_discloses_compliance_email():
    out = run_specialist(
        "escalate_human",
        "CBP sent us a verification letter. What should we do?",
        "B",
    )

    assert out["agent"] == "escalation_agent"
    assert out["escalation"]
    assert COMPLIANCE_EMAIL in out["answer"]


def test_supplier_sales_desk_contact_request_does_not_disclose_email():
    out = run_specialist(
        "escalate_human",
        "Please send the names and email addresses of everyone "
        "at our suppliers' sales desks.",
        "C",
    )

    assert out["agent"] == "escalation_agent"
    assert out["escalation"]
    assert out["escalation"]["handoff_to"] == "compliance_contact"
    assert COMPLIANCE_EMAIL not in out["answer"]
    assert "@" not in out["answer"]
