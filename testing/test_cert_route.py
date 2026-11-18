"""Tests for the certificate-route reviewer fixes."""
from security.output_guardrails import review

REFUSED = [{
    "tool": "draft_certification", "args": {"po_number": "PO-8010"},
    "result": {"ok": False, "error": {"code": "NOT_SUPPORTED",
               "message": "No draft: NB-0414 is 'does_not_qualify' on the ship date 2026-11-02. Tariff shift not met."}},
}]
NOT_FOUND = [{"tool": "draft_certification", "args": {"po_number": "PO-0000"},
              "result": {"ok": False, "error": {"code": "NOT_FOUND", "message": "No order with number PO-0000"}}}]
DRAFTED = [{"tool": "draft_certification", "args": {"po_number": "PO-8005"},
            "result": {"ok": True, "data": {"status": "DRAFT - FOR SIGNATURE BY AN AUTHORIZED PERSON. NOT VALID UNTIL SIGNED."},
                       "trace": [], "warnings": [], "rules_version": "2026-10-24"}}]
Q = "Please prepare the certificate for order PO-8010."


def test_correct_refusal_is_not_blocked():
    r = review(Q, "I can't draft this certificate. The boot on that order does not qualify under CUSMA on the ship date.",
               REFUSED, "A", language="en")
    assert r.decision != "hand_to_person"
    assert "verdict_matches_trace" not in r.rule_ids


def test_refusal_gets_no_draft_label():
    r = review(Q, "I can't draft this certificate. The boot on that order does not qualify under CUSMA on the ship date.",
               REFUSED, "A", language="en")
    assert "DRAFT, FOR SIGNATURE" not in r.answer


def test_missing_order_number_reply_gets_no_label():
    r = review("Prepare a certificate please.", "Please give me the order number and I will check it.", [], "A", language="en")
    assert "DRAFT, FOR SIGNATURE" not in r.answer


def test_not_found_order_gets_no_label():
    r = review("Prepare the certificate for PO-0000.", "I can't find an order with that number.", NOT_FOUND, "A", language="en")
    assert "DRAFT, FOR SIGNATURE" not in r.answer


def test_a_real_draft_still_gets_the_label():
    r = review("Prepare the certificate for PO-8005.", "Here is the certificate information for order PO-8005.",
               DRAFTED, "A", language="en")
    assert "DRAFT, FOR SIGNATURE" in r.answer


def test_a_wrong_verdict_after_a_refusal_is_still_blocked():
    r = review(Q, "NB-0414 qualifies under CUSMA on the ship date.", REFUSED, "A", language="en")
    assert "verdict_matches_trace" in r.rule_ids


def test_tier_c_refusal_is_not_treated_as_a_draft():
    r = review(Q, "I can't draft this certificate. The boot on that order does not qualify under CUSMA on the ship date.",
               REFUSED, "C", language="en")
    assert "certification_rules" not in r.rule_ids
