"""Human escalation handling for the CUSMA Trade Assistant."""


from mcp_servers.cusma_tools import call_tool


COMPLIANCE_EMAIL = "northfield.compliance@outlook.com"

HUMAN_REVIEW_RESPONSE = (
    "This request needs review by a member of our Trade Compliance team. "
    f"Please email {COMPLIANCE_EMAIL} and include the relevant product SKU "
    "or purchase order number, any supporting documents, the nature of your "
    "request, and any relevant deadline. "
    "A team member will respond within two business days."
)

PRIVATE_CONTACT_REVIEW_RESPONSE = (
    "This request needs review by a member of our Trade Compliance team. "
    "I can't provide or export a bulk list of supplier contact details. "
    "A team member can review the request and determine what information "
    "can appropriately be shared."
)

def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    """Return True when any term occurs in normalized text."""
    return any(term in text for term in terms)


def _is_bulk_contact_request(question: str) -> bool:
    """Return True for requests seeking bulk supplier contact details."""

    text = (question or "").casefold()

    supplier_context = _contains_any(
        text,
        (
            "supplier",
            "suppliers",
            "sales desk",
            "sales desks",
        ),
    )

    contact_details = _contains_any(
        text,
        (
            "email",
            "emails",
            "phone",
            "phones",
            "cell",
            "cells",
            "contact",
            "contacts",
            "address",
            "addresses",
            "number",
            "numbers",
        ),
    )

    bulk_request = _contains_any(
        text,
        (
            "all",
            "every",
            "everyone",
            "whole",
            "list",
            "export",
            "names",
        ),
    )

    return supplier_context and contact_details and bulk_request


def escalate_to_human(
    question: str,
    tier: str | None = None,
) -> dict:
    """Create a compliance escalation and return the user-facing handoff."""

    result = call_tool(
        "create_escalation",
        {
            "reason": "escalate_human",
            "summary": question[:300],
            "handoff_to": "compliance_contact",
        },
        tier,
    )

    answer = (
        PRIVATE_CONTACT_REVIEW_RESPONSE
        if _is_bulk_contact_request(question)
        else HUMAN_REVIEW_RESPONSE
    )

    return {
        "agent": "escalation_agent",
        "answer": answer,
        "trace": [
            {
                "tool": "create_escalation",
                "args": {
                    "reason": "escalate_human",
                    "handoff_to": "compliance_contact",
                },
                "result": result,
            }
        ],
        "escalation": result.get("data"),
    }
