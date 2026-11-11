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

    return {
        "agent": "escalation_agent",
        "answer": HUMAN_REVIEW_RESPONSE,
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
