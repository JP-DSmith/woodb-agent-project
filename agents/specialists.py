"""Which specialist agent handles each request type.

The router in agents/router_agent.py decides the request type.

keyword_route() remains available as a lightweight stand-in/fallback
for tests and environments where the trained router is unavailable.
"""

from __future__ import annotations

import re

from agents import (
    certification_agent,
    duty_agent,
    intake_agent,
    qualification_agent,
    sourcing_agent,
)
from agents.escalation_agent import escalate_to_human
from mcp_servers.cusma_tools import (
    call_tool,
    normalise_tier,
)


AGENTS = {
    "qualification_check": qualification_agent.AGENT,
    "rules_info": qualification_agent.AGENT,
    "duty_cost": duty_agent.AGENT,
    "sourcing_whatif": sourcing_agent.AGENT,
    "certification_draft": certification_agent.AGENT,
    "document_intake": intake_agent.AGENT,
}


# These request types do not have a normal specialist agent.
NOT_HANDLED_HERE = {
    "escalate_human",
    "out_of_scope",
}


_KEYWORDS = [
    (
        "escalate_human",
        (
            r"\b(sign|signature|audit|verification letter|"
            r"lawyer|legal|dispute|guarantee)\w*"
        ),
    ),
    (
        "certification_draft",
        r"\b(certificat\w*|certification)\b",
    ),
    (
        "sourcing_whatif",
        (
            r"\b(what if|switch|instead|source|"
            r"sourcing|supplier)\b"
        ),
    ),
    (
        "duty_cost",
        r"\b(duty|duties|tariff(?! shift)|droits?)\b",
    ),
    (
        "document_intake",
        (
            r"\b(invoice|declaration|attached|"
            r"log this|facture)\b"
        ),
    ),
    (
        "rules_info",
        (
            r"\b(de minimis|rvc|regional value|tariff shift|"
            r"section 338|knee band|declarations?)\b|"
            r"\b(what is|what does|what are|explain|why)\b"
            r".*\b(rules?|cusma|usmca|origin|tariff|duty|height)\b"
        ),
    ),
    (
        "qualification_check",
        (
            r"\b(qualif\w*|origin|cusma|usmca|passe|"
            r"parts?|materials?|components?|bill of materials|"
            r"come from|made in|admissible|ACEUM|net cost|"
            r"costs?|prices?|pay for|(?:do|did) (?:we|you) pay)\b"
        ),
    ),
]


def keyword_route(question: str) -> str:
    """Return the first matching fallback intent."""

    q = question.lower()

    for label, pattern in _KEYWORDS:
        if re.search(pattern, q):
            return label

    return "out_of_scope"


def run_specialist(
    label: str,
    question: str,
    tier: str,
    model=None,
    language: str = "en",
) -> dict:
    """Run the specialist associated with a routed request type.

    Request types that require human review are delegated to the
    dedicated escalation agent. Other unhandled types retain their
    existing handoff behavior so the reviewer receives a trace.
    """

    tier = normalise_tier(tier)

    agent = AGENTS.get(label)

    # --------------------------------------------------------
    # Normal specialist agents
    # --------------------------------------------------------

    if agent is not None:
        return agent.run(
            question,
            tier,
            model=model,
            language=language,
        )

    # --------------------------------------------------------
    # Human escalation
    #
    # The dedicated escalation agent owns the compliance
    # contact details and user-facing human-review response.
    # --------------------------------------------------------

    if label == "escalate_human":
        result = escalate_to_human(
            question,
            tier,
        )

        return {
            **result,
            "tier": tier,
            "language": language,
            "steps": 0,
        }

    # --------------------------------------------------------
    # Tier-C out-of-scope requests
    # --------------------------------------------------------

    if label == "out_of_scope" and tier == "C":
        return {
            "agent": None,
            "answer": (
                "I can only help with CUSMA qualification, "
                "duty and sourcing for our boots."
            ),
            "trace": [],
            "escalation": None,
            "tier": tier,
            "language": language,
            "steps": 0,
        }

    # --------------------------------------------------------
    # Other request types handled through a normal handoff
    # --------------------------------------------------------

    handoff = {
        "out_of_scope": "staff_on_duty",
        "document_intake": "intake_agent",
    }.get(
        label,
        "staff_on_duty",
    )

    result = call_tool(
        "create_escalation",
        {
            "reason": label,
            "summary": question[:300],
            "handoff_to": handoff,
        },
        tier,
    )

    return {
        "agent": None,
        "answer": (
            "I've passed your request to a colleague who can help."
        ),
        "trace": [
            {
                "tool": "create_escalation",
                "args": {
                    "reason": label,
                    "handoff_to": handoff,
                },
                "result": result,
            }
        ],
        "escalation": result.get("data"),
        "tier": tier,
        "language": language,
        "steps": 0,
    }
