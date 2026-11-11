"""Which specialist agent handles each request type (the 8 types in the request-types file, T05).

The router (agents/router_agent.py) decides the type. Until it is ready, keyword_route() is a simple
stand-in so the agents can be tried end to end. It is not the router and should be removed at T30.
"""
from __future__ import annotations

import re

from agents import certification_agent, duty_agent, intake_agent, qualification_agent, sourcing_agent
from mcp_servers.cusma_tools import call_tool, normalise_tier

AGENTS = {
    "qualification_check": qualification_agent.AGENT,
    "rules_info": qualification_agent.AGENT,
    "duty_cost": duty_agent.AGENT,
    "sourcing_whatif": sourcing_agent.AGENT,
    "certification_draft": certification_agent.AGENT,
    "document_intake": intake_agent.AGENT,
}
# Types with no specialist agent here: handled by a person or a polite decline.
NOT_HANDLED_HERE = {"escalate_human", "out_of_scope"}

_KEYWORDS = [
    ("escalate_human", r"\b(sign|signature|audit|verification letter|lawyer|legal|dispute|guarantee)\w*"),
    ("certification_draft", r"\b(certificat\w*|certification)\b"),
    ("sourcing_whatif", r"\b(what if|switch|instead|source|sourcing|supplier)\b"),
    ("duty_cost", r"\b(duty|duties|tariff(?! shift)|droits?)\b"),
    ("document_intake", r"\b(invoice|declaration|attached|log this|facture)\b"),
    ("rules_info", r"\b(de minimis|rvc|regional value|tariff shift|section 338|knee band|declarations?)\b|"
                   r"\b(what is|what does|what are|explain|why)\b.*\b(rules?|cusma|usmca|origin|tariff|duty|height)\b"),
    ("qualification_check", r"\b(qualif\w*|origin|cusma|usmca|passe|parts?|materials?|components?|"
                            r"bill of materials|come from|made in|admissible|ACEUM|net cost|costs?|prices?|pay for|"
                            r"(?:do|did) (?:we|you) pay)\b"),
]


def keyword_route(question: str) -> str:
    """STAND-IN for the trained router: first matching keyword wins, otherwise out_of_scope."""
    q = question.lower()
    for label, pattern in _KEYWORDS:
        if re.search(pattern, q):
            return label
    return "out_of_scope"


def run_specialist(label: str, question: str, tier: str, model=None, language: str = "en") -> dict:
    """Run the specialist for a request type. For types handled elsewhere, record a handoff so the
    reviewer still gets a trace. Returns {"agent", "answer", "trace", "escalation", ...}."""
    tier = normalise_tier(tier)
    agent = AGENTS.get(label)
    if agent is not None:
        return agent.run(question, tier, model=model, language=language)
    if label == "out_of_scope" and tier == "C":
        return {"agent": None, "answer": "I can only help with CUSMA qualification, duty and sourcing for our boots.",
                "trace": [], "escalation": None, "tier": tier, "language": language, "steps": 0}
    handoff = {"escalate_human": "compliance_contact", "out_of_scope": "staff_on_duty",
               "document_intake": "intake_agent"}.get(label, "staff_on_duty")
    result = call_tool("create_escalation", {"reason": label, "summary": question[:300], "handoff_to": handoff}, tier)
    return {"agent": None, "answer": "I've passed your request to a colleague who can help.",
            "trace": [{"tool": "create_escalation", "args": {"reason": label, "handoff_to": handoff}, "result": result}],
            "escalation": result.get("data"), "tier": tier, "language": language, "steps": 0}
