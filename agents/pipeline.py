"""The whole system, end to end (T30): one function the app calls for every message.

    message (English) + tier from login
      -> input guard  (security/input_guardrails.py; checks the original AND the English text)
      -> router       (agents/llm_router.py, falling back to agents/router_agent.py: reads the question TEXT
                       ONLY, never the tier; CUSMA_ROUTER=onnx uses the previous router)
      -> specialist agent + tools  (agents/specialists.py; the tier controls what tools may show)
      -> reviewer     (security/output_guardrails.py; checks every number and name against the trace)
      -> reply (English; the app translates it back)

A supplier document (input_type="document") skips the router: the guard checks it in document
mode and it goes straight to the intake agent.

Usage from the app (web/chat_services.py):
    from agents.pipeline import answer
    out = answer(english_text, tier, original=text_as_typed, language=detected_language)
    out["response"]   -> reply text (English)
    out["result"]     -> qualification card for the portal, or None
    out["seconds"], out["usage"] -> time and model use for this message (for the evaluation runner)
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Callable

from agents.base import measure_usage
from agents.specialists import keyword_route, run_specialist
from mcp_servers.cusma_tools import SUPPLIERS, call_tool, demo_user, eng, normalise_tier, order_scope
from security.input_guardrails import check_with_translation
from security.output_guardrails import review

logger = logging.getLogger(__name__)


LABELS = {"qualification_check", "rules_info", "duty_cost", "sourcing_whatif", "certification_draft",
          "escalate_human", "document_intake", "out_of_scope"}

_router_cache: dict = {}


_BINDING_ASSURANCE = re.compile(
    r"(?is)"
    r"\b(?:guarantee|guaranteed|promise)\b"
    r".{0,160}"
    r"\b(?:"
    r"cbp|customs|verification|verify|audit|"
    r"duty|duties|no duty|zero duty|"
    r"in writing|written|will pass|hold up"
    r")\b"
    r"|"
    r"\b(?:"
    r"cbp|customs|verification|verify|audit|"
    r"duty|duties|no duty|zero duty|"
    r"in writing|written|will pass|hold up"
    r")\b"
    r".{0,160}"
    r"\b(?:guarantee|guaranteed|promise)\b"
)


def _requires_binding_assurance_review(message: str) -> bool:
    """True when the user asks for a binding guarantee in a high-risk context."""
    return bool(_BINDING_ASSURANCE.search(message or ""))


def default_router() -> tuple[Callable[[str], str], str]:
    """The router the app uses, chosen by CUSMA_ROUTER:
        llm (default)  the prompted LLM router (agents/llm_router.py); if the model can't be reached, each
                       question is routed by the ONNX router instead
        onnx           the previous router (agents/router_agent.py, nearest neighbour); "trained" also works
        keyword        the offline keyword stand-in (tests, no model files)
    Without the ONNX model files, the keyword stand-in takes the ONNX router's place."""
    if "fn" in _router_cache:
        return _router_cache["fn"], _router_cache["name"]
    choice = os.environ.get("CUSMA_ROUTER", "llm").strip().lower()
    onnx, onnx_name = keyword_route, "keyword_stand_in"
    if choice != "keyword":
        try:
            from agents import router_agent
            route = getattr(router_agent, "route_user_input", None)
            if route is not None:
                onnx, onnx_name = route, "router_agent"
        except Exception:  # model/libraries not available: fall back, and say so in the output
            pass
    if choice in ("keyword", "onnx", "trained"):
        fn, name = onnx, onnx_name
    else:
        from agents.llm_router import make_router
        fn, name = make_router(fallback=onnx), "llm_router"
    _router_cache.update(fn=fn, name=name)
    return fn, name


def _label(raw) -> str:
    """Accept a plain label or the router's dict ({'predicted_class': ...})."""
    label = raw.get("predicted_class") if isinstance(raw, dict) else raw
    return label if label in LABELS else "escalate_human"


def result_card(trace: list[dict]) -> dict | None:
    """The portal's qualification card, from the last successful check_origin in the trace."""
    for step in reversed(trace or []):
        res = step.get("result") or {}
        if step.get("tool") == "check_origin" and res.get("ok"):
            d = res["data"]
            return {"sku": d["sku"], "verdict": d["verdict"], "hs_code": d.get("hs_code"),
                    "rvc": d["rvc"]["value"], "rvc_threshold": d["rvc"]["threshold"],
                    "tariff_shift_passed": d["tariff_shift"]["passed"],
                    "review_reason": d.get("escalate_reason"), "rules_version": res.get("rules_version")}
    return None


def answer(message: str, tier: str | None, *, original: str | None = None, language: str = "en",
           input_type: str = "message", router: Callable | None = None, model=None,
           customer_ids: list[str] | None = None, user: str | None = None) -> dict:
    """Answer one message (see _answer), and add what it took: "seconds" (wall time, every step) and
    "usage" (model calls, tokens and cost, from agents/base.py). Neither is shown to the user.

    Who is asking: pass the tier, plus customer_ids for an external user (they then see only those
    customers' orders), or pass user=<id from config/demo_users.json>, which sets both."""
    if user is not None:
        tier, customer_ids = demo_user(user)   # unknown or invalid user: tier C, no customers
    start = time.perf_counter()
    with measure_usage() as meter, order_scope(customer_ids):
        out = _answer(message, tier, original=original, language=language, input_type=input_type,
                      router=router, model=model)
    return {**out, "seconds": round(time.perf_counter() - start, 3), "usage": meter.as_dict()}


def _answer(message: str, tier: str | None, *, original: str | None = None, language: str = "en",
            input_type: str = "message", router: Callable | None = None, model=None) -> dict:
    """Answer one message. `message` is the English text; `original` is the text as typed (if translated).
    Returns {"response", "decision", "label", "agent", "router", "tier", "trace", "escalation",
             "guard", "review", "result", "draft"}. Only "response" and "result" are for the user."""
    tier = normalise_tier(tier)
    out = {"tier": tier, "language": language, "label": None, "agent": None, "router": None,
           "trace": [], "escalation": None, "guard": None, "review": None, "result": None, "draft": None}

    # 1. Input guard: the text as typed, then its English translation.
    mode = "document" if input_type == "document" else "message"
    guard = check_with_translation(original or message, message if original else None, mode=mode)
    out["guard"] = {"decision": guard.decision, "rule_id": guard.rule_id, "category": guard.category,
                    "flags": list(guard.flags)}
    if guard.decision == "block":
        return {**out, "response": guard.reply or "I can't help with that request.", "decision": "blocked"}
    if guard.decision == "flag_security":
        esc = call_tool("create_escalation", {"reason": "security_flag", "handoff_to": "compliance_contact",
                                              "summary": f"{guard.category or ''} {guard.rule_id or ''}".strip()}, tier)
        return {**out, "response": guard.reply or "I've passed this to a colleague to check.",
                "decision": "flagged", "escalation": esc.get("data"),
                "trace": [{"tool": "create_escalation", "args": {"reason": "security_flag"}, "result": esc}]}

    # 2. Router: the question text only. Documents go straight to intake.
    if input_type == "document":
        label, out["router"] = "document_intake", "document_upload"
    else:
        fn, out["router"] = (router, "given") if router else default_router()
        label = _label(fn(message))

        # Binding guarantees about customs, verification, duty, audits,
        # or written assurances require human review regardless of the
        # semantic router's nearest intent.
        if _requires_binding_assurance_review(message):
            label = "escalate_human"

    out["label"] = label

    # 3. Specialist agent with its tools (the tier is applied inside the tools).
    try:
        spec = run_specialist(label, message, tier, model=model, language=language)
    except Exception as err:  # e.g. OPENROUTER_API_KEY not set, or the model service is down
        logger.error(
            "Specialist agent failed: label=%s tier=%s error_type=%s error=%s",
            label,
            normalise_tier(tier),
            type(err).__name__,
            str(err)[:500],
        )
        esc = call_tool("create_escalation", {"reason": "assistant_unavailable", "summary": message[:300]}, tier)
        return {**out, "response": "The assistant isn't available right now, so I've passed your request "
                                   "to a colleague.", "decision": "unavailable", "escalation": esc.get("data"),
                "error": f"{type(err).__name__}: {err}"[:200],
                "trace": [{"tool": "create_escalation", "args": {"reason": "assistant_unavailable"}, "result": esc}]}
    out.update(agent=spec["agent"], trace=spec["trace"], escalation=spec["escalation"])
    out["draft"] = spec["answer"]          # the agent's own answer, before the reviewer (for testing; never shown)

    # 4. Reviewer: checks the reply against the trace, tier, and user's language.
    res = review(message, spec["answer"], spec["trace"], tier, language=language,
                 catalogue_ids=list(eng.SKUS), supplier_names=[s["name"] for s in SUPPLIERS.values()])
    out["review"] = {"decision": res.decision, "rescued": res.rescued,
                     "findings": [{"rule_id": f.rule_id, "action": f.action, "detail": f.detail} for f in res.findings]}
    if res.decision == "hand_to_person" and not out["escalation"]:
        e = res.escalation or {}
        esc = call_tool("create_escalation", {"reason": e.get("reason", "reviewer_handoff"),
                                              "summary": e.get("summary", message[:300])}, tier)
        out["escalation"] = esc.get("data")
        out["trace"] = out["trace"] + [{"tool": "create_escalation", "args": {"reason": "reviewer_handoff"}, "result": esc}]

    out["result"] = result_card(out["trace"]) if res.decision != "hand_to_person" else None
    return {**out, "response": res.answer, "decision": res.decision}
