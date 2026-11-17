"""Prompted LLM router: asks a small model (gpt-4o-mini by default) which of the 8 routes a question belongs to.

Chosen in the router study (validation results) over the nearest-neighbour ONNX router (agents/router_agent.py)
and a fine-tuned MiniLM. The committed evidence for the router as it is now (accuracy per question set, time and
cost per question, against the ONNX router) is evaluation/evidence/router_comparison_llm_vs_onnx.md, made with
evaluation/compare_routers.py (routers "llm=agents.llm_router:route" and "onnx=live", validation and test sets);
the predictions for each question id are in router_comparison_llm_vs_onnx_predictions.csv.

The prompt is built only from the route descriptions in config/request_types_and_personas.json ("description"
and "not_this"): no example sentences and no test questions. The reply must name exactly one route; anything
else goes to a person (escalate_human). If the model can't be reached (no key, service down), the question is
routed by the fallback router instead (the ONNX router in the pipeline), so routing never stops.

    from agents.llm_router import make_router
    route = make_router(fallback=onnx_route)
    route("How much duty is payable on order PO-8016?")    # -> "duty_cost"

Settings: CUSMA_ROUTER_MODEL (default openai/gpt-4o-mini). Which router the pipeline uses: CUSMA_ROUTER
(llm = this one, the default; onnx = the previous router; keyword = the offline stand-in).
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "request_types_and_personas.json"
LABELS = ["qualification_check", "rules_info", "sourcing_whatif", "duty_cost", "certification_draft",
          "document_intake", "escalate_human", "out_of_scope"]
DEFAULT_MODEL = "openai/gpt-4o-mini"

PROMPT = """You route questions sent to a CUSMA (USMCA) rules-of-origin assistant for Northfield Rubber Footwear,
a Canadian maker of rubber boots. Choose the ONE route that fits the question best.

Routes:
{routes}


Answer with the route name only, exactly as written above, and nothing else."""


def route_descriptions(path: Path = CONFIG) -> str:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    lines = []
    for rt in cfg["request_types"]:
        line = f"- {rt['id']}: {rt['description']}"
        if rt.get("not_this"):
            line += f" (Not this: {rt['not_this']})"
        lines.append(line)
    return "\n".join(lines)


def system_prompt() -> str:
    return PROMPT.format(routes=route_descriptions())


def parse(text: str | None) -> str:
    """The route named in the reply. Exactly one route name, or escalate_human (no route, or more than one)."""
    found = [label for label in LABELS if re.search(rf"\b{label}\b", (text or "").lower())]
    return found[0] if len(found) == 1 else "escalate_human"


def make_router(model=None, fallback: Callable[[str], object] | None = None, model_id: str | None = None):
    """A router function: question text in, route label out. `model` can be swapped (tests); by default an
    OpenRouterModel at temperature 0 is created on first use. If the model call fails, `fallback` routes the
    question (or, without one, it goes to a person)."""
    state = {"model": model}
    system = system_prompt()
    model_id = model_id or os.environ.get("CUSMA_ROUTER_MODEL") or DEFAULT_MODEL

    def route(question: str):
        try:
            if state["model"] is None:
                from agents.base import OpenRouterModel
                state["model"] = OpenRouterModel(model=model_id, temperature=0.0)
            reply = state["model"]([{"role": "system", "content": system},
                                    {"role": "user", "content": f"Question: {question}\nRoute:"}], [])
        except Exception as err:   # no key, timeout, service down: keep routing with the fallback
            logger.warning("LLM router unavailable (%s: %s); using the fallback router",
                           type(err).__name__, str(err)[:200])
            return fallback(question) if fallback else "escalate_human"
        return parse((reply or {}).get("content"))

    route.__name__ = "llm_router"
    return route


_default: dict = {}


def route(question: str) -> str:
    """The LLM router on its own, with no fallback (a failed call goes to escalate_human), for measuring it:
    evaluation/compare_routers.py --router llm=agents.llm_router:route"""
    if "fn" not in _default:
        _default["fn"] = make_router()
    return _default["fn"](question)
