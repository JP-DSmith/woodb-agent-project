"""The prompted LLM router (agents/llm_router.py) and how the pipeline chooses its router. No model calls."""
import json
import os

import pytest

from agents import llm_router, pipeline
from evaluation import run_evaluation


class Fake:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, []

    def __call__(self, messages, tools):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return {"content": self.reply, "tool_calls": []}


def test_prompt_uses_only_the_route_descriptions():
    cfg = json.loads(llm_router.CONFIG.read_text(encoding="utf-8"))
    prompt = llm_router.system_prompt()
    for rt in cfg["request_types"]:
        assert f"- {rt['id']}: {rt['description']}" in prompt
        for example in rt.get("examples", []) if isinstance(rt.get("examples"), list) else []:
            assert example not in prompt                           # no example sentences
    assert "Examples" not in prompt and "Question:" not in prompt  # no example questions at all
    assert set(llm_router.LABELS) == pipeline.LABELS


@pytest.mark.parametrize("reply,label", [
    ("duty_cost", "duty_cost"),
    ("  Duty_Cost\n", "duty_cost"),
    ("Route: rules_info", "rules_info"),
    ("rules_info or duty_cost", "escalate_human"),       # more than one route
    ("I'm not sure", "escalate_human"),                  # no route
    ("", "escalate_human"),
    (None, "escalate_human"),
])
def test_reply_must_name_exactly_one_route(reply, label):
    assert llm_router.parse(reply) == label


def test_routes_with_the_model_and_sends_only_the_question():
    model = Fake("certification_draft")
    route = llm_router.make_router(model=model)
    assert route("Prepare a certificate of origin for order PO-8005.") == "certification_draft"
    system, user = model.calls[0]
    assert system["role"] == "system" and user["content"] == "Question: Prepare a certificate of origin for order PO-8005.\nRoute:"


def test_model_failure_uses_the_fallback_router():
    route = llm_router.make_router(model=Fake(error=TimeoutError("down")), fallback=lambda q: "duty_cost")
    assert route("How much duty?") == "duty_cost"
    assert llm_router.make_router(model=Fake(error=RuntimeError("x")))("How much duty?") == "escalate_human"


def test_missing_key_falls_back_instead_of_crashing(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    route = llm_router.make_router(fallback=lambda q: "rules_info")
    assert route("What is RVC?") == "rules_info"


def test_pipeline_uses_the_llm_router_by_default(monkeypatch):
    monkeypatch.delenv("CUSMA_ROUTER", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)      # no key: it must fall back, not call anything
    pipeline._router_cache.clear()
    fn, name = pipeline.default_router()
    assert name == "llm_router" and pipeline._label(fn("How much duty for order PO-8017?")) in pipeline.LABELS


@pytest.mark.parametrize("setting,names", [("onnx", {"router_agent", "keyword_stand_in"}),
                                           ("trained", {"router_agent", "keyword_stand_in"}),
                                           ("keyword", {"keyword_stand_in"}),
                                           ("LLM", {"llm_router"})])
def test_router_can_be_chosen_with_cusma_router(monkeypatch, setting, names):
    monkeypatch.setenv("CUSMA_ROUTER", setting)
    pipeline._router_cache.clear()
    assert pipeline.default_router()[1] in names


def test_pipeline_answer_with_the_llm_router(monkeypatch):
    class Specialist:
        def __call__(self, messages, tools):
            return {"content": "RVC is the share of value made in North America.", "tool_calls": []}
    monkeypatch.setattr(pipeline, "default_router",
                        lambda: (llm_router.make_router(model=Fake("rules_info")), "llm_router"))
    out = pipeline.answer("What is RVC?", "A", model=Specialist())
    assert out["router"] == "llm_router" and out["label"] == "rules_info"


def test_evaluation_runner_can_still_run_the_previous_router():
    args = run_evaluation.build_parser().parse_args(["--router", "onnx"])
    assert args.router == "onnx"
    assert run_evaluation.build_parser().parse_args([]).router == "current"


# ---- supplier / origin questions belong to sourcing_whatif (review of #72) ------------------------------------
# Questions about who supplies a part today, or where it comes from, must go to the sourcing agent, as the ONNX
# router already does (a Portal regression case). These are written for these tests: none of them is in a
# held-out set, and none is in the router prompt (the prompt holds only the route definitions).
SUPPLIER_QUESTIONS = {
    "who supplies a component": ["Who supplies the outsole for NB-0101?", "Which supplier makes the lining for NB-0412?",
                                 "Who do we buy the toe cap for NB-0413 from?"],
    "where a component comes from": ["Where does the outsole for NB-0101 come from?",
                                     "What country is the rubber in NB-0102 sourced from?",
                                     "Where is the NB-0414 upper material made?"],
    "supplier and origin": ["Which supplier makes the outsole for NB-0101, and where is it made?",
                            "Who supplies the NB-0412 lining and which country is it made in?"],
    "parts from outside North America": ["Which parts of NB-0101 come from outside North America?",
                                         "Are any components of NB-0413 sourced from outside North America?"],
}
ALL_SUPPLIER_QUESTIONS = [q for qs in SUPPLIER_QUESTIONS.values() for q in qs]
PORTAL_CASES = ["Which supplier makes the outsole for NB-0101?",
                "Which supplier makes the outsole for NB-0101, and where is it made?",
                "Who supplies the outsole for NB-0101?", "Where does the outsole for NB-0101 come from?",
                "Which parts of NB-0101 come from outside North America?"]


def _definition(route_id: str) -> str:
    prompt = llm_router.system_prompt()
    start = prompt.index(f"- {route_id}:")
    end = prompt.find("\n- ", start + 1)
    return prompt[start:end if end > 0 else None].lower()


def test_sourcing_definition_covers_current_supplier_and_origin():
    sourcing = _definition("sourcing_whatif")
    assert "who supplies" in sourcing and "come from" in sourcing and "outside north america" in sourcing
    assert "today" in sourcing                                          # not only "what if we change"
    assert "sourcing_whatif" in _definition("out_of_scope")            # out_of_scope points supplier questions back
    assert "sourcing_whatif" in _definition("qualification_check")


def test_regression_questions_are_not_in_the_prompt_or_any_held_out_set():
    from evaluation import adversarial_suite
    prompt = llm_router.system_prompt().lower()
    held = set()
    for path in list(llm_router.ROOT.glob("data/test_*.jsonl")) + list(llm_router.ROOT.glob("data/router_split/*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                held.add(str(row.get("query") or row.get("input") or row.get("question") or "").strip().lower())
    for case in list(adversarial_suite.ATTACKS) + list(adversarial_suite.TEST_CASES):
        held.add(str(case.get("query") or case.get("message") or case.get("input") or "").strip().lower())
    for q in ALL_SUPPLIER_QUESTIONS + PORTAL_CASES:
        assert q.lower() not in prompt and q.lower() not in held, q


@pytest.mark.parametrize("question", PORTAL_CASES)
def test_onnx_fallback_routes_supplier_questions_to_sourcing(question):
    pytest.importorskip("onnxruntime")
    from agents import router_agent
    assert pipeline._label(router_agent.route_user_input(question)) == "sourcing_whatif"


live = pytest.mark.skipif(not (os.environ.get("OPENROUTER_API_KEY") and os.environ.get("CUSMA_LIVE_ROUTER_TESTS")),
                          reason="real model: set OPENROUTER_API_KEY and CUSMA_LIVE_ROUTER_TESTS=1 (costs < $0.01)")


@live
@pytest.mark.parametrize("question", ALL_SUPPLIER_QUESTIONS + PORTAL_CASES)
def test_llm_router_sends_supplier_and_origin_questions_to_sourcing(question):
    assert llm_router.route(question) == "sourcing_whatif"
