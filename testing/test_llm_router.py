"""The prompted LLM router (agents/llm_router.py) and how the pipeline chooses its router. No model calls."""
import json

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
