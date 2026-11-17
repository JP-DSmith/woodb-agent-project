"""Offline tests for the backup router (fine_tuning/backup). No model calls, no downloads."""
import json

import pytest

from fine_tuning.backup import economics as econ
from fine_tuning.backup import routers

ROOT = routers.ROOT


def _jsonl(name):
    return [json.loads(l) for l in (ROOT / "data" / "router_split" / f"{name}.jsonl").open(encoding="utf-8")]


def test_llm_prompt_uses_route_descriptions_but_no_example_sentences():
    cfg = json.loads((ROOT / "config" / "request_types_and_personas.json").read_text(encoding="utf-8"))
    text = routers.route_descriptions()
    for rt in cfg["request_types"]:
        assert rt["id"] in text and rt["description"] in text
        examples = rt.get("examples") or {}
        flat = [e for group in examples.values() for e in group] if isinstance(examples, dict) else list(examples)
        assert flat and not any(e in text for e in flat)   # includes the examples added in 960148c


def test_few_shot_examples_come_from_training_only():
    shots = routers.few_shot_examples(2)
    train = {r["query"] for r in _jsonl("train")}
    held_out = {r["query"] for r in _jsonl("val_new_wording") + _jsonl("val_new_style")}
    picked = [line[len("Question: "):] for line in shots.splitlines() if line.startswith("Question: ")]
    assert len(picked) == 2 * len(routers.LABELS)
    assert all(q in train for q in picked) and not any(q in held_out for q in picked)
    assert routers.few_shot_examples(0) == ""


class FakeModel:
    def __init__(self, model=None, temperature=0.0):
        self.model = model

    def __call__(self, messages, tools):
        q = messages[-1]["content"]
        if "two" in q:
            return {"content": "rules_info or duty_cost"}
        if "junk" in q:
            return {"content": "I think it's about boots"}
        return {"content": " Duty_Cost \n"}


def test_llm_router_reads_one_label_and_falls_back_to_escalate(monkeypatch):
    import agents.base
    monkeypatch.setattr(agents.base, "OpenRouterModel", FakeModel)
    route = routers.make_llm_router("openai/gpt-4o-mini", few_shot=1)
    assert route("what's the duty on boots?") == "duty_cost"
    assert route("two answers please") == "escalate_human"          # ambiguous reply
    assert route("junk") == "escalate_human"                        # no label
    assert route.__name__ == "llm_gpt-4o-mini"


def test_economics_break_even_and_table():
    ft = {"name": "ft", "kind": "finetuned", "accuracy": 0.9, "cost_usd": 0.0, "seconds": 0.01}
    llm = {"name": "mini", "kind": "llm", "accuracy": 0.8, "cost_usd": 0.0001, "seconds": 1.0}
    big = {"name": "big", "kind": "llm", "accuracy": 0.95, "cost_usd": 0.002, "seconds": 1.0}
    inputs = dict(econ.INPUTS)
    saving = econ.per_question_usd(llm, inputs) - econ.per_question_usd(ft, inputs)
    fixed = econ.monthly_fixed_usd("finetuned", inputs) + econ.one_time_usd(3600, inputs) / inputs["amortise_months"]
    assert econ.break_even_volume(ft, llm, 3600, inputs) == pytest.approx(fixed / saving)
    assert econ.break_even_volume(ft, big, 3600, inputs) is None        # the large model is more accurate
    assert econ.monthly_usd(llm, 1000, inputs=inputs) == pytest.approx(1000 * (0.0001 + 0.2 * inputs["misroute_usd"]))
    text = econ.table([ft, llm, big], 3600, inputs)
    assert "1,000,000/month" in text and "Running cost only" in text and "never" in text
    assert "Misrouted questions per month" in text and "Including misroutes at $0.50" in text
    assert econ.monthly_usd(llm, 1000, inputs=inputs, misroutes=False) == pytest.approx(1000 * 0.0001)
    cost_only = econ.break_even_volume(ft, llm, 3600, inputs, misroutes=False)
    assert cost_only == pytest.approx(fixed / (0.0001 - 0.01 * inputs["cpu_usd_per_second"]))


def test_training_script_runs_on_a_tiny_model(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from fine_tuning.backup import train_finetuned as tf
    tok = transformers.PreTrainedTokenizerFast(tokenizer_file=str(ROOT / "models" / "router" / "tokenizer.json"),
                                               pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]")
    cfg = transformers.BertConfig(vocab_size=30522, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                  intermediate_size=32, num_labels=8)
    model = transformers.BertForSequenceClassification(cfg)
    real = tf.load
    monkeypatch.setattr(tf, "load", lambda name: real(name)[:64])
    info = tf.train({**tf.DEFAULTS, "epochs": 1, "batch_size": 16, "model_dir": str(tmp_path)}, tokenizer=tok, model=model)
    assert (tmp_path / "training_info.json").exists() and info["best_epoch"]["epoch"] == 1
    monkeypatch.setattr(routers, "MODEL_DIR", str(tmp_path))
    routers.reset_finetuned()
    assert routers.finetuned_route("Does the Glacier Pro qualify?") in routers.LABELS


# ---------------------------------------------------------------- analysis
from fine_tuning.backup import analysis as an


def _rows(router, pairs):
    return [{"router": router, "set": "s", "id": str(i), "expected": e, "predicted": p, "correct": e == p,
             "seconds": 0.01, "cost_usd": 0.0} for i, (e, p) in enumerate(pairs)]


def test_metrics_macro_f1_and_confusion():
    rows = _rows("a", [("duty_cost", "duty_cost"), ("duty_cost", "rules_info"), ("rules_info", "rules_info")])
    m = an.metrics(rows)
    assert m["accuracy"] == pytest.approx(2 / 3)
    assert m["per_route"]["duty_cost"]["f1"] == pytest.approx(2 / 3)       # precision 1, recall 0.5
    assert m["per_route"]["rules_info"]["f1"] == pytest.approx(2 / 3)      # precision 0.5, recall 1
    assert m["macro_f1"] == pytest.approx(2 / 3)
    d, r = an.LABELS.index("duty_cost"), an.LABELS.index("rules_info")
    assert m["confusion"][d][r] == 1 and m["confusion"][d][d] == 1


def test_bootstrap_interval_contains_the_score_and_mcnemar_is_paired():
    rows = _rows("a", [("duty_cost", "duty_cost")] * 70 + [("duty_cost", "rules_info")] * 30)
    ci = an.bootstrap(rows, n=300)
    assert ci["accuracy"][0] < 0.7 < ci["accuracy"][1]
    better = _rows("b", [("duty_cost", "duty_cost")] * 100)
    t = an.mcnemar(rows, better)
    assert (t["only_a_right"], t["only_b_right"]) == (0, 30) and t["p_value"] < 0.001
    assert an.mcnemar(rows, rows)["p_value"] == 1.0


def _tiny(transformers, torch):
    torch.manual_seed(0)
    tok = transformers.PreTrainedTokenizerFast(tokenizer_file=str(ROOT / "models" / "router" / "tokenizer.json"),
                                               pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]")
    cfg = transformers.BertConfig(vocab_size=30522, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                  intermediate_size=32, num_labels=8)
    return tok, transformers.BertForSequenceClassification(cfg)


def test_frozen_encoder_keeps_pretrained_weights_and_full_training_changes_them(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from fine_tuning.backup import train_finetuned as tf
    real = tf.load
    monkeypatch.setattr(tf, "load", lambda name: real(name)[:48] if name.startswith("train") else real(name)[::40])
    base = {**tf.DEFAULTS, "epochs": 1, "batch_size": 16}
    frozen = tf.train({**base, "freeze_encoder": True, "model_dir": str(tmp_path / "f")}, *_tiny(transformers, torch))
    full = tf.train({**base, "model_dir": str(tmp_path / "t")}, *_tiny(transformers, torch))
    wc_frozen, wc_full = frozen["history"][-1]["weight_change"], full["history"][-1]["weight_change"]
    assert all(v == 0 for g, v in wc_frozen.items() if g != "classification head")
    assert wc_frozen["classification head"] > 0
    assert any(v > 0 for g, v in wc_full.items() if g.startswith("encoder"))
    assert frozen["trainable_weights"] < full["trainable_weights"] == full["total_weights"]
    assert frozen["history"][0]["epoch"] == 0 and "f1_per_route" in frozen["history"][1]["val_new_style"]


def test_live_router_check_spots_identical_and_changed_weights(tmp_path):
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    onnx = pytest.importorskip("onnx")
    from onnx import numpy_helper
    inits = {i.name: numpy_helper.to_array(i) for i in onnx.load(str(ROOT / "models" / "router" / "model.onnx")).graph.initializer}
    model = transformers.BertModel(transformers.BertConfig(vocab_size=30522, hidden_size=384, num_hidden_layers=6,
                                                           num_attention_heads=12, intermediate_size=1536), add_pooling_layer=False)
    sd = model.state_dict()
    for k in sd:                         # stand-in for the download: the same weights the ONNX file holds
        if k in inits and inits[k].shape == tuple(sd[k].shape):
            sd[k] = torch.tensor(inits[k])
    model.load_state_dict(sd)
    model.save_pretrained(tmp_path / "same")
    same = an.live_router_matches_pretrained(str(ROOT / "models" / "router" / "model.onnx"), str(tmp_path / "same"))
    assert same["exact_compared"] > 50 and same["exact_identical"] == same["exact_compared"]
    model.load_state_dict({k: v + 0.01 for k, v in sd.items()})
    model.save_pretrained(tmp_path / "trained")
    changed = an.live_router_matches_pretrained(str(ROOT / "models" / "router" / "model.onnx"), str(tmp_path / "trained"))
    assert changed["exact_identical"] == 0


def test_search_picks_the_best_combination_and_its_epochs(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from fine_tuning.backup import experiments as ex, train_finetuned as tf
    real = tf.load
    monkeypatch.setattr(tf, "load", lambda name: real(name)[:32] if name.startswith("train") else real(name)[::60])
    results = ex.search({**tf.DEFAULTS}, str(tmp_path), {"learning_rate": [1e-4, 1e-3], "batch_size": [16]}, epochs=2,
                        build=lambda name: _tiny(transformers, torch))
    assert len(results) == 2 and results[0]["mean_macro_f1"] >= results[1]["mean_macro_f1"]
    assert all(len(r["curve"]) == 3 for r in results)          # epoch 0 plus 2 epochs
    best = ex.best_settings({**tf.DEFAULTS}, results)
    assert best["learning_rate"] == results[0]["learning_rate"] and best["epochs"] == results[0]["best_epoch"]
    assert ex.best_settings({**tf.DEFAULTS}, results, max_epochs=6)["epochs"] == 6
    an.chart_search(results, tmp_path / "s.png")
    assert (tmp_path / "s.png").exists() and "Learning rate" in an.search_table(results)


def test_frozen_probe_router_trains_a_classifier_and_leaves_the_encoder_unchanged(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("sklearn")
    tok, model = _tiny(transformers, torch)
    model.base_model.save_pretrained(tmp_path)
    tok.save_pretrained(tmp_path)
    before = {k: v.clone() for k, v in transformers.AutoModel.from_pretrained(tmp_path).state_dict().items()}
    real_open = type(ROOT).open
    route = routers.make_probe_router(str(tmp_path), train_file="train")
    monkeypatch.setattr(type(ROOT), "open", lambda self, *a, **k: __import__("io").StringIO(
        "".join(list(real_open(self, *a, **k))[:80])) if self.name == "train.jsonl" else real_open(self, *a, **k))
    assert route("Does the Glacier Pro qualify?") in routers.LABELS
    assert route.info["train_questions"] == 80 and route.info["weights_trained"] > 0
    after = transformers.AutoModel.from_pretrained(tmp_path).state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_llm_answers_are_cached_so_reruns_cost_nothing(tmp_path, monkeypatch):
    import agents.base
    calls = []

    class Counting(FakeModel):
        def __call__(self, messages, tools):
            calls.append(1)
            return super().__call__(messages, tools)

    monkeypatch.setattr(agents.base, "OpenRouterModel", Counting)
    cache = tmp_path / "llm_cache.json"
    first = routers.make_llm_router("openai/gpt-4o-mini", cache_path=cache)
    assert first("what's the duty on boots?") == "duty_cost"
    first.save()
    again = routers.make_llm_router("openai/gpt-4o-mini", cache_path=cache)
    assert again("what's the duty on boots?") == "duty_cost" and len(calls) == 1
    other = routers.make_llm_router("openai/gpt-4o-mini", few_shot=2, cache_path=cache)   # different setup: new call
    other("what's the duty on boots?")
    assert len(calls) == 2


def test_reused_llm_answers_still_report_their_cost_and_time(tmp_path, monkeypatch):
    import agents.base

    class Priced(FakeModel):
        def __call__(self, messages, tools):
            meter = agents.base._meter.get()
            if meter is not None:
                meter.add("openai/gpt-4o-mini", 500, 3, 0.00008)
            return super().__call__(messages, tools)

    monkeypatch.setattr(agents.base, "OpenRouterModel", Priced)
    cache = tmp_path / "llm_cache.json"
    first = routers.make_llm_router("openai/gpt-4o-mini", cache_path=cache)
    with agents.base.measure_usage() as m1:
        first("what's the duty on boots?")
    first.save()
    again = routers.make_llm_router("openai/gpt-4o-mini", cache_path=cache)
    with agents.base.measure_usage() as m2:
        again("what's the duty on boots?")
    assert m2.cost_usd == pytest.approx(m1.cost_usd) and m2.prompt_tokens == 500 and again.unpriced == 0
    rows = [{"router": "llm small", "question": "what's the duty on boots?", "seconds": 0.0}]
    assert routers.restore_seconds(rows, "llm small", again) == 1 and rows[0]["seconds"] is not None
    cache.write_text('{"openai/gpt-4o-mini|0|0.0|old": "duty_cost"}', encoding="utf-8")   # older cache: answer only
    old = routers.make_llm_router("openai/gpt-4o-mini", cache_path=cache)
    assert old("old") == "duty_cost" and old.unpriced == 1
