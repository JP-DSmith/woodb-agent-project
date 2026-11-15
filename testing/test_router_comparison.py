"""Offline tests for the router validation split and the router comparison (no model, no network)."""
import json
from collections import Counter

from evaluation import compare_routers as cr
from fine_tuning.make_router_split import ROOT, make_split, read_jsonl

TRAIN, VAL = read_jsonl(ROOT / "data" / "train_dataset.jsonl"), read_jsonl(ROOT / "data" / "val_dataset.jsonl")


def test_split_holds_out_three_templates_per_route_and_never_mixes_them():
    split = make_split(TRAIN, VAL)
    held = {f for frames in split["held_out"].values() for f in frames}
    assert all(len(frames) == 3 for frames in split["held_out"].values())
    assert not held & {r["frame_id"] for r in split["train"]}
    assert not held & {r["frame_id"] for r in split["val_seen_wording"]}
    assert {r["frame_id"] for r in split["val_new_wording"]} == held
    assert len(split["train"]) + len(split["val_seen_wording"]) + len(split["val_new_wording"]) == len(TRAIN) + len(VAL)


def test_split_is_the_same_every_time_and_matches_the_committed_files():
    assert make_split(TRAIN, VAL)["held_out"] == make_split(TRAIN, VAL)["held_out"]
    saved = json.loads((ROOT / "data" / "router_split" / "held_out_templates.json").read_text(encoding="utf-8"))
    assert saved["held_out_templates"] == make_split(TRAIN, VAL)["held_out"]
    assert len(read_jsonl(ROOT / "data" / "router_split" / "val_new_wording.jsonl")) == saved["counts"]["val_new_wording"]["questions"]


def test_split_uses_no_test_questions():
    test_texts = {r["query"] for r in read_jsonl(ROOT / "data" / "test_holdout_human_reviewed.jsonl")}
    for name in ("train", "val_seen_wording", "val_new_wording"):
        assert not test_texts & {r["query"] for r in read_jsonl(ROOT / "data" / "router_split" / f"{name}.jsonl")}


def test_test_sets_only_load_with_final():
    assert set(cr.VALIDATION_SETS) == {"val_new_wording", "val_new_style", "val_seen_wording"}
    assert not set(cr.VALIDATION_SETS) & set(cr.FINAL_SETS)


def _sets():
    return {"val_new_wording": [{"set": "val_new_wording", "id": "a", "question": "q1", "expected": "duty_cost"},
                                {"set": "val_new_wording", "id": "b", "question": "q2", "expected": "rules_info"}]}


def test_a_plug_in_router_is_scored_and_unknown_labels_count_as_escalate_human():
    answers = {"q1": "duty_cost", "q2": "not_a_route"}
    rows = cr.run_router("fake", lambda q: answers[q], _sets())
    assert [r["predicted"] for r in rows] == ["duty_cost", "escalate_human"]
    assert [r["correct"] for r in rows] == [True, False]
    text = cr.report(rows, ["val_new_wording"], final=False)
    assert "50.0% (1/2)" in text and "Validation only" in text


def test_the_trained_routers_dict_output_is_accepted_and_errors_count_as_wrong():
    def router(q):
        if q == "q2":
            raise RuntimeError("model down")
        return {"predicted_class": "duty_cost"}
    rows = cr.run_router("fake", router, _sets())
    assert rows[0]["correct"] and not rows[1]["correct"] and rows[1]["error"].startswith("RuntimeError")


def test_load_router_accepts_module_function(monkeypatch):
    name, fn = cr.load_router("kw=agents.specialists:keyword_route")
    assert name == "kw" and callable(fn)


# ---------------------------------------------------------------- paraphrasing (fake model, no network)
from fine_tuning import paraphrase_questions as pq


class FakeModel:
    def __init__(self, text_for):
        self.text_for = text_for

    def __call__(self, messages, tools):
        prompt = messages[0]["content"]
        styles = [line[2:].split(":")[0] for line in prompt.split("Styles:")[1].splitlines() if line.startswith("- ")]
        return {"content": json.dumps({"rewrites": [{"style": s, "text": self.text_for(s)} for s in styles]})}


def test_styles_are_the_same_every_time_and_never_copied_from_tests():
    assert pq.styles_for("train-00001", 3) == pq.styles_for("train-00001", 3)
    assert len(set(pq.styles_for("train-00001", 3))) == 3


def test_rewrites_that_lose_a_code_or_copy_the_question_are_dropped():
    q = "Does NB-0413 on PO-8013 qualify?"
    assert pq.check_rewrite(q, "hey does nb-0413 on po 8013 pass?") is None
    assert pq.check_rewrite(q, "does the glacier pro on that order pass?") == "code_lost"
    assert pq.check_rewrite(q, "Does NB-0413 on PO-8013 qualify?") == "copy"
    assert pq.check_rewrite(q, "") == "empty"


def test_rewrites_keep_the_route_and_point_to_the_original(tmp_path):
    rows = [{"id": "train-1", "query": "Does NB-0413 qualify?", "intent": "qualification_check", "frame_id": "QC01"}]
    model = FakeModel(lambda s: f"{s}: is NB-0413 ok under the trade deal")
    res = pq.run(rows, 3, model, cache=tmp_path / "c.jsonl", workers=2)
    styled = pq.styled_rows(rows, res)
    assert len(styled) == 3
    assert all(r["intent"] == "qualification_check" and r["paraphrase_of"] == "train-1" for r in styled)
    # a second run uses the cache and makes no calls
    res2 = pq.run(rows, 3, lambda *a: (_ for _ in ()).throw(AssertionError("called")), cache=tmp_path / "c.jsonl")
    assert res2 == res


def test_a_failed_call_is_skipped_and_retried_later(tmp_path):
    rows = [{"id": "train-2", "query": "What is RVC?", "intent": "rules_info", "frame_id": "RI01"}]
    def broken(messages, tools):
        raise RuntimeError("timeout")
    assert pq.run(rows, 2, broken, cache=tmp_path / "c.jsonl", workers=1) == {}
    assert len(pq.run(rows, 2, FakeModel(lambda s: f"{s} what does rvc mean"), cache=tmp_path / "c.jsonl")["train-2"]["kept"]) == 2


def test_spot_check_sheet_is_written(tmp_path):
    rows = [{"id": f"x-s{i}", "intent": "duty_cost", "style": "casual", "original_query": "q", "query": "r"} for i in range(5)]
    pq.write_spot_check(rows, tmp_path / "s.xlsx", n=3)
    from openpyxl import load_workbook
    ws = load_workbook(tmp_path / "s.xlsx").active
    assert ws.max_row == 5 and ws["F2"].value == "Same meaning?"
