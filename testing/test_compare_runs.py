"""Offline tests for evaluation/compare_runs.py (no model, no API key)."""
import csv

from evaluation.compare_runs import compare, main

COLS = ["case_id", "case_kind", "category", "tier", "query", "route_correct", "answer_correct",
        "attack_passed_policy", "actual_route", "actual_verdict", "actual_dollar_value",
        "attack_status", "tools_used", "pipeline_decision", "seconds", "calls",
        "prompt_tokens", "completion_tokens", "cost_usd", "error"]


def _write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLS})


def _row(cid, route, answer, **extra):
    base = {"case_id": cid, "case_kind": "holdout", "category": "qualification", "tier": "A",
            "query": f"question {cid}", "route_correct": route, "answer_correct": answer,
            "actual_route": "qualification", "seconds": "1.0", "calls": "1",
            "prompt_tokens": "100", "completion_tokens": "10", "cost_usd": "0.0001"}
    base.update(extra)
    return base


def test_fixed_and_broke_are_listed(tmp_path):
    before, after = tmp_path / "b.csv", tmp_path / "a.csv"
    _write(before, [_row("c1", "True", "False"), _row("c2", "True", "True"), _row("c3", "True", "True")])
    _write(after, [_row("c1", "True", "True"), _row("c2", "False", "True", actual_route="duty"),
                   _row("c3", "True", "True")])
    report, fl = compare(str(before), str(after))
    changes = {(f["case_id"], f["what"], f["change"]) for f in fl}
    assert changes == {("c1", "answer", "fixed"), ("c2", "route", "broke")}
    assert "1 fixed, 1 broke" in report
    assert "## Routing" in report and "## Time and cost" in report


def test_only_shared_questions_are_compared(tmp_path):
    before, after = tmp_path / "b.csv", tmp_path / "a.csv"
    _write(before, [_row("c1", "True", "True"), _row("only_before", "False", "False")])
    _write(after, [_row("c1", "True", "True"), _row("only_after", "True", "True")])
    report, fl = compare(str(before), str(after))
    assert "Compared: 1 questions in both runs" in report
    assert fl == []


def test_out_writes_report_and_flips(tmp_path, capsys):
    before, after = tmp_path / "b.csv", tmp_path / "a.csv"
    _write(before, [_row("c1", "True", "False")])
    _write(after, [_row("c1", "True", "True")])
    main([str(before), str(after), "--out", str(tmp_path / "cmp")])
    assert (tmp_path / "cmp.md").read_text(encoding="utf-8").startswith("# Run comparison")
    flips = list(csv.DictReader(open(tmp_path / "cmp_flips.csv", encoding="utf-8")))
    assert flips[0]["case_id"] == "c1" and flips[0]["change"] == "fixed"
