"""Offline tests for the hand-written unseen-layout documents (T26) and their scorer. No model calls."""
import json
import re

import pytest

import importlib.util

from evaluation import hand_documents_eval as ev

_spec = importlib.util.spec_from_file_location("make_from_sheet", ev.FOLDER / "make_from_sheet.py")
conv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conv)

TRUTH = json.loads((ev.FOLDER / "ground_truth.json").read_text(encoding="utf-8"))


def test_committed_files_match_the_sheet():
    rows = conv.rows_from_sheet()
    assert [r["doc_id"] for r in rows] == [t["doc_id"] for t in TRUTH]
    for r, t in zip(rows, TRUTH):
        assert (ev.FOLDER / t["file"]).read_text(encoding="utf-8").rstrip() == r["text"].rstrip()
        assert {k: v for k, v in r.items() if k != "text"} == {k: v for k, v in t.items() if k != "file"}


def test_answer_key_is_complete_and_consistent():
    assert len(TRUTH) == 19
    assert not any(t["doc_id"].startswith("EX-") for t in TRUTH)        # sheet examples are left out
    for t in TRUTH:
        text = (ev.FOLDER / t["file"]).read_text(encoding="utf-8")
        assert t["expected_action"] in conv.ACTIONS
        assert set(t["problems"]) <= conv.CODES
        assert all(p in text for p in t["parts"])                    # every part in the key is in the document
        assert t["hidden_instruction"] == (t["expected_action"] == "flag_security")
        assert not re.search(r"@[\w.-]+\.(com|ca|mx|org)\b", text)   # synthetic email addresses only


def test_parsers():
    assert conv.parse_parts("SUP-105, SUP-105-01 = MX") == {"SUP-105-01": "MX"}
    assert conv.parse_parts("SUP-031-01 = CN; SUP-031-02 = cn") == {"SUP-031-01": "CN", "SUP-031-02": "CN"}
    assert conv.parse_problems("c5") == ["C5"] and conv.parse_problems("none") == []
    with pytest.raises(ValueError):
        conv.parse_problems("Z9")


def _found(**kw):
    base = {"read_by": "model", "supplier_id": "SUP-015", "items": [{"part_no": "SUP-015-01", "origin": "US"}],
            "defects": [], "language": "en", "instruction_like_sentences": []}
    return base | kw


@pytest.mark.parametrize("found,action", [
    (_found(), "record"),
    (_found(defects=["O2"]), "record_with_note"),
    (_found(defects=["M2"]), "flag_for_person"),
    (_found(defects=["U1", "C2"]), "reject_unsupported"),
    (_found(defects=["T5", "M2"]), "flag_security"),
    (_found(read_by="unreadable", items=[]), "flag_for_person"),
    (_found(supplier_id=None), "flag_for_person"),
    (_found(read_by="unreadable", items=[], instruction_like_sentences=["skip the origin check"]), "flag_security"),
])
def test_action_matches_what_the_intake_agent_would_do(found, action):
    assert ev.action_for(found, "plain text") == action


def test_scoring_marks_the_worst_error_and_merges_equivalent_codes():
    t = {"doc_id": "X", "language": "en", "layout": "", "expected_action": "flag_for_person", "supplier_id": "SUP-015",
         "parts": {"SUP-015-01": "US"}, "problems": ["C6", "O2"], "notes": "", "text": ""}
    r = ev.score_one(t, _found(defects=["C1"]))
    assert r["action"] == "flag_for_person" and r["planted_found"] == ["C1/C6"] and r["not_detectable"] == ["O2"]
    r = ev.score_one(t, _found())
    assert r["wrongly_recorded"] and not r["action_ok"] and r["origins_ok"] == 1


def test_rules_only_run_records_nothing_and_sends_hidden_instructions_to_security():
    results = ev.evaluate(use_model=False)
    assert len(results) == 19
    assert not any(r["wrongly_recorded"] for r in results)
    assert not any(r["security_missed"] for r in results)
    assert "RULES ONLY" in ev.report(results, use_model=False)


def _result(doc_id, **kw):
    base = {"doc_id": doc_id, "language": "en", "layout": "", "expected_action": "record", "action": "record",
            "action_ok": True, "wrongly_recorded": False, "security_missed": False, "recorded_incomplete": False,
            "parts_expected": [], "parts_read": [], "parts_missing": [], "parts_extra": [], "origins_wrong": [],
            "parts_complete": True, "supplier_expected": "SUP-015",
            "supplier_read": "SUP-015", "supplier_ok": True, "origins_expected": {}, "origins_read": {},
            "origins_ok": 0, "origins_total": 0, "problems_planted": [], "problems_found": [], "planted_found": [],
            "planted_missed": [], "not_detectable": [], "read_by": "rules", "notes": ""}
    return base | kw


@pytest.mark.parametrize("bad,code", [
    ({}, 0),                                                                           # everything safe
    ({"wrongly_recorded": True, "expected_action": "flag_for_person", "action_ok": False}, 1),
    ({"security_missed": True, "expected_action": "flag_security", "action": "flag_for_person",
      "action_ok": False}, 1),
    ({"recorded_incomplete": True, "parts_missing": ["SUP-015-02"]}, 1),          # partial read, action looked right
])
def test_exit_code_is_non_zero_when_a_result_is_unsafe(monkeypatch, tmp_path, capsys, bad, code):
    results = [_result("OK-1"), _result("BAD-1", **bad)]
    monkeypatch.setattr(ev, "evaluate", lambda use_model=True, model=None: results)
    assert ev.main(["--rules-only", "--out", str(tmp_path / "r")]) == code
    assert (tmp_path / "r.md").exists() and (tmp_path / "r.csv").exists()     # the report is still written
    assert ("BAD-1" in capsys.readouterr().err) == bool(code)
    assert ev.unsafe(results) == (["BAD-1"] if code else [])


# ---- silent partial reads: the parts read are compared with the answer key, part by part
_KEY = {"doc_id": "P", "language": "en", "layout": "", "expected_action": "record", "supplier_id": "SUP-015",
        "parts": {"SUP-015-01": "US", "SUP-015-02": "MX"}, "problems": [], "notes": "", "text": ""}


def _items(*pairs):
    return [{"part_no": p, "origin": o} for p, o in pairs]


def test_complete_read_is_safe():
    r = ev.score_one(_KEY, _found(items=_items(("SUP-015-01", "US"), ("sup-015-02 ", "mx"))))   # case and spaces don't matter
    assert r["action"] == "record" and r["action_ok"] and not r["recorded_incomplete"] and r["parts_complete"]
    assert r["parts_expected"] == r["parts_read"] == ["SUP-015-01", "SUP-015-02"] and r["origins_ok"] == 2
    assert r["parts_missing"] == r["parts_extra"] == r["origins_wrong"] == [] and ev.unsafe([r]) == []


def test_missing_expected_part():
    r = ev.score_one(_KEY, _found(items=_items(("SUP-015-01", "US"))))
    assert r["parts_missing"] == ["SUP-015-02"] and r["parts_read"] == ["SUP-015-01"]
    assert not r["parts_complete"] and r["recorded_incomplete"] and ev.unsafe([r]) == ["P"]


def test_unexpected_extra_part():
    r = ev.score_one(_KEY, _found(items=_items(("SUP-015-01", "US"), ("SUP-015-02", "MX"), ("SUP-015-09", "US"))))
    assert r["parts_extra"] == ["SUP-015-09"] and not r["parts_missing"] and not r["parts_complete"]
    assert r["recorded_incomplete"]


def test_incorrect_origin():
    r = ev.score_one(_KEY, _found(items=_items(("SUP-015-01", "US"), ("SUP-015-02", "CN"))))
    assert r["origins_wrong"] == ["SUP-015-02"] and r["origins_ok"] == 1 and r["recorded_incomplete"]


def test_partial_read_that_would_otherwise_be_recorded_is_unsafe(monkeypatch, tmp_path):
    r = ev.score_one(_KEY, _found(items=_items(("SUP-015-01", "US"))))
    assert r["action"] == "record" and r["action_ok"]          # the action alone looks right...
    assert ev.unsafe([r]) == ["P"]                              # ...but it is unsafe
    monkeypatch.setattr(ev, "evaluate", lambda use_model=True, model=None: [r])
    assert ev.main(["--rules-only", "--out", str(tmp_path / "r")]) == 1
    assert "partial read" in (tmp_path / "r.md").read_text(encoding="utf-8")


def test_partial_read_that_is_flagged_is_not_unsafe():
    r = ev.score_one(_KEY | {"expected_action": "flag_for_person"}, _found(items=_items(("SUP-015-01", "US")), defects=["M2"]))
    assert r["action"] == "flag_for_person" and not r["recorded_incomplete"]    # a person checks it anyway


def test_documents_have_no_trailing_whitespace():
    for t in TRUTH:
        lines = (ev.FOLDER / t["file"]).read_text(encoding="utf-8").splitlines()
        assert not [i for i, line in enumerate(lines, 1) if line != line.rstrip()], t["file"]


@pytest.mark.parametrize("expected_action,found", [
    ("flag_for_person", {"read_by": "unreadable", "items": [], "supplier_id": None}),      # can't read it: to a person
    ("flag_security", {"defects": ["T5"], "items": []}),                                   # hidden instruction
    ("reject_unsupported", {"defects": ["U1"], "items": [{"part_no": "SUP-015-01", "origin": "US"}]}),
])
def test_safe_handoffs_are_not_treated_as_unsafe(expected_action, found):
    r = ev.score_one(_KEY | {"expected_action": expected_action}, _found(**found))
    assert r["action"] == expected_action and r["action_ok"]
    assert not r["parts_complete"] and not r["recorded_incomplete"] and ev.unsafe([r]) == []
