"""Intake stress test (T34) in the normal test suite: any unsafe intake reading fails the build.

Unsafe = a document recorded as pending (clean) when it has a problem, was only partly read, or shows
the wrong country of origin. Documents the rules can't read, or flag, go to a person and are safe.
No API key is needed: the agent reads with its rules only.
"""
import json

import pytest

from evaluation import intake_stress_test as stress


@pytest.fixture(scope="module")
def sets():
    return stress.load_sets()


@pytest.mark.parametrize("variant", sorted(stress.VARIANTS))
def test_no_unsafe_intake_result(variant, sets):
    results = stress.run_stress({variant: stress.VARIANTS[variant]}, sets)
    assert results["unsafe"] == [], f"{len(results['unsafe'])} unsafe readings, first: {results['unsafe'][:3]}"
    assert sum(results["counts"][variant]["all"].values()) == len(sets)


def test_every_language_is_covered(sets):
    assert {t["language"] for _, t, _ in sets} == set(stress.LANGUAGES)


def test_unchanged_documents_are_all_read_correctly(sets):
    results = stress.run_stress({"none": lambda text: text}, sets)
    counts = results["counts"]["none"]["all"]
    assert results["unsafe"] == [] and counts["handed_to_person_unread"] == 0
    # flagged for a person, with one problem hidden behind another (DOC-0026, DOC-0206, DOC-0160 in T34)
    assert counts["flagged_for_person"] <= 3 and counts["correct"] + counts["flagged_for_person"] == len(sets)


# ---------------------------------------------------------------- the checks themselves can fail
def _found(truth, **over):
    found = {"read_by": "rules", "supplier_id": truth["supplier_id"], "defects": [],
             "items": [{"origin": i["origin_country"]} for i in truth["items"]]}
    found.update(over)
    return found


def test_problem_recorded_as_clean_is_unsafe(sets):
    truth = next(t for _, t, _ in sets if stress.expected_defects(t))
    assert stress.outcome(truth, _found(truth)) == "silent_miss"


def test_partial_read_recorded_as_clean_is_unsafe(sets):
    truth = next(t for _, t, _ in sets if not t["defects"] and len(t["items"]) >= 2 and t["doc_type"] == "invoice")
    partial = _found(truth, items=[{"origin": truth["items"][0]["origin_country"]}])
    assert stress.outcome(truth, partial) == "silent_partial_read"


def test_wrong_origin_recorded_as_clean_is_unsafe(sets):
    truth = next(t for _, t, _ in sets if not t["defects"])
    wrong = "CN" if truth["items"][0]["origin_country"] != "CN" else "VN"
    found = _found(truth)
    found["items"][0] = {"origin": wrong}
    assert stress.outcome(truth, found) == "silent_wrong_origin"


def test_flagged_or_unread_documents_are_safe(sets):
    truth = next(t for _, t, _ in sets if stress.expected_defects(t))
    assert stress.outcome(truth, _found(truth, defects=["H2"], items=[])) == "handed_to_person_unread"
    assert stress.outcome(truth, _found(truth, defects=["H2"])) == "flagged_for_person"


def test_an_email_is_scored_on_the_one_item_it_shows(sets):
    truth = next((t for _, t, _ in sets if t["doc_type"] == "supplier_email" and len(t["items"]) >= 2
                  and not t["defects"]), None)
    if truth is None:
        pytest.skip("no clean multi-item email")
    one = _found(truth, items=[{"origin": truth["items"][0]["origin_country"]}])
    assert stress.outcome(truth, one) == "correct"


def test_command_line_fails_on_unsafe_results_and_writes_json(tmp_path, monkeypatch):
    unsafe = {"counts": {"tabs": {"all": stress.Counter({"silent_miss": 1}), "en": stress.Counter({"silent_miss": 1})}},
              "unsafe": [{"variant": "tabs", "set": "output", "doc_id": "DOC-0001", "outcome": "silent_miss",
                          "items_expected": 1, "items_read": 1, "defects_expected": ["M1"], "defects_found": []}]}
    monkeypatch.setattr(stress, "run_stress", lambda: unsafe)
    out = tmp_path / "stress.json"
    assert stress.main(["--json", str(out)]) == 1
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["unsafe"][0]["doc_id"] == "DOC-0001" and saved["counts"]["tabs"]["all"]["silent_miss"] == 1


def test_percentages_handle_an_empty_group():
    assert stress.pct(0, 0) == 0.0 and stress.pct(1, 4) == 25.0


def test_importing_the_module_does_not_run_it(capsys):
    import importlib
    importlib.reload(stress)
    assert capsys.readouterr().out == ""
