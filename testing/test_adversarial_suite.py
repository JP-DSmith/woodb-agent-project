"""Checks that the attack suite is well formed and that the input guard behaves as each row says it should.
Run from the repo root: pytest -q

Rows come from three places (see the source field): "hand-written" (written by hand), "team-written, AI-assisted"
(team members, with AI help) and "generated" (Kush's hidden-instruction documents).
A row marked known_gap is an attack the input guard should stop but does not yet. The guard test skips it,
and a separate test fails as soon as the guard starts catching it, so the flag gets removed.
"""
import re

import pytest

from evaluation.adversarial_suite import (
    ATTACKS, CATEGORIES, LAYERS, OUTCOMES, TEST_CASES, TIERS, guard_result,
)

ROWS = ATTACKS + TEST_CASES
SOURCES = {"hand-written", "team-written, AI-assisted", "generated"}
NAMES = re.compile(r"\b(?:mahek|nerissa|kush|daniel|jason|nazia|mukobi)\b", re.I)
KNOWN_GAPS = [r for r in ATTACKS if r.get("known_gap")]


def test_counts():
    # 15 original + 101 from the sheet + 15 hidden-instruction documents; 10 original + 73 from the sheet
    assert len(ATTACKS) == 131
    assert len(TEST_CASES) == 83


def test_ids_are_unique():
    ids = [r["id"] for r in ROWS]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("row", ATTACKS, ids=lambda r: r["id"])
def test_attack_fields(row):
    assert row["category"] in CATEGORIES
    assert row["tier"] in TIERS
    assert row["expected_outcome"] in OUTCOMES
    assert row["target_layer"] in LAYERS
    assert row["input_type"] in ("message", "document")
    assert row["input"].strip()
    assert row["source"] in SOURCES
    assert row["guard_expected"] in ("allow", "block", "flag_security")
    for rx in row["must_not_match"]:
        re.compile(rx)


@pytest.mark.parametrize("row", TEST_CASES, ids=lambda r: r["id"])
def test_test_case_fields(row):
    assert row["category"] in CATEGORIES
    assert row["tier"] in TIERS
    assert row["input"].strip()
    assert row["expected_route"]
    assert row["source"] in SOURCES
    assert row["guard_expected"] == "allow"   # test cases are ordinary use and must not be blocked


@pytest.mark.parametrize("row", ROWS, ids=lambda r: r["id"])
def test_no_team_names_in_suite(row):
    assert not NAMES.search(str(row))


@pytest.mark.parametrize("row", ROWS, ids=lambda r: r["id"])
def test_guard_behaves_as_expected(row):
    if row.get("known_gap"):
        pytest.skip("known gap: the input guard should stop this but does not yet")
    assert guard_result(row).decision == row["guard_expected"]


@pytest.mark.parametrize("row", KNOWN_GAPS, ids=lambda r: r["id"])
def test_known_gaps_are_still_gaps(row):
    assert row["target_layer"] == "input_guard"
    # when the guard learns to catch a row, this fails: remove "known_gap" from the row
    assert guard_result(row).decision != row["guard_expected"]


def test_translation_row_needs_the_translated_text():
    row = next(r for r in ATTACKS if r["id"] == "MR-11")
    from security.input_guardrails import check

    assert check(row["input"]).decision == "allow"        # the original alone gets through
    assert guard_result(row).decision == "block"          # the translated text is caught
