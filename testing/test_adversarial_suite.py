"""Checks that the hand-written attack suite is well formed and that the input guard
behaves as each row says it should. Run from the repo root: pytest -q"""
import re

import pytest

from evaluation.adversarial_suite import (
    ATTACKS, CATEGORIES, LAYERS, OUTCOMES, TEST_CASES, TIERS, guard_result,
)

ROWS = ATTACKS + TEST_CASES
NAMES = re.compile(r"\b(?:mahek|nerissa|kush|daniel|jason|nazia|mukobi)\b", re.I)


def test_counts():
    assert len(ATTACKS) == 15
    assert len(TEST_CASES) == 10


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
    assert row["source"] == "hand-written"
    for rx in row["must_not_match"]:
        re.compile(rx)


@pytest.mark.parametrize("row", TEST_CASES, ids=lambda r: r["id"])
def test_test_case_fields(row):
    assert row["category"] in CATEGORIES
    assert row["tier"] in TIERS
    assert row["input"].strip()
    assert row["expected_route"]
    assert row["guard_expected"] == "allow"   # test cases are ordinary use and must not be blocked


@pytest.mark.parametrize("row", ROWS, ids=lambda r: r["id"])
def test_no_team_names_in_suite(row):
    assert not NAMES.search(str(row))


@pytest.mark.parametrize("row", ROWS, ids=lambda r: r["id"])
def test_guard_behaves_as_expected(row):
    assert guard_result(row).decision == row["guard_expected"]


def test_translation_row_needs_the_translated_text():
    row = next(r for r in ATTACKS if r["id"] == "MR-11")
    from security.input_guardrails import check

    assert check(row["input"]).decision == "allow"        # the original alone gets through
    assert guard_result(row).decision == "block"          # the translated text is caught
