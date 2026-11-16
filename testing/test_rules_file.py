"""The committed rules file must match the research workbook it is built from."""
import json
import sys
from pathlib import Path

RULES_FOLDER = Path("prototypes/cusma_qualification/Rules")
sys.path.insert(0, str(RULES_FOLDER))

import build_rules  # noqa: E402


def test_rules_json_matches_workbook():
    """Edit the workbook, re-run build_rules.py, commit both. This test catches a missed step."""
    committed = json.loads((RULES_FOLDER / "footwear_rules.json").read_text())
    assert committed == build_rules.build(), "footwear_rules.json is out of date: run build_rules.py"


def test_mens_and_womens_leather_codes():
    """Women's/unisex leather carries the 50% Sec. 338 duty anyway; men's does not."""
    duty = build_rules.build()["us_duty"]
    assert duty["6403.91.90"]["applies_anyway"] == 0.5
    assert duty["6403.91.60"]["general"] == 0.085
    assert duty["6403.91.60"]["applies_anyway"] == 0.0
