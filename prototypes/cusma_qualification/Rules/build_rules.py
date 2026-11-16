"""Build footwear_rules.json from the research workbook's Rules tab.

The workbook (footwear_rules_research.xlsx) is the source: the rule text, thresholds and
US duty rates come from the research. This script copies the numbers out of the Rules tab
into footwear_rules.json, which the rules engine reads. Nobody edits the JSON by hand:
change the workbook, then re-run this script and commit both files together.

Rules tab columns read:
  A  tariff code              E  excluded inputs        H  value threshold     K  Chapter 99 notes
  D  tariff shift allowed     F  value test (Yes/No)    I  de minimis          L  Sec. 232/338 notes
  J  US general duty          N  date checked (the newest one becomes the rules version)

Run from the repository root (works locally, in Colab or in CI; no Drive paths):
    python prototypes/cusma_qualification/Rules/build_rules.py            # write the JSON
    python prototypes/cusma_qualification/Rules/build_rules.py --check    # only check it is up to date
"""
import json
import re
import sys
from pathlib import Path

import openpyxl

RULES_FOLDER = Path(__file__).resolve().parent
WORKBOOK = RULES_FOLDER / "footwear_rules_research.xlsx"
OUT = RULES_FOLDER / "footwear_rules.json"

# Team decisions that are not on the Rules tab, kept here in one place.
TEAM_SETTINGS = {
    "escalation_margin": 0.02,          # within 2 points of the threshold -> a person decides
    "knee_review_band_in": [14, 17],    # Assumption #8: boots 14-17 in. tall -> a person decides
}


def pct(text, default=None):
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", str(text))
    return float(m.group(1)) / 100 if m else default


def build(workbook=WORKBOOK):
    """Read the Rules tab and return the rules as a dict (nothing is written)."""
    ws = openpyxl.load_workbook(workbook, data_only=True)["Rules"]
    header_row = next(r for r in range(1, 15) if str(ws.cell(r, 1).value or "").startswith("Tariff code"))
    footwear, duty = None, {}

    for r in range(header_row + 1, ws.max_row + 1):
        if ws.cell(r, 1).value in (None, ""):
            continue
        shift, excluded, value_test = ws.cell(r, 4).value, ws.cell(r, 5).value, ws.cell(r, 6).value
        threshold, de_min, general = ws.cell(r, 8).value, ws.cell(r, 9).value, ws.cell(r, 10).value
        ch99, special = str(ws.cell(r, 11).value or ""), str(ws.cell(r, 12).value or "")

        # The US 8-digit duty line is written in column K ("Duty line: 6401.99.10 ...")
        m = re.search(r"(\d{4}\.\d{2}\.\d{2})", ch99)
        if not m:
            print(f"  row {r}: no 8-digit duty line in column K, skipped")
            continue
        code8 = m.group(1)
        plus = re.search(r"\+\s*\d+(?:\.\d+)?\s*%", ch99)
        extra = pct(plus.group(0)) if plus else 0.0
        anyway = pct(special, 0.0) if special.strip().upper().startswith("YES") else 0.0
        scope = "stretch" if str(value_test).strip().lower() == "no" else "core"
        duty[code8] = {"general": float(general), "usmca": 0.0, "extra_if_not_qualifying": extra,
                       "applies_anyway": anyway, "scope": scope, "from_rules_row": r}

        # A second code in the notes, e.g. "Exception: 6401.92.60 (>90% PVC ...) 4.6%"
        for m2 in re.finditer(r"Exception:\s*(\d{4}\.\d{2}\.\d{2})\s*\([^)]*\)\s*(\d+(?:\.\d+)?)%", ch99):
            code_e = m2.group(1)
            # Sec. 338 "applies anyway" carries over unless column L says this code is "not listed"
            not_listed = code_e in special and "not listed" in special.split(code_e, 1)[1][:40]
            duty[code_e] = dict(duty[code8], general=float(m2.group(2)) / 100,
                                applies_anyway=0.0 if not_listed else duty[code8]["applies_anyway"])

        # The footwear rule itself comes from the first footwear row (value test = Yes)
        if scope == "core" and footwear is None:
            rng = re.search(r"(\d{2})\.(\d{2})\s*-\s*(\d{2})\.(\d{2})", str(shift))
            footwear = {
                "excluded_headings_range": [int(rng.group(1) + rng.group(2)), int(rng.group(3) + rng.group(4))],
                "excluded_subheadings": sorted(set(re.findall(r"\b\d{4}\.\d{2}\b", str(excluded)))),
                "rvc_method": "net_cost",
                "rvc_threshold": float(threshold),
                "de_minimis_share": pct(de_min),
            }

    if footwear is None or not duty:
        raise ValueError("Could not find the footwear rows on the Rules tab.")
    lo, hi = footwear["excluded_headings_range"]
    footwear["applies_to_headings"] = [str(h) for h in range(lo, hi + 1)]
    footwear["escalation_margin"] = TEAM_SETTINGS["escalation_margin"]
    checked = [str(ws.cell(r, 14).value)[:10] for r in range(header_row + 1, ws.max_row + 1) if ws.cell(r, 14).value]
    return {"rules_version": max(checked) if checked else "",
            "source": f"{Path(workbook).name}, Rules tab (read by build_rules.py)",
            "footwear_rule": footwear,
            "classification": {"knee_review_band_in": TEAM_SETTINGS["knee_review_band_in"]},
            "us_duty": duty}


def main(argv):
    rules = build()
    if "--check" in argv:
        current = json.loads(OUT.read_text()) if OUT.exists() else None
        if current != rules:
            sys.exit(f"{OUT.name} is out of date: re-run build_rules.py and commit it with the workbook.")
        print(f"{OUT.name} matches {WORKBOOK.name} (rules version {rules['rules_version']}).")
        return
    OUT.write_text(json.dumps(rules, indent=2) + "\n")
    print(f"Read {WORKBOOK.name} -> wrote {OUT.name} (rules version {rules['rules_version']})")
    for code, d in rules["us_duty"].items():
        print(f"   {code}  general {d['general']:.1%}  +{d['extra_if_not_qualifying']:.0%} if not qualifying"
              f"  {d['applies_anyway']:.0%} anyway  ({d['scope']})")


if __name__ == "__main__":
    main(sys.argv[1:])
