"""Intake extraction accuracy by language (T34).

Runs the intake agent on the generated supplier documents (data/documents/output*/) and compares
what it reads with each document's answer key (ground_truth.json). Nothing is written to the
document records file.

    python evaluation/intake_accuracy.py                 # prints the tables
    python evaluation/intake_accuracy.py --json out.json # also saves the numbers for the report

Measures, overall and per language (en / fr / es):
  - document type, supplier: share read correctly
  - origin per item line: share of lines whose country of origin matches the answer key, among
    lines where the document shows a readable origin (scan errors are counted as readable if fixed)
  - action: flagged for a person vs recorded as pending, against expected_intake_action
  - problems: recall and precision per problem group (missing, confusing, out of date,
    handwritten, hidden instruction, unsupported)

Scoring notes (see the README in data/documents):
  - C1 (origin conflict) and C6 (origin doesn't match the supplier's country) look the same on the
    page, so they are scored as one problem, "origin mismatch".
  - O2 (old address) can't be checked: the World has no supplier addresses yet.
  - A wrong part number (C4) planted in a shipment email is not visible in the text (shipment
    emails don't show part numbers), so it is not counted.
  - Totals that don't add up (C5) on a purchase order are not visible either: a purchase order's
    TOTAL includes freight that isn't listed on it.
  - An expired declaration found through the World (world_flags) counts as O1. A supplier whose
    declaration on file has run out by an invoice's date is not an intake problem: the rules engine
    applies declaration dates when it answers questions.
  - These documents come from the same templates the agent was built to read, so these numbers show
    that the reading and checks work as designed, not how well it would read other layouts.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents import intake_agent  # noqa: E402

DOCS = REPO_ROOT / "data" / "documents"
GROUPS = {"M": "missing", "C": "confusing", "O": "out of date", "H": "handwritten",
          "T": "hidden instruction", "U": "unsupported"}


def _norm(codes: set[str]) -> set[str]:
    return {"C1/C6" if c in ("C1", "C6") else c for c in codes}


def expected_defects(t: dict) -> set[str]:
    want = {d["defect_id"] for d in t["defects"]} - {"O2"}
    if t["doc_type"] == "supplier_email" and t.get("email_variant") == "shipment":
        want.discard("C4")
    if t["doc_type"] == "purchase_order":
        want.discard("C5")
    if {"declaration_expired"} & set(t.get("world_flags", [])):
        want.add("O1")
    return _norm(want)


def load_sets() -> list[tuple[str, dict, Path]]:
    out = []
    for folder in sorted(p for p in DOCS.glob("output*") if (p / "ground_truth.json").exists()):
        if folder.name == "output_sample10":
            continue
        for t in json.loads((folder / "ground_truth.json").read_text(encoding="utf-8")):
            path = next((folder / "documents").glob(t["doc_id"] + "_*"))
            out.append((folder.name, t, path))
    return out


def evaluate() -> dict:
    stats = defaultdict(lambda: defaultdict(int))
    groups = defaultdict(lambda: defaultdict(int))
    errors = []
    for folder, t, path in load_sets():
        found = intake_agent.check_document(path.read_text(encoding="utf-8"))
        for lang in ("all", t["language"]):
            s = stats[lang]
            s["documents"] += 1
            s["doc_type_ok"] += found["doc_type"] == t["doc_type"]
            s["supplier_ok"] += found["supplier_id"] == t["supplier_id"]
            want_flag = bool(expected_defects(t))   # any visible problem means "hand to a person"
            got_flag = intake_agent._status(found["defects"]) == "flagged"
            s["action_ok"] += want_flag == got_flag
            s["missed_flag"] += want_flag and not got_flag
            s["false_flag"] += got_flag and not want_flag
            # origin per line, matched by position, where the answer key has a shown origin
            shown = {d["field"] for d in t["defects"]}
            for k, (item, read) in enumerate(zip(t["items"], found["items"])):
                if t["doc_type"] in ("purchase_order",) or (t["doc_type"] == "supplier_email" and
                                                             t.get("email_variant") == "shipment"):
                    continue
                if k == 0 and any(f.startswith("items[0].origin") for f in shown) or item.get("unsupported"):
                    continue
                s["origin_lines"] += 1
                s["origin_ok"] += read.get("origin") == item["origin_country"]
            want, got = expected_defects(t), _norm(set(found["defects"]) - {"O2"})
            for code in want | got:
                g = GROUPS[code[0]]
                groups[(lang, g)]["expected"] += code in want
                groups[(lang, g)]["found"] += code in got
                groups[(lang, g)]["correct"] += code in want and code in got
        if lang != "all" and (want != got) and len(errors) < 25:
            errors.append({"doc_id": t["doc_id"], "set": folder, "expected": sorted(want), "found": sorted(got)})
    return {"stats": stats, "groups": groups, "errors": errors}


def pct(a, b):
    return round(100 * a / b, 1) if b else None


def report(res: dict) -> dict:
    table = {}
    for lang, s in sorted(res["stats"].items(), key=lambda kv: (kv[0] != "all", kv[0])):
        table[lang] = {"documents": s["documents"], "doc_type_%": pct(s["doc_type_ok"], s["documents"]),
                       "supplier_%": pct(s["supplier_ok"], s["documents"]),
                       "origin_line_%": pct(s["origin_ok"], s["origin_lines"]), "origin_lines": s["origin_lines"],
                       "action_%": pct(s["action_ok"], s["documents"]),
                       "missed_flags": s["missed_flag"], "false_flags": s["false_flag"]}
    probs = {}
    for (lang, g), c in sorted(res["groups"].items()):
        probs.setdefault(lang, {})[g] = {"recall_%": pct(c["correct"], c["expected"]),
                                         "precision_%": pct(c["correct"], c["found"]), "expected": c["expected"]}
    return {"by_language": table, "problems": probs, "sample_differences": res["errors"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="save the results to this file")
    a = ap.parse_args()
    out = report(evaluate())
    print(f"{'':6}{'docs':>6}{'type%':>8}{'supplier%':>11}{'origin%':>9}{'action%':>9}{'missed':>8}{'false':>7}")
    for lang, r in out["by_language"].items():
        print(f"{lang:6}{r['documents']:>6}{r['doc_type_%']:>8}{r['supplier_%']:>11}{r['origin_line_%']:>9}"
              f"{r['action_%']:>9}{r['missed_flags']:>8}{r['false_flags']:>7}")
    print("\nProblems found (all languages): recall % / precision %")
    for g, r in out["problems"].get("all", {}).items():
        print(f"  {g:20} {r['recall_%']!s:>6} / {r['precision_%']!s:<6} ({r['expected']} expected)")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print("\nsaved", a.json)


if __name__ == "__main__":
    main()
