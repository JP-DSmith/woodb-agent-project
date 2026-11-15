"""Intake stress test (T34): the generated supplier documents with realistic formatting changes.

Each document is altered six ways (tabs, single spaces, lower case, shortened heading, blank lines
between rows, a confidentiality footer) and read again by the intake agent's rules alone, as it would
be without an API key. Each reading is sorted into one outcome:

  safe
    correct                   read in full, right status (pending or flagged), right problems found
    flagged_for_person        flagged, so a person checks it (problem labels may differ)
    handed_to_person_unread   nothing recorded; the document goes to a person
  unsafe (any of these fails the test)
    silent_miss               recorded as pending (clean) although the document has a real problem
    silent_partial_read       recorded as pending with fewer or more item lines than the document has
    silent_wrong_origin       recorded as pending with a country of origin that differs from the answer key

A flagged document is always safe: a person reviews it before it counts as proof of origin.
Item lines are counted against what the document shows: a supplier email shows only its first item
(the generator writes one), even when the answer key lists more.

    python evaluation/intake_stress_test.py                  # prints the tables; exit code 1 if anything is unsafe
    python evaluation/intake_stress_test.py --json out.json  # also saves the counts

testing/test_intake_stress.py runs the same check in the normal test suite.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents import intake_agent as ia  # noqa: E402
from evaluation.intake_accuracy import _norm, expected_defects, load_sets  # noqa: E402

LANGUAGES = ("en", "fr", "es")
SAFE = ("correct", "flagged_for_person", "handed_to_person_unread")
UNSAFE = ("silent_miss", "silent_partial_read", "silent_wrong_origin")
OUTCOMES = SAFE + UNSAFE


# ---------------------------------------------------------------- the six formatting changes
def tabs(text: str) -> str:
    """Columns separated by tabs instead of spaces (e.g. copied from a spreadsheet)."""
    return re.sub(r" {2,}", "\t", text)


def single_space(text: str) -> str:
    """Columns squeezed to one space (e.g. a PDF-to-text export)."""
    return re.sub(r" {2,}", " ", text)


def lower_case(text: str) -> str:
    """Everything in lower case (e.g. OCR output)."""
    return text.lower()


def short_heading(text: str) -> str:
    """Heading shortened: "COMMERCIAL INVOICE" -> "INVOICE", and the French and Spanish equivalents."""
    return (text.replace("COMMERCIAL INVOICE", "INVOICE").replace("FACTURE COMMERCIALE", "FACTURE")
                .replace("FACTURA COMERCIAL", "FACTURA"))


def extra_blank_lines(text: str) -> str:
    """Blank lines between table rows (e.g. an email client re-wrapping the text)."""
    return re.sub(r"\n(?=\s*\d+\s{2,}SUP-|\s*SUP-\d{3}-\d{2}\s{2,})", "\n\n", text)


def confidentiality_footer(text: str) -> str:
    """A harmless footer added after the form."""
    return text + "\n\nThis message and any attachments are confidential and intended only for the addressee."


VARIANTS = {"tabs": tabs, "single_space": single_space, "lower_case": lower_case,
            "short_heading": short_heading, "extra_blank_lines": extra_blank_lines,
            "confidentiality_footer": confidentiality_footer}


# ---------------------------------------------------------------- scoring one reading
def visible_items(truth: dict) -> list[dict]:
    """The item lines the document actually shows. A supplier email shows only its first item."""
    return truth["items"][:1] if truth["doc_type"] == "supplier_email" else truth["items"]


def outcome(truth: dict, found: dict) -> str:
    """Sort one reading into an outcome (see the module docstring)."""
    if found["read_by"] == "unreadable" or not found["supplier_id"] or not found["items"]:
        return "handed_to_person_unread"
    want = expected_defects(truth)
    got = _norm(set(found["defects"]) - {"O2"})
    expected_items = visible_items(truth)
    full_read = len(found["items"]) == len(expected_items)
    if ia._status(found["defects"]) == "flagged":
        return "correct" if want and want <= got and full_read else "flagged_for_person"
    # recorded as pending: it must be clean, read in full, with the right origins
    if want:
        return "silent_miss"
    if not full_read:
        return "silent_partial_read"
    wrong = [i for i, (item, read) in enumerate(zip(expected_items, found["items"]))
             if not item.get("unsupported") and read.get("origin") not in (None, item["origin_country"])]
    return "silent_wrong_origin" if wrong else "correct"


def run_stress(variants: dict | None = None, sets: list | None = None) -> dict:
    """Read every document under every variation. Returns
    {"counts": {variant: {language|"all": Counter}}, "unsafe": [{variant, doc_id, outcome, ...}]}."""
    variants = VARIANTS if variants is None else variants
    sets = load_sets() if sets is None else sets
    counts: dict = defaultdict(lambda: defaultdict(Counter))
    unsafe = []
    for name, change in variants.items():
        for folder, truth, path in sets:
            found = ia.check_document(change(path.read_text(encoding="utf-8")))
            result = outcome(truth, found)
            counts[name]["all"][result] += 1
            counts[name][truth["language"]][result] += 1
            if result in UNSAFE:
                unsafe.append({"variant": name, "set": folder, "doc_id": truth["doc_id"], "outcome": result,
                               "items_expected": len(visible_items(truth)), "items_read": len(found["items"]),
                               "defects_expected": sorted(expected_defects(truth)), "defects_found": found["defects"]})
    return {"counts": counts, "unsafe": unsafe}


def pct(part: int, whole: int) -> float:
    """Percentage, 0.0 for an empty group instead of a division by zero."""
    return round(100 * part / whole, 1) if whole else 0.0


def report(results: dict) -> None:
    counts = results["counts"]
    print(f"{'variation':24}" + "".join(f"{o[:15]:>17}" for o in OUTCOMES))
    for name, groups in counts.items():
        row, n = groups["all"], sum(groups["all"].values())
        print(f"{name:24}" + "".join(f"{pct(row[o], n):16.1f}%" for o in OUTCOMES))
    print("\nby language (correct / safe hand-off / unsafe):")
    for name, groups in counts.items():
        parts = []
        for lang in LANGUAGES:
            row = groups.get(lang, Counter())
            n = sum(row.values())
            if not n:
                parts.append(f"{lang}  no documents")
                continue
            handoff = row["flagged_for_person"] + row["handed_to_person_unread"]
            bad = sum(row[o] for o in UNSAFE)
            parts.append(f"{lang} {pct(row['correct'], n):5.1f}/{pct(handoff, n):5.1f}/{pct(bad, n):4.1f}")
        print(f"  {name:24}", "   ".join(parts))
    print(f"\nunsafe results: {len(results['unsafe'])}")
    for u in results["unsafe"][:20]:
        print(f"  {u['variant']:24} {u['doc_id']}  {u['outcome']}  items {u['items_read']}/{u['items_expected']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", help="save the counts and any unsafe results to this file")
    args = ap.parse_args(argv)
    results = run_stress()
    report(results)
    if args.json:
        payload = {"counts": {name: {group: dict(c) for group, c in groups.items()}
                              for name, groups in results["counts"].items()},
                   "unsafe": results["unsafe"]}
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1)
        print("saved", args.json)
    return 1 if results["unsafe"] else 0


if __name__ == "__main__":
    sys.exit(main())
