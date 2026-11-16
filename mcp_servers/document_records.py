"""Supplier document records: where the intake agent (T26) and the question tools meet.

The intake agent writes one row per supplier document it reads, to data/documents/document_records.csv.
The question tools (cusma_tools.py) read those rows, so an answer about a boot reflects the documents
received from that boot's suppliers.

Safety rule: an unchecked document never changes an answer on its own.
  pending   received, not checked yet                 -> a note only
  flagged   the intake agent found a problem          -> "qualifies" becomes "needs review", with the reason
  verified  a person checked it, no problem           -> listed as evidence; nothing changes
  verified  a person confirmed a serious defect       -> "needs review" until the supplier file is updated
  rejected  a person decided the document is invalid  -> a note only (e.g. a confirmed hidden instruction)
A document can only ever move "qualifies" to "needs review". It can never make a boot qualify.
"""
from __future__ import annotations

import csv
import os
import shutil
import tempfile
from contextlib import contextmanager
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# CUSMA_RECORDS_FILE points every reader and writer at another file (e.g. for a test or evaluation run).
RECORDS_FILE = Path(os.environ["CUSMA_RECORDS_FILE"]) if os.environ.get("CUSMA_RECORDS_FILE") else \
    REPO_ROOT / "data" / "documents" / "document_records.csv"

FIELDS = ["doc_id", "received_date", "doc_type", "language", "supplier_id", "part_description",
          "hs_code_shown", "origin_shown", "declaration_start", "declaration_end",
          "status", "defect_ids", "problem_note", "checked_by", "source_file"]
REQUIRED = ["doc_id", "received_date", "doc_type", "supplier_id", "status"]
DOC_TYPES = {"invoice", "origin_declaration", "supplier_email", "purchase_order", "other"}
STATUSES = {"pending", "flagged", "verified", "rejected"}
# The document defect catalogue (T10 spec), with the intake action each one calls for.
DEFECTS = {
    "M1": ("No country of origin", "flag_for_person"), "M2": ("No signature", "flag_for_person"),
    "M3": ("No date", "flag_for_person"), "M4": ("No declaration at all", "flag_for_person"),
    "C1": ("Origin conflict", "flag_for_person"), "C2": ("Vague description", "flag_for_person"),
    "C3": ("Mixed currencies", "flag_for_person"), "C4": ("Wrong part number", "flag_for_person"),
    "C5": ("Totals do not add up", "flag_for_person"), "C6": ("Origin does not match supplier country", "flag_for_person"),
    "O1": ("Expired declaration", "flag_for_person"), "O2": ("Old supplier address", "record_with_note"),
    "H1": ("Scan errors", "record_if_clear_else_flag"), "H2": ("Margin note", "flag_for_person"),
    "H3": ("Crossed-out value", "flag_for_person"),
    "T1": ("Hidden instruction (EN)", "flag_security"), "T2": ("Hidden instruction (FR)", "flag_security"),
    "T3": ("Hidden instruction (ES)", "flag_security"), "T4": ("Instruction in a different language", "flag_security"),
    "T5": ("Fake authority", "flag_security"), "U1": ("Boat bumper document", "reject_unsupported"),
}
SERIOUS = {"flag_for_person", "flag_security", "record_if_clear_else_flag"}   # can open a review
CHECKED_BY = {"", "intake_agent", "person"}


def defect_list(row: dict) -> list[str]:
    """defect_ids is one cell: 'C6;T1' (separated by ; or ,)."""
    return [d.strip().upper() for d in str(row.get("defect_ids") or "").replace(",", ";").split(";") if d.strip()]


def validate(row: dict) -> list[str]:
    """Reasons a record is not valid (empty list = valid)."""
    why = [f"missing {k}" for k in REQUIRED if not str(row.get(k) or "").strip()]
    for key, ok in (("doc_type", DOC_TYPES), ("status", STATUSES), ("checked_by", CHECKED_BY)):
        if str(row.get(key) or "").strip() not in ok and key not in [w.split()[-1] for w in why]:
            why.append(f"{key} '{row.get(key)}' is not one of {sorted(ok - {''})}")
    for key in ("received_date", "declaration_start", "declaration_end"):
        v = str(row.get(key) or "").strip()
        if v:
            try:
                date.fromisoformat(v)
            except ValueError:
                why.append(f"{key} '{v}' is not a YYYY-MM-DD date")
    bad = [d for d in defect_list(row) if d not in DEFECTS]
    if bad:
        why.append(f"defect_ids {bad} are not in the defect catalogue (M1 to U1)")
    if row.get("status") == "flagged" and not defect_list(row):
        why.append("a flagged record needs at least one defect_id")
    unknown = set(row) - set(FIELDS)
    if unknown:
        why.append(f"unknown fields {sorted(unknown)}")
    return why


def load_records(path: Path | None = None) -> list[dict]:
    """All valid records (invalid rows are skipped). No file yet = no records."""
    path = Path(path or RECORDS_FILE)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f) if not validate(r)]


def append_record(row: dict, path: Path | None = None) -> dict:
    """For the intake agent: check a record and add it to the file. Returns {"ok": ...}."""
    why = validate(row)
    if why:
        return {"ok": False, "error": {"code": "BAD_INPUT", "message": "; ".join(why)}}
    path = Path(path or RECORDS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})
    return {"ok": True, "data": {"doc_id": row["doc_id"], "status": row["status"]}}


def records_for(supplier_ids, as_of: date, path: Path | None = None) -> list[dict]:
    """Records for these suppliers received on or before the date."""
    ids = set(supplier_ids)
    return [r for r in load_records(path) if r["supplier_id"] in ids and date.fromisoformat(r["received_date"]) <= as_of]


def opens_review(r: dict) -> bool:
    """Does this record put a qualifying boot into 'needs review'?
    Flagged by intake, or confirmed by a person, with at least one defect that needs a person."""
    serious = any(DEFECTS[d][1] in SERIOUS for d in defect_list(r))
    return r["status"] == "flagged" or (r["status"] == "verified" and serious)


def problems(r: dict) -> str:
    return ", ".join(f"{d} {DEFECTS[d][0].lower()}" for d in defect_list(r))


def describe(r: dict) -> str:
    what = problems(r) or "no problem found"
    who = "checked by a person" if r["status"] in ("verified", "rejected") else "not yet checked by a person"
    return f"Document {r['doc_id']} ({r['doc_type']} from {r['supplier_id']}, received {r['received_date']}): " \
           f"{r['status']}, {what}; {who}"


@contextmanager
def temporary_records(start_from_real: bool = True, keep: bool = False):
    """Use a throw-away copy of the records file inside the block, so an evaluation or attack run that
    sends documents to the intake agent never writes to the real records.

        with temporary_records() as path:
            ...run the tests...           # reads and writes go to `path`

    start_from_real  begin with a copy of today's records (so answers match normal use); False = empty
    keep             leave the copy on disk afterwards (its path is the one yielded), e.g. to inspect it"""
    global RECORDS_FILE
    real = RECORDS_FILE
    folder = Path(tempfile.mkdtemp(prefix="cusma_records_"))
    temp = folder / "document_records.csv"
    if start_from_real and real.exists():
        shutil.copyfile(real, temp)
    RECORDS_FILE = temp
    try:
        yield temp
    finally:
        RECORDS_FILE = real
        if not keep:
            shutil.rmtree(folder, ignore_errors=True)
