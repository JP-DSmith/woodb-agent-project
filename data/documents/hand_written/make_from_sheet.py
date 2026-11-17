"""Turn the hand-written unseen-layout documents (T26) into test files.

    python data/documents/hand_written/make_from_sheet.py

Reads T26_Unseen_Layout_Documents.xlsx (sheet "Documents") in this folder and writes:
    documents/<doc id>_<language>.txt   the document text, exactly as written
    ground_truth.json                  the answer key for each document

TEST SET ONLY. These documents were written by hand to check how the intake agent copes with layouts and
languages it has never seen. Never use them to build, tune or prompt the intake agent; that would make the
test meaningless. The two example rows (EX-01, EX-02) were written as instructions for the sheet and are
left out. Empty rows are skipped.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHEET = HERE / "T26_Unseen_Layout_Documents.xlsx"
CODES = {"M1", "M2", "M3", "M4", "C1", "C2", "C3", "C4", "C5", "C6", "O1", "O2",
         "H1", "H2", "H3", "T1", "T2", "T3", "T4", "T5", "U1"}
ACTIONS = {"record", "record_with_note", "flag_for_person", "flag_security", "reject_unsupported"}


def _clean(value) -> str:
    return "" if value is None else str(value).strip()


def parse_parts(cell: str) -> dict[str, str]:
    """'SUP-031-01 = CN; SUP-031-02 = CN' -> {'SUP-031-01': 'CN', 'SUP-031-02': 'CN'}"""
    return {p.upper(): c.upper() for p, c in re.findall(r"(SUP-\d{3}-\d{2})\s*=\s*([A-Za-z]{2})\b", cell)}


def parse_problems(cell: str) -> list[str]:
    codes = [c.upper() for c in re.split(r"[\s,;]+", cell) if c and c.lower() != "none"]
    bad = [c for c in codes if c not in CODES]
    if bad:
        raise ValueError(f"unknown problem codes {bad}")
    return sorted(set(codes))


def rows_from_sheet(path: Path = SHEET) -> list[dict]:
    from openpyxl import load_workbook
    ws = load_workbook(path, read_only=True, data_only=True)["Documents"]
    rows = list(ws.iter_rows(values_only=True))
    header = [_clean(h) for h in rows[0]]
    col = {name: next(i for i, h in enumerate(header) if h.startswith(name))
           for name in ("Doc ID", "Written by", "How written", "Layout", "Language", "Supplier ID", "Document text",
                        "Doc type", "Parts and TRUE origins", "Problems planted", "Hidden instruction?",
                        "Expected action", "Notes")}
    out = []
    for r in rows[1:]:
        get = lambda name: _clean(r[col[name]]) if col[name] < len(r) else ""
        doc_id, text = get("Doc ID"), get("Document text")
        if not doc_id or not text or doc_id.startswith("EX-"):
            continue
        action = get("Expected action")
        if action not in ACTIONS:
            raise ValueError(f"{doc_id}: unknown expected action {action!r}")
        out.append({
            "doc_id": doc_id,
            "written_by": get("Written by").upper(),
            "how_written": get("How written"),
            "layout": get("Layout"),
            "language": get("Language").lower(),
            "supplier_id": get("Supplier ID").upper() if get("Supplier ID").upper().startswith("SUP-") else None,
            "doc_type": get("Doc type"),
            "parts": parse_parts(get("Parts and TRUE origins")),
            "problems": parse_problems(get("Problems planted")),
            "hidden_instruction": get("Hidden instruction?").lower() == "yes",
            "expected_action": action,
            "notes": get("Notes"),
            "text": text.replace("\r\n", "\n"),
        })
    return out


def main() -> None:
    rows = rows_from_sheet()
    folder = HERE / "documents"
    folder.mkdir(exist_ok=True)
    for old in folder.glob("*.txt"):
        old.unlink()
    truth = []
    for row in rows:
        name = f"{row['doc_id']}_{row['language']}.txt"
        (folder / name).write_text(row["text"].rstrip() + "\n", encoding="utf-8", newline="\n")
        truth.append({k: v for k, v in row.items() if k != "text"} | {"file": f"documents/{name}"})
    (HERE / "ground_truth.json").write_text(json.dumps(truth, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(truth)} documents and ground_truth.json")


if __name__ == "__main__":
    main()
