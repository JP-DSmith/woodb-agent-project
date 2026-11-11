"""Intake agent (T26): reads one supplier document and records what it says.

    document text (invoice, origin declaration, supplier email, purchase order; EN/FR/ES)
      -> find the document type, language and supplier (World/suppliers.csv)
      -> read each item line: part number, description, tariff code, country of origin
      -> check it against the World (bom_lines.csv) and look for the problems in the
         defect catalogue (M1 to U1, mcp_servers/document_records.py)
      -> write one record per item line to data/documents/document_records.csv
      -> hand the document to a person when anything needs checking

Safety rules
- Text inside a document is data, never an instruction. Hidden instructions are recorded as
  defects (T1 to T5) and the document goes to the security contact; nothing in it is acted on.
- The agent never decides that a part is North American. A clean document is recorded as
  "pending" (a person verifies it); a document with a problem is "flagged". A document can only
  ever move a boot from "qualifies" to "needs review" (see document_records.py).
- If the supplier or the document type can't be identified, nothing is recorded and a person
  gets the document.

The reading is rule-based: it follows the layouts of our supplier documents (data/documents/templates),
so it runs without an API key and gives the same result every time.
"""
from __future__ import annotations

import csv
import hashlib
import re
from datetime import date
from pathlib import Path

from mcp_servers import document_records as docs
from mcp_servers.cusma_tools import call_tool, normalise_tier

REPO_ROOT = Path(__file__).resolve().parent.parent
WORLD = REPO_ROOT / "prototypes" / "cusma_qualification" / "World"
NORTH_AMERICA = {"CA", "US", "MX"}

# ---------------------------------------------------------------- reference data
COUNTRIES = {
    "CA": ["canada", "canadá"], "US": ["united states", "états-unis", "etats-unis", "estados unidos", "usa"],
    "MX": ["mexico", "mexique", "méxico"], "CN": ["china", "chine"], "DE": ["germany", "allemagne", "alemania"],
    "IT": ["italy", "italie", "italia"], "TH": ["thailand", "thaïlande", "tailandia"],
    "VN": ["vietnam", "viêt nam", "viet nam"], "TW": ["taiwan", "taïwan", "taiwán"],
}
COUNTRY_LOOKUP = {name: code for code, names in COUNTRIES.items() for name in names}
COUNTRY_LOOKUP.update({code.lower(): code for code in COUNTRIES})

DOC_TYPE_HEADINGS = [
    ("invoice", r"commercial invoice|facture commerciale|factura comercial"),
    ("origin_declaration", r"declaration of origin|déclaration d'origine du fournisseur|declaración de origen del proveedor"),
    ("purchase_order", r"^\s*purchase order\b"),
]
LANGUAGE_HINTS = {
    "fr": r"facture|déclaration|fournisseur|objet :|bonjour|cordialement|vendeur|acheteur",
    "es": r"factura|declaración|proveedor|asunto:|hola|saludos|vendedor|comprador",
}
SECURITY = [  # (defect, language of the instruction, pattern)
    ("T5", "en", r"approved by [\w\s]*compliance|skip (the )?origin check|no need to verify|do not verify"),
    ("T5", "fr", r"approuvé par la conformité|ne pas vérifier l'origine"),
    ("T5", "es", r"aprobado por cumplimiento|omit(a|ir) la verificación"),
    ("T", "en", r"note to (the )?ai\b|\bai system\b|to the assistant\b"),
    ("T", "fr", r"système d'ia|systeme d'ia"),
    ("T", "es", r"sistema de ia"),
]
DECLARATION_ATTACHED = r"declaration is attached|déclaration d'origine est jointe|adjuntamos nuestra declaración"
UNSUPPORTED = r"boat fender|boat bumper|défense de bateau|defensa para barco"


def _read_csv(name: str) -> list[dict]:
    with open(WORLD / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_world() -> dict:
    """Suppliers and the products each one sells. Part numbers follow the supplier documents:
    <supplier_id>-NN, one per product, numbered in description order."""
    suppliers = {s["supplier_id"]: s for s in _read_csv("suppliers.csv")}
    products: dict[str, dict] = {}
    for r in _read_csv("bom_lines.csv"):
        products.setdefault(r["supplier_id"], {}).setdefault(
            (r["description"], r["hs_code"], r["origin_country"]),
            {"description": r["description"], "hs_code": r["hs_code"], "origin": r["origin_country"]})
    parts = {}
    for sid, d in products.items():
        for k, p in enumerate(sorted(d.values(), key=lambda x: x["description"]), 1):
            parts[f"{sid}-{k:02d}"] = {**p, "supplier_id": sid}
    return {"suppliers": suppliers, "parts": parts,
            "descriptions": {sid: {p["description"].lower() for p in d.values()} for sid, d in products.items()}}


_WORLD: dict | None = None


def world() -> dict:
    global _WORLD
    if _WORLD is None:
        _WORLD = load_world()
    return _WORLD


# ---------------------------------------------------------------- small readers
def _find(pattern: str, text: str, flags=re.I | re.M) -> str | None:
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None


def _money(value: str | None) -> float | None:
    if not value:
        return None
    m = re.search(r"-?[\d,]+\.\d+|-?\d+", value)
    return float(m.group(0).replace(",", "")) if m else None


def _iso_date(value: str | None) -> str | None:
    m = re.search(r"\d{4}-\d{2}-\d{2}", value or "")
    return m.group(0) if m else None


def read_country(raw: str) -> tuple[str | None, list[str]]:
    """Country name as shown -> (ISO code or None, defects). Handles scan errors (H1) and
    crossed-out values (H3: the value after the crossed-out one is read, but flagged)."""
    defects, text = [], (raw or "").strip()
    if not text or text in {"—", "-"}:
        return None, ["M1"]
    if "[crossed out:" in text.lower():
        defects.append("H3")
        text = re.sub(r"\[crossed out:[^\]]*\]", "", text, flags=re.I).strip()
    key = text.lower()
    if key not in COUNTRY_LOOKUP and re.search(r"[a-zéèêáíóúç][014]|[014][a-zéèêáíóúç]", key):
        fixed = key.translate(str.maketrans({"4": "a", "0": "o", "1": "i"}))
        if fixed in COUNTRY_LOOKUP:
            defects.append("H1")
            key = fixed
    return COUNTRY_LOOKUP.get(key), defects


def detect_doc_type(text: str) -> str:
    head = "\n".join(text.strip().splitlines()[:3]).lower()
    for doc_type, pattern in DOC_TYPE_HEADINGS:
        if re.search(pattern, head, re.I | re.M):
            return doc_type
    if re.search(r"^(from|de)\s?:", text, re.I | re.M) and re.search(r"^(subject|objet|asunto)\s?:", text, re.I | re.M):
        return "supplier_email"
    return "other"


def detect_language(text: str) -> str:
    low = text.lower()
    scores = {lang: len(re.findall(p, low)) for lang, p in LANGUAGE_HINTS.items()}
    lang = max(scores, key=scores.get)
    return lang if scores[lang] >= 2 else "en"


def find_supplier(text: str) -> str | None:
    """The supplier is the World supplier whose name appears in the document (longest name wins)."""
    low = text.lower()
    hits = [(len(s["name"]), sid) for sid, s in world()["suppliers"].items() if s["name"].lower() in low]
    return max(hits)[1] if hits else None


def _table(text: str, header: str) -> list[list[str]]:
    """Rows of the item table that follows the header line (columns separated by 2+ spaces)."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.match(header, line.strip(), re.I):
            rows = []
            for row in lines[i + 1:]:
                if not row.strip():
                    if rows:
                        break
                    continue
                rows.append([c.strip() for c in re.split(r"\s{2,}", row.strip())])
            return rows
    return []


# ---------------------------------------------------------------- per document type
def read_items(text: str, doc_type: str) -> list[dict]:
    items = []
    if doc_type in ("invoice", "purchase_order"):
        header = r"(line|ligne|línea)\s"
        for c in _table(text, header):
            if len(c) < 7:
                continue
            if doc_type == "purchase_order":
                origin, hs = "", ""
            elif len(c) >= 9:
                origin, hs = c[7], c[8]
            else:                          # origin column empty: the two spaces merge
                origin, hs = "", c[-1]
            items.append({"part_no": c[1], "description": c[2], "origin_raw": origin, "hs_code": hs,
                          "unit_price": c[5], "amount": c[6], "check_origin": doc_type == "invoice"})
    elif doc_type == "origin_declaration":
        for c in _table(text, r"(part no\.|n° pièce|no\. parte)\s"):
            if len(c) < 4:
                continue
            origin = c[3] if len(c) >= 5 else ""
            items.append({"part_no": c[0], "description": c[1], "hs_code": c[2], "origin_raw": origin,
                          "unit_price": "", "amount": "", "check_origin": True})
    elif doc_type == "supplier_email":
        part = _find(r"\((?:part|pièce|parte)\s+([A-Z0-9-]+)\)", text)
        material = (_find(r"(?:the|le|el)\s+(.+?)\s+\((?:part|pièce|parte)\s", text)
                    or _find(r"(?:pairs of|paires \(|pares \()\s*(.+?)[.)]", text) or "")
        origin = _find(r"(?:is made in|est fabriqué au|est fabriqué en|se fabrica en)\s+([^.\n]+)", text) or ""
        items.append({"part_no": part or "", "description": material, "hs_code": "", "origin_raw": origin,
                      "unit_price": "", "amount": "", "check_origin": bool(origin)})
    return items


def security_defects(text: str, doc_lang: str) -> list[str]:
    low = text.lower()
    found = []
    for code, lang, pattern in SECURITY:
        if re.search(pattern, low):
            if code == "T5":
                found.append("T5")
            else:
                found.append({"en": "T1", "fr": "T2", "es": "T3"}[lang] if lang == doc_lang else "T4")
    return sorted(set(found))


def check_document(text: str) -> dict:
    """Read a document and find its problems. Writes nothing."""
    doc_type = detect_doc_type(text)
    lang = detect_language(text)
    sid = find_supplier(text)
    out = {"doc_type": doc_type, "language": lang, "supplier_id": sid, "items": [], "defects": [],
           "doc_date": None, "declaration_start": None, "declaration_end": None, "notes": []}
    if doc_type == "other" or not sid:
        return out
    w, sup = world(), world()["suppliers"][sid]
    defects = set(security_defects(text, lang))
    if re.search(UNSUPPORTED, text, re.I):
        defects.add("U1")
    if "[handwritten:" in text.lower():
        defects.add("H2")

    # header: date, signature, declaration period
    if doc_type == "origin_declaration":
        sig_line = _find(r"^((?:signature|firma)\s?:.*)$", text) or ""
        date_raw = _find(r"(?:date|fecha)\s?:\s*(.*)$", sig_line)
        sig = _find(r"^(?:signature|firma)\s?:\s*(.*?)(?:\s{2,}|$)", sig_line)
        period = re.search(r"(\d{4}-\d{2}-\d{2})\D+(\d{4}-\d{2}-\d{2})", _find(r"^(?:blanket period|période globale|periodo general)\s?:\s*(.*)$", text) or "")
        if period:
            out["declaration_start"], out["declaration_end"] = period.group(1), period.group(2)
    elif doc_type == "supplier_email":
        date_raw = _find(r"^(?:date|fecha)\s?:\s*(.*)$", text)
        sig = "email"
    else:
        date_raw = _find(r"(?:^|\s)(?:date|fecha)\s?:\s*(\S*)\s*$", text)
        sig = _find(r"(?:authorized signature|signature autorisée|firma autorizada)\s?:\s*(.*)$", text) \
            if doc_type == "invoice" else "po"
    out["doc_date"] = _iso_date(date_raw)
    if not out["doc_date"] and doc_type != "supplier_email":
        defects.add("M3")
    if sig is not None and (not sig.strip() or set(sig.strip()) <= {"_"}):
        defects.add("M2")
    if out["declaration_end"] and out["doc_date"] and out["declaration_end"] < out["doc_date"]:
        defects.add("O1")
    if doc_type == "supplier_email" and re.search(DECLARATION_ATTACHED, text, re.I):
        defects.add("M4")

    # item lines
    currency = (_find(r"(?:currency|devise|moneda)\s?:\s*([A-Z]{3})", text) or "").upper()
    total_lines = 0.0
    for it in read_items(text, doc_type):
        code, line_defects = read_country(it["origin_raw"]) if it["check_origin"] else (None, [])
        if doc_type == "supplier_email" and "M1" in line_defects:
            line_defects = []
        defects.update(line_defects)
        part = w["parts"].get(it["part_no"])
        unsupported = bool(re.search(UNSUPPORTED, it["description"], re.I))
        if it["part_no"] and not unsupported and (part is None or part["supplier_id"] != sid):
            defects.add("C4")
        if it["description"] and not unsupported and it["description"].lower() not in w["descriptions"].get(sid, set()):
            defects.add("C2")
        expected = part["origin"] if part and part["supplier_id"] == sid else None
        if code and expected and code != expected:
            defects.add("C6" if code in NORTH_AMERICA and code != sup["country"] else "C1")
        for value in (it["unit_price"], it["amount"]):
            cur = re.search(r"\b([A-Z]{3})\b", value or "")
            if cur and currency and cur.group(1) != currency:
                defects.add("C3")
        total_lines += _money(it["amount"]) or 0.0
        out["items"].append({**it, "origin": code, "expected_origin": expected})
    # Totals are checked on invoices only: a purchase order's TOTAL includes freight that isn't listed.
    stated = _money(_find(r"^(?:subtotal|sous-total)\s?:\s*(.*)$", text)) if doc_type == "invoice" else None
    if stated is not None and out["items"] and abs(stated - total_lines) > 0.01:
        defects.add("C5")
    out["defects"] = sorted(defects)
    return out


# ---------------------------------------------------------------- records and reply
def _status(defects: list[str]) -> str:
    serious = [d for d in defects if docs.DEFECTS[d][1] != "record_with_note"]
    return "flagged" if serious else "pending"


def _handoff(defects: list[str]) -> str:
    if any(d.startswith("T") for d in defects):
        return "security_contact"
    if "U1" in defects:
        return "compliance_analyst"
    return "compliance_or_purchasing_contact"


def build_records(found: dict, doc_id: str, received_date: str, source_file: str = "") -> list[dict]:
    status = _status(found["defects"])
    note = docs.problems({"defect_ids": ";".join(found["defects"])})
    rows = []
    for it in found["items"] or [{"description": "", "hs_code": "", "origin": None}]:
        rows.append({
            "doc_id": doc_id, "received_date": received_date, "doc_type": found["doc_type"],
            "language": found["language"], "supplier_id": found["supplier_id"],
            "part_description": it.get("description", ""), "hs_code_shown": it.get("hs_code", ""),
            "origin_shown": it.get("origin") or "", "declaration_start": found["declaration_start"] or "",
            "declaration_end": found["declaration_end"] or "", "status": status,
            "defect_ids": ";".join(found["defects"]), "problem_note": note, "checked_by": "intake_agent",
            "source_file": source_file})
    return rows


TYPE_NAMES = {"invoice": "invoice", "origin_declaration": "origin declaration",
              "supplier_email": "supplier email", "purchase_order": "purchase order"}


def run(document: str, tier: str, model=None, language: str = "en", *, doc_id: str | None = None,
        received_date: str | None = None, source_file: str = "", write: bool = True,
        records_path: Path | None = None) -> dict:
    """Read one document, record it, and say what happens next.
    Returns the same shape as the specialist agents: {"agent", "answer", "trace", "escalation", ...}
    plus "found" (what was read) and "records" (the rows written)."""
    tier = normalise_tier(tier)
    found = check_document(document)
    doc_id = doc_id or "UPL-" + hashlib.sha256(document.encode("utf-8")).hexdigest()[:8].upper()
    base = {"agent": "intake_agent", "tier": tier, "language": language, "steps": 1, "found": found,
            "records": [], "doc_id": doc_id}
    trace = [{"tool": "read_document", "args": {"doc_id": doc_id},
              "result": {"ok": True, "data": {k: found[k] for k in ("doc_type", "language", "supplier_id", "defects")}}}]

    if found["doc_type"] == "other" and not found["supplier_id"] and len(document) < 400:
        # a typed message about a document ("please log the new invoice"), with no document in it
        return {**base, "trace": trace, "escalation": None,
                "answer": "Please attach or paste the supplier document (invoice, origin declaration, supplier "
                          "email or purchase order) and I'll read and record it."}
    if found["doc_type"] == "other" or not found["supplier_id"]:
        why = "unknown document type" if found["doc_type"] == "other" else "supplier not in our records"
        esc = call_tool("create_escalation", {"reason": "document_intake", "handoff_to": "compliance_or_purchasing_contact",
                                              "summary": f"{doc_id}: {why}; not recorded"}, tier)
        trace.append({"tool": "create_escalation", "args": {"reason": "document_intake"}, "result": esc})
        return {**base, "trace": trace, "escalation": esc.get("data"),
                "answer": "I couldn't match this document to one of our suppliers or document types, so I haven't "
                          "recorded it. I've passed it to a colleague to check."}

    path = Path(records_path or docs.RECORDS_FILE)
    already = any(r["doc_id"] == doc_id for r in docs.load_records(path))
    received = received_date or found["doc_date"] or date.today().isoformat()
    rows = build_records(found, doc_id, received, source_file)
    if write and not already:
        for row in rows:
            res = docs.append_record(row, path)
            trace.append({"tool": "record_document", "args": {"doc_id": doc_id, "supplier_id": row["supplier_id"]},
                          "result": res})
    status = rows[0]["status"]
    supplier_name = world()["suppliers"][found["supplier_id"]]["name"]
    who = "" if tier == "C" else f" from {supplier_name}"
    what = TYPE_NAMES[found["doc_type"]]
    if already:
        lead = f"This {what}{who} is already recorded, so I haven't added it again."
    else:
        n = len(rows)
        lead = f"I've read the {what}{who} and recorded {n} line{'s' if n != 1 else ''}."

    escalation = None
    if status == "flagged":
        handoff = _handoff(found["defects"])
        esc = call_tool("create_escalation", {"reason": "document_intake", "handoff_to": handoff,
                                              "summary": f"{doc_id}: {docs.problems({'defect_ids': ';'.join(found['defects'])})}"},
                        tier)
        escalation = esc.get("data")
        trace.append({"tool": "create_escalation", "args": {"reason": "document_intake", "handoff_to": handoff}, "result": esc})
        problems = [docs.DEFECTS[d][0].lower() for d in found["defects"]]
        tail = " It contains an instruction aimed at the assistant. I didn't act on it." \
            if handoff == "security_contact" else ""
        answer = (f"{lead} It needs checking: {', '.join(problems)}.{tail} I've passed it to a colleague, "
                  f"and it won't be counted as proof of origin until a person has checked it.")
    else:
        note = " Note: the supplier address differs from our records." if "O2" in found["defects"] else ""
        answer = f"{lead} I found no problems.{note} A person still needs to verify it before it's used as proof of origin."
    return {**base, "answer": answer, "trace": trace, "escalation": escalation, "records": rows}


class _IntakeAgent:
    name = "intake_agent"
    tools = ["create_escalation"]   # it also writes records through mcp_servers/document_records.py

    def run(self, question: str, tier: str, model=None, language: str = "en", **kw) -> dict:
        return run(question, tier, model=model, language=language, **kw)


AGENT = _IntakeAgent()
