"""Intake stress test (T34): the generated supplier documents with realistic formatting changes.

Each document is altered six ways (tabs, single spaces, lower case, shortened heading, blank lines
between rows, a confidentiality footer) and read again by the intake agent's rules alone, as it would
be without an API key. Each reading is sorted into:
  correct                   read correctly, right action (pending or flagged)
  flagged_for_person        flagged, so a person checks it (safe)
  handed_to_person_unread   not recorded, handed to a person (safe)
  silent_miss               a real problem recorded as clean (unsafe)
  silent_wrong_origin       recorded as clean with a wrong country of origin (unsafe)

    python evaluation/intake_stress_test.py                 # prints the tables
    python evaluation/intake_stress_test.py --json out.json # also saves the counts
"""
import sys, json, re, random, pathlib, collections as C
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from agents import intake_agent as ia
from evaluation.intake_accuracy import load_sets, expected_defects, _norm

def tabs(t):            # columns separated by tabs instead of spaces (e.g. copied from a spreadsheet)
    return re.sub(r" {2,}", "\t", t)
def single_space(t):    # columns squeezed to one space (e.g. PDF-to-text export)
    return re.sub(r" {2,}", " ", t)
def lower_case(t):      # everything in lower case (e.g. OCR output)
    return t.lower()
def short_heading(t):   # heading shortened: "COMMERCIAL INVOICE" -> "INVOICE" and so on
    return (t.replace("COMMERCIAL INVOICE", "INVOICE").replace("FACTURE COMMERCIALE", "FACTURE")
             .replace("FACTURA COMERCIAL", "FACTURA"))
def extra_lines(t):     # blank lines between table rows (e.g. email client re-wrapping)
    return re.sub(r"\n(?=\s*\d+\s{2,}SUP-|\s*SUP-\d{3}-\d{2}\s{2,})", "\n\n", t)
def signature_block(t): # a harmless footer added after the form
    return t + "\n\nThis message and any attachments are confidential and intended only for the addressee."
VARIANTS = {"tabs": tabs, "single_space": single_space, "lower_case": lower_case,
            "short_heading": short_heading, "extra_blank_lines": extra_lines, "confidentiality_footer": signature_block}

def outcome(t, found):
    want = expected_defects(t)
    got = _norm(set(found["defects"]) - {"O2"})
    if found["read_by"] == "unreadable" or not found["supplier_id"] or not found["items"]:
        return "handed_to_person_unread"
    status = ia._status(found["defects"])
    origins_ok = all((it.get("origin") in (None, i["origin_country"])) for it, i in zip(found["items"], t["items"])
                     if not i.get("unsupported"))
    if want and status == "pending":
        return "silent_miss"                 # a real problem recorded as clean
    if not origins_ok and status == "pending":
        return "silent_wrong_origin"
    if want <= got and status == ("flagged" if want else "pending"):
        return "correct"
    if status == "flagged":
        return "flagged_for_person"          # safe: a person looks at it (problem labels may differ)
    return "other"

res = C.defaultdict(lambda: C.defaultdict(C.Counter))
for name, fn in VARIANTS.items():
    for folder, t, path in load_sets():
        found = ia.check_document(fn(path.read_text(encoding="utf-8")))
        o = outcome(t, found)
        res[name]["all"][o] += 1; res[name][t["language"]][o] += 1
if len(sys.argv) > 2 and sys.argv[1] == "--json":
    json.dump({k: {l: dict(v) for l, v in d.items()} for k, d in res.items()}, open(sys.argv[2], "w"), indent=1)
order = ["correct", "flagged_for_person", "handed_to_person_unread", "silent_miss", "silent_wrong_origin", "other"]
print(f"{'variation':24}" + "".join(f"{o[:14]:>16}" for o in order))
for name in VARIANTS:
    a = res[name]["all"]; n = sum(a.values())
    print(f"{name:24}" + "".join(f"{100*a[o]/n:15.1f}%" for o in order))
print("\nby language (correct / safe hand-off / silent error):")
for name in VARIANTS:
    row = []
    for l in ("en", "fr", "es"):
        a = res[name][l]; n = sum(a.values())
        safe = a["flagged_for_person"] + a["handed_to_person_unread"]; bad = a["silent_miss"] + a["silent_wrong_origin"] + a["other"]
        row.append(f"{l} {100*a['correct']/n:5.1f}/{100*safe/n:5.1f}/{100*bad/n:4.1f}")
    print(f"  {name:24}", "   ".join(row))
