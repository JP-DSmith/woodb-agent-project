"""Fill every document template with sample values to check that no field is missing."""
import json, string, pathlib, sys
base = pathlib.Path(__file__).parent
spec = json.loads((base / "T10_spec.json").read_text(encoding="utf-8"))
sample = dict(invoice_no="INV-2026-0412", invoice_date="2026-10-14", supplier_name="Valcourt Caoutchouc Ltée",
  supplier_address="1200 rue Industrielle, Drummondville, QC", supplier_tax_id="123456789 RT0001",
  buyer_address="45 Factory Rd, Kitchener, ON", po_no="PO-8831", incoterm="FCA", currency="CAD",
  subtotal="12,400.00", freight="350.00", total="12,750.00", ship_via="Purolator", ship_date="2026-10-15",
  signature="M. Tremblay", decl_no="DEC-0077", period_start="2026-01-01", period_end="2026-12-31",
  signer_name="Marie Tremblay", signer_title="Directrice", decl_date="2026-01-05", sender_name="Marie Tremblay",
  sender_email="m.tremblay@example.com", recipient_name="Sam Lee", recipient_first="Sam",
  recipient_email="s.lee@example.com", email_date="2026-10-15", material="natural rubber compound",
  part_no="RC-110", origin_country="CA", body_detail="", extra_note="", po_date="2026-10-01",
  required_date="2026-10-20", buyer_contact="Sam Lee")
line = dict(line_no="1", part_no="RC-110", description="Natural rubber compound", qty="500", unit="kg",
  unit_price="24.80", amount="12,400.00", origin_country="CA", hs_code="4005.99", origin_criterion="B")
def fill(name, vals):
    return string.Template((base / "templates" / name).read_text()).substitute(vals)
count = 0
for dt, d in spec["document_types"].items():
    groups = d.get("variants", {"main": d.get("files")})
    for files in groups.values():
        for lang, f in files.items():
            vals = dict(sample)
            if "line_template" in d:
                vals[d["line_slot"]] = fill(d["line_template"], line)
            out = fill(f, vals)
            count += 1
print(f"rendered {count} templates OK")
