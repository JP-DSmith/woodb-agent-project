"""Tools the specialist agents call (T27).

Every tool:
  * takes the asker's tier ("A", "B" or "C"). The tier comes from the login (or the demo
    "Asking as" selector), never from the message, and the agent cannot change it;
  * gets every number from the rules engine or the World files, never from the model;
  * returns the rules engine's envelope:
        {"ok": True,  "data": ..., "trace": [...], "warnings": [...], "rules_version": ...}
        {"ok": False, "error": {"code": ..., "message": ...}}
    error codes: NOT_FOUND, NOT_SUPPORTED, MISSING_DATA, NOT_ALLOWED, BAD_INPUT;
  * hides what the tier may not see, following config/policy.yaml:
        A  sees everything
        B  no costs (supplier prices, labour, overhead, net cost); supplier names shown
        C  no costs and no supplier names or sourcing alternatives
    This is the first line of protection. The reviewer (security/output_guardrails.py) is the second.

Tool names follow the names the reviewer already recognises (import_duty, create_escalation, ...).
"""
from __future__ import annotations

import copy
import csv
import itertools
import re
from datetime import date
from pathlib import Path

from mcp_servers import document_records as docs
from prototypes.cusma_qualification.Engine import rules_engine as eng

REPO_ROOT = Path(__file__).resolve().parent.parent
WORLD_FOLDER = REPO_ROOT / "prototypes" / "cusma_qualification" / "World"
POLICY_FILE = REPO_ROOT / "config" / "policy.yaml"

# Used when config/policy.yaml is missing or PyYAML is not installed (same values as the policy).
DEFAULT_TIERS = {
    "A": {"costs": "show", "supplier_names": "show", "cert_draft": "route_to_signer"},
    "B": {"costs": "hide", "supplier_names": "show", "cert_draft": "route_to_signer"},
    "C": {"costs": "hide", "supplier_names": "hide", "cert_draft": "hand_to_person"},
}
UNKNOWN_TIER = "C"


def load_tiers(path: Path = POLICY_FILE) -> dict:
    try:
        import yaml  # optional
        tiers = (yaml.safe_load(path.read_text()) or {}).get("tiers") or {}
        return {t: {**DEFAULT_TIERS.get(t, DEFAULT_TIERS["C"]), **rules} for t, rules in tiers.items()} or DEFAULT_TIERS
    except Exception:
        return DEFAULT_TIERS


TIERS = load_tiers()


# The question datasets (T15/T22) label tiers T1/T2/T3; the policy uses A/B/C.
TIER_ALIASES = {"T1": "A", "T2": "B", "T3": "C"}


def normalise_tier(tier: str | None) -> str:
    """A/B/C as in config/policy.yaml. T1/T2/T3 are accepted; anything else is the unknown-user tier."""
    t = str(tier or "").strip().upper()
    t = TIER_ALIASES.get(t, t)
    return t if t in TIERS else UNKNOWN_TIER


def _rules(tier: str | None) -> dict:
    return TIERS[normalise_tier(tier)] if normalise_tier(tier) in TIERS else DEFAULT_TIERS["C"]


def _read(name: str) -> list[dict]:
    with open(WORLD_FOLDER / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


ORDERS = {r["po_number"]: r for r in _read("orders.csv")}
CUSTOMERS = {r["customer_id"]: r for r in _read("customers.csv")}
SUPPLIERS = eng.SUPPLIERS

# Placeholders until the company file exists (Northfield's own details are not in the World yet).
COMPANY = {
    "name": "Northfield Rubber Footwear Inc.",
    "address": "[EXPORTER ADDRESS - to be added to the company file]",
    "certifier": "[NAME AND TITLE OF AUTHORIZED SIGNER]",
}

COST_KEYS = {"cost_cad", "labour_cad", "overhead_cad", "net_cost_cad", "vnm_cad", "new_cost_cad"}
SUPPLIER_KEYS = {"supplier_id", "supplier_name", "new_supplier_id"}


def _err(code: str, message: str) -> dict:
    return {"ok": False, "error": {"code": code, "message": message}}


def _ok(data, trace, warnings=None) -> dict:
    return {"ok": True, "data": data, "trace": trace, "warnings": warnings or [],
            "rules_version": eng.RULES["rules_version"]}


def _as_of(value) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(value) if value else date.today()


# ------------------------------------------------------------------ masking
_NET_COST = re.compile(r"Net cost = [^;]*?CAD")
_RVC_SUM = re.compile(r"RVC = \([^)]*\) / [\d.,]+ = ")
_MONEY = re.compile(r"\b\d[\d,]*\.\d{2} CAD\b")


def _mask_step(step: str, hide_costs: bool, hide_names: bool) -> str:
    """Mask one human-readable rule step. Percentages and verdicts stay; internal money does not."""
    if hide_costs:
        step = _NET_COST.sub("Net cost = [confidential]", step)
        step = _RVC_SUM.sub("RVC = ", step)
        step = _MONEY.sub("[confidential]", step)
    if hide_names:
        step = re.sub(r"SUP-\d{3}", "[supplier]", step)
        for s in SUPPLIERS.values():
            step = step.replace(s["name"], "[supplier]")
    return step


def _mask_value(obj, hide_costs: bool, hide_names: bool):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if hide_costs and k in COST_KEYS:
                continue
            if hide_names and k in SUPPLIER_KEYS:
                continue
            out[k] = _mask_value(v, hide_costs, hide_names)
        return out
    if isinstance(obj, list):
        return [_mask_value(v, hide_costs, hide_names) for v in obj]
    if isinstance(obj, str) and (hide_costs or hide_names):
        return _mask_step(obj, hide_costs, hide_names)
    return obj


def mask(result: dict, tier: str | None) -> dict:
    """Remove what this tier may not see from a tool result (data, trace and warnings)."""
    r = _rules(tier)
    hide_costs, hide_names = r.get("costs") == "hide", r.get("supplier_names") == "hide"
    if not result.get("ok") or not (hide_costs or hide_names):
        return result
    out = copy.deepcopy(result)
    out["data"] = _mask_value(out.get("data"), hide_costs, hide_names)
    out["trace"] = [_mask_step(s, hide_costs, hide_names) for s in out.get("trace", [])]
    out["warnings"] = [_mask_step(s, hide_costs, hide_names) for s in out.get("warnings", [])]
    if hide_costs or hide_names:
        out["warnings"].append("Some fields are confidential for this user and were removed.")
    return out


# ------------------------------------------------------------------ supplier documents
def _boot_suppliers(sku: str, when: date) -> list[str]:
    m = eng.get_materials(sku, when)
    return sorted({l["supplier_id"] for l in m["data"]}) if m.get("ok") else []


def _apply_documents(res: dict, sku: str, when: date) -> dict:
    """Add the supplier documents received for this boot to a check_origin or duty result.
    Only ever moves 'qualifies' to 'needs_review' (see document_records.py)."""
    if not res.get("ok"):
        return res
    recs = docs.records_for(_boot_suppliers(sku, when), when)
    if not recs:
        return res
    data = res["data"]
    data["documents"] = [{k: r[k] for k in ("doc_id", "doc_type", "supplier_id", "received_date", "status", "defect_ids")}
                         for r in recs]
    res["trace"] = res["trace"] + [docs.describe(r) for r in recs]
    blocking = [r for r in recs if docs.opens_review(r)]
    if blocking and data.get("verdict") == "qualifies":
        reason = "Supplier document problem: " + "; ".join(
            f"{r['doc_id']} {docs.problems(r) or 'flagged'} ({r['supplier_id']})" for r in blocking)
        data["verdict"], data["verdict_before_documents"] = "needs_review", "qualifies"
        if "escalate_reason" in data:
            data["escalate_reason"] = reason
        if "duty_payable_usd" in data:
            data["duty_payable_usd"] = data["duty_if_not_qualifying_usd"]
            res["warnings"] = res.get("warnings", []) + ["Verdict needs review; duty shown both ways until a person confirms"]
        res["trace"].append(f"Verdict changed from qualifies to needs_review: {reason}. A person must check the document.")
        res["warnings"] = res.get("warnings", []) + [reason + ". A person must check it before this boot is treated as qualifying."]
    elif blocking:
        res["warnings"] = res.get("warnings", []) + [f"Also: open supplier document problems ({', '.join(r['doc_id'] for r in blocking)})."]
    return res


# ------------------------------------------------------------------ tools
def get_boot(sku: str, tier: str) -> dict:
    """Boot specifications (style, height, upper, toe, gender, price)."""
    return mask(eng.get_boot(sku.upper()), tier)


def get_materials(sku: str, tier: str, as_of: str | None = None) -> dict:
    """The boot's parts: description, tariff code, origin, and whether each counts as North American."""
    res = eng.get_materials(sku.upper(), _as_of(as_of))
    if res["ok"]:
        recs = docs.records_for({l["supplier_id"] for l in res["data"]}, _as_of(as_of))
        for line in res["data"]:
            line["supplier_name"] = SUPPLIERS[line["supplier_id"]]["name"]
            mine = [r for r in recs if r["supplier_id"] == line["supplier_id"]]
            if mine:
                line["documents"] = [{k: r[k] for k in ("doc_id", "doc_type", "status", "defect_ids")} for r in mine]
    return mask(res, tier)


def check_origin(sku: str, tier: str, as_of: str | None = None) -> dict:
    """CUSMA verdict for one boot: tariff code, tariff shift, regional value content, verdict."""
    when = _as_of(as_of)
    return mask(_apply_documents(eng.check_origin(sku.upper(), when), sku.upper(), when), tier)


def find_alternative_suppliers(sku: str, line_no: int, tier: str) -> dict:
    """Other suppliers in our list that sell the same type of part as this line of the boot."""
    if _rules(tier).get("supplier_names") == "hide":
        return _err("NOT_ALLOWED", "Sourcing alternatives are not shared with external users.")
    m = eng.get_materials(sku.upper(), date.today())
    if not m["ok"]:
        return m
    line = next((l for l in m["data"] if int(l["line_no"]) == int(line_no)), None)
    if line is None:
        return _err("NOT_FOUND", f"{sku} has no line {line_no}")
    alts = [{"supplier_id": s["supplier_id"], "supplier_name": s["name"], "country": s["country"],
             "declaration_status": s["declaration_status"], "declaration_expiry": s["declaration_expiry"]}
            for s in SUPPLIERS.values()
            if s.get("component") == line["component"] and s["supplier_id"] != line["supplier_id"]]
    return mask(_ok({"sku": sku.upper(), "line_no": int(line_no), "component": line["component"],
                     "current_supplier_id": line["supplier_id"], "current_country": line["origin_country"],
                     "alternatives": alts},
                    [f"{len(alts)} other suppliers sell '{line['component']}'"]), tier)


def what_if(sku: str, changes: list, tier: str, as_of: str | None = None) -> dict:
    """Re-check a boot with changes to its parts. Each change: {"line": n} plus one or more of
    "new_supplier_id", "origin_country", "new_cost_cad". Returns the verdict before and after."""
    r = _rules(tier)
    allowed = {"line", "origin_country"} | ({"new_supplier_id"} if r.get("supplier_names") != "hide" else set()) \
        | ({"new_cost_cad"} if r.get("costs") != "hide" else set())
    if not isinstance(changes, list) or not changes:
        return _err("BAD_INPUT", "changes must be a non-empty list")
    for ch in changes:
        extra = set(ch) - allowed
        if extra:
            return _err("NOT_ALLOWED", f"This user may not change: {', '.join(sorted(extra))}")
        if "new_supplier_id" in ch and ch["new_supplier_id"] not in SUPPLIERS:
            return _err("NOT_FOUND", f"No supplier with id {ch['new_supplier_id']}")
    return mask(eng.what_if(sku.upper(), changes, _as_of(as_of)), tier)


def _special_tariffs(code: str, order_value: float) -> list[dict]:
    rate = eng.RULES["us_duty"].get(code, {}).get("applies_anyway", 0.0)
    if not rate:
        return []
    return [{"name": "Section 338", "rate": rate, "applies": "even_if_qualifying",
             "amount_usd": round(order_value * rate, 2)}]


def import_duty(sku: str, quantity: int, unit_price_usd: float, tier: str, as_of: str | None = None) -> dict:
    """US duty on a shipment: payable on the date given, and what it would be if the boot did or did not qualify."""
    when = _as_of(as_of)
    res = _apply_documents(eng.duty(sku.upper(), int(quantity), float(unit_price_usd), when), sku.upper(), when)
    if res["ok"]:
        res["data"]["special_tariffs"] = _special_tariffs(res["data"]["hs_code"], res["data"]["order_value_usd"])
    return mask(res, tier)


def get_order(po_number: str, tier: str) -> dict:
    """A customer order: customer, boot, quantity, unit price, ship date."""
    o = ORDERS.get(po_number.upper())
    if not o:
        return _err("NOT_FOUND", f"No order with number {po_number}")
    c = CUSTOMERS.get(o["customer_id"], {})
    data = {**o, "quantity": int(o["quantity"]), "unit_price_usd": float(o["unit_price_usd"]),
            "customer_company": c.get("company", ""), "customer_city": c.get("city", ""), "customer_state": c.get("state", "")}
    return mask(_ok(data, [f"Order {o['po_number']}: {o['quantity']} pairs of {o['sku']}, ships {o['ship_date']}"]), tier)


def order_duty(po_number: str, tier: str) -> dict:
    """US duty on an order, using its own quantity, price and ship date."""
    o = ORDERS.get(po_number.upper())
    if not o:
        return _err("NOT_FOUND", f"No order with number {po_number}")
    res = import_duty(o["sku"], int(o["quantity"]), float(o["unit_price_usd"]), tier, o["ship_date"])
    if res["ok"]:
        res["data"].update(po_number=o["po_number"], sku=o["sku"], ship_date=o["ship_date"],
                           quantity=int(o["quantity"]), unit_price_usd=float(o["unit_price_usd"]))
        res["trace"].insert(0, f"Order {o['po_number']}: {o['quantity']} pairs of {o['sku']} at "
                               f"{float(o['unit_price_usd']):.2f} USD, ships {o['ship_date']}")
    return res


RULE_TOPICS_FILE = REPO_ROOT / "config" / "rule_topics.json"


def _load_rule_topics(path: Path = RULE_TOPICS_FILE) -> dict:
    """Rule texts live in config/rule_topics.json (topic -> summary, text, source) so a rule can be added or
    corrected without touching code. JSON, not YAML, so it loads without extra libraries on Cloud Run."""
    import json
    return json.loads(path.read_text(encoding="utf-8"))["topics"]


RULE_TOPICS = _load_rule_topics()


def _section_338_text() -> str:
    """Built from the rules file, so it shows exactly which leather codes the extra tariff hits and which it doesn't."""
    duty = eng.RULES["us_duty"]
    hit = [f"{c} ({d['applies_anyway']:.0%} extra)" for c, d in sorted(duty.items()) if d.get("applies_anyway")]
    clear = [f"{c} ({d['general']:.1%} general, {d['usmca']:.0%} if it qualifies)" for c, d in sorted(duty.items())
             if c.startswith("6403") and not d.get("applies_anyway")]
    text = ("Section 338 adds an extra US tariff even when the boot qualifies under CUSMA, on: " + ", ".join(hit) +
            ". 6403.91.90 is leather boots for women and others (not men, youths or boys).") if hit else \
        "No Section 338 tariffs are in the current rules file."
    if clear:
        text += " Not on the Section 338 list: " + ", ".join(clear) + " (men's, youths' and boys' leather boots)."
    return text


def lookup_rule(topic: str, tier: str) -> dict:
    """Plain-language explanation of a CUSMA rule, with its source. Topics are listed below."""
    fr = eng.RULES["footwear_rule"]
    lo, hi = eng.RULES["classification"]["knee_review_band_in"]
    fill = dict(rvc=fr["rvc_threshold"], dm=fr["de_minimis_share"], margin=fr["escalation_margin"], lo=lo, hi=hi)
    keys = list(RULE_TOPICS) if topic in ("all", "", None) else [t for t in RULE_TOPICS if t == topic]
    if not keys:
        return _err("NOT_FOUND", f"No rule topic '{topic}'. Topics: {', '.join(RULE_TOPICS)}. If none fits, "
                                 "say this rule isn't covered and hand off with create_escalation.")
    out, figures = {}, []

    for k in keys:
        topic_data = RULE_TOPICS[k]

        rule_text = (
            _section_338_text()
            if k == "section_338"
            else topic_data["text"].format(**fill)
        )

        source = topic_data["source"]

        out[k] = (
            f"{rule_text} "
            f"(Source: {source})"
        )

        # Published figures in the substantive rule text are public
        # and may be quoted by every tier. Do not extract numbers
        # from the citation itself, because article, heading and
        # section numbers are references rather than answer figures.
        figures += [
            float(number.replace(",", ""))
            for number in re.findall(
                r"\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?",
                rule_text,
            )
        ]

    out["rule_figures"] = sorted(
        set(figures)
    )
    return _ok(out, [f"Rules file version {eng.RULES['rules_version']}"])


lookup_rule.__doc__ += " " + "; ".join(f"{k} ({v['summary']})" for k, v in RULE_TOPICS.items()) + \
    ". Use 'all' to get every topic. If no topic covers the question, do not answer from memory."


_ticket = itertools.count(1)


def create_escalation(reason: str, summary: str, tier: str, handoff_to: str = "staff_on_duty") -> dict:
    """Hand the request to a person (role names from config/policy.yaml). Returns a reference number."""
    ref = f"ESC-{next(_ticket):04d}"
    return _ok({"reference": ref, "handoff_to": handoff_to, "reason": reason, "summary": summary},
               [f"Handed to {handoff_to}: {reason}"])


def draft_certification(po_number: str, tier: str) -> dict:
    """Draft (never sign) a CUSMA certification of origin for an order. Only for a boot that qualifies
    on the ship date. Tier C requests are taken and handed to a person, with no draft sent."""
    if _rules(tier).get("cert_draft") == "hand_to_person":
        return _err("NOT_ALLOWED", "Certification drafts are prepared for external users by a compliance analyst.")
    o = ORDERS.get(po_number.upper())
    if not o:
        return _err("NOT_FOUND", f"No order with number {po_number}")
    origin = _apply_documents(eng.check_origin(o["sku"], _as_of(o["ship_date"])), o["sku"], _as_of(o["ship_date"]))
    if not origin["ok"]:
        return origin
    v = origin["data"]
    if v["verdict"] != "qualifies":
        return _err("NOT_SUPPORTED", f"No draft: {o['sku']} is '{v['verdict']}' on the ship date {o['ship_date']}. "
                                     f"{v['escalate_reason'] or ''}".strip())
    boot, cust = eng.SKUS[o["sku"]], CUSTOMERS.get(o["customer_id"], {})
    fields = {
        "status": "DRAFT - FOR SIGNATURE BY AN AUTHORIZED PERSON. NOT VALID UNTIL SIGNED.",
        "certifier": "Exporter / producer",
        "exporter": f"{COMPANY['name']}, {COMPANY['address']}",
        "producer": f"{COMPANY['name']}, {COMPANY['address']}",
        "importer": f"{cust.get('company', '')}, [STREET ADDRESS], {cust.get('city', '')}, {cust.get('state', '')}",
        "description": f"{boot['style_name']} ({o['sku']}), {o['quantity']} pairs",
        "hs_code": v["hs_code"],
        "origin_criterion": "B [to confirm: produced in Canada from non-originating materials meeting the "
                            "product-specific rule]",
        "blanket_period": "single shipment",
        "authorized_signature": COMPANY["certifier"],
        "date": "[DATE OF SIGNATURE]",
    }
    return mask(_ok(fields, [f"{o['sku']} qualifies on {o['ship_date']} ({v['hs_code']}); draft only, a person signs"] +
                    origin["trace"][-2:], ["Placeholders in [brackets] must be completed before signing."]), tier)


# The registry the agents use: name -> (function, description, JSON schema for the arguments, excluding tier).
_S = {"type": "string"}
TOOLS = {
    "get_boot": (get_boot, get_boot.__doc__, {"sku": _S}),
    "get_materials": (get_materials, get_materials.__doc__, {"sku": _S, "as_of": _S}),
    "check_origin": (check_origin, check_origin.__doc__, {"sku": _S, "as_of": _S}),
    "find_alternative_suppliers": (find_alternative_suppliers, find_alternative_suppliers.__doc__,
                                   {"sku": _S, "line_no": {"type": "integer"}}),
    "what_if": (what_if, what_if.__doc__, {"sku": _S, "as_of": _S, "changes": {"type": "array", "items": {"type": "object"}}}),
    "import_duty": (import_duty, import_duty.__doc__, {"sku": _S, "quantity": {"type": "integer"},
                                                       "unit_price_usd": {"type": "number"}, "as_of": _S}),
    "get_order": (get_order, get_order.__doc__, {"po_number": _S}),
    "order_duty": (order_duty, order_duty.__doc__, {"po_number": _S}),
    "lookup_rule": (lookup_rule, lookup_rule.__doc__, {"topic": _S}),
    "create_escalation": (create_escalation, create_escalation.__doc__, {"reason": _S, "summary": _S, "handoff_to": _S}),
    "draft_certification": (draft_certification, draft_certification.__doc__, {"po_number": _S}),
}
REQUIRED = {"get_boot": ["sku"], "get_materials": ["sku"], "check_origin": ["sku"],
            "find_alternative_suppliers": ["sku", "line_no"], "what_if": ["sku", "changes"],
            "import_duty": ["sku", "quantity", "unit_price_usd"], "get_order": ["po_number"],
            "order_duty": ["po_number"], "lookup_rule": ["topic"], "create_escalation": ["reason", "summary"],
            "draft_certification": ["po_number"]}


def call_tool(name: str, args: dict, tier: str) -> dict:
    """Run one tool with the asker's tier. The tier is added here; arguments from the model cannot override it."""
    if name not in TOOLS:
        return _err("NOT_FOUND", f"Unknown tool {name}")
    fn = TOOLS[name][0]
    args = {k: v for k, v in (args or {}).items() if k != "tier"}
    try:
        return fn(**args, tier=normalise_tier(tier))
    except TypeError as e:
        return _err("BAD_INPUT", f"{name}: {e}")
    except (KeyError, ValueError) as e:
        return _err("BAD_INPUT", f"{name}: {e}")
