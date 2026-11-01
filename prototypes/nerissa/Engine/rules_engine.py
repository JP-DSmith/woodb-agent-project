
"""The rules engine (step 3 uses it; later, the agents call it as tools).

Plain Python, no AI. The agents call these functions and never work out
origin or duty themselves. Every function returns the envelope agreed in the
tool contract: {ok, data, trace, warnings, rules_version} or {ok: False, error}.

this is a toolbox, not a program.
it contains functions, but own it's own, it doesn't do anything
other code opens and uses it

step 3 calls check_origin and duty
agents will call functions from here instead of working it out themselves

functions:
get_boot - lloks up boot details
get_materials - boot parts
classify - computes US tariff code
check_origin -
what_if - runs check_origin before and after
dupty - US duty with and without CUSMA

"""
import csv
import json
import pathlib
from datetime import date

# ---- FOLDERS ----
DRIVE         = pathlib.Path('/content/drive/MyDrive/Deep_Learning')
ENGINE_FOLDER = DRIVE / 'Engine'     # where rules_engine.py lives
WORLD_FOLDER  = DRIVE / 'World'      # the answer key is saved here, next to the world files
RULES_FOLDER = DRIVE / 'Rules'       # where footwear_rules

RULES = json.loads((RULES_FOLDER / "footwear_rules.json").read_text())   # written by step 1
FR = RULES["footwear_rule"]
CUSMA = {"CA", "US", "MX"}


def _load(name):
    with open(WORLD_FOLDER / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


SKUS = {r["sku"]: r for r in _load("skus.csv")}
BOM = _load("bom_lines.csv")
SUPPLIERS = {r["supplier_id"]: r for r in _load("suppliers.csv")}


def _ok(data, trace, warnings=None):
    return {"ok": True, "data": data, "trace": trace, "warnings": warnings or [],
            "rules_version": RULES["rules_version"]}


def _err(code, message):
    return {"ok": False, "error": {"code": code, "message": message}}


# ---------------------------------------------------------------- lookups
def get_boot(sku):
    if sku not in SKUS:
        return _err("NOT_FOUND", f"No boot with id {sku}")
    return _ok(SKUS[sku], [f"Found {sku}"])


def _is_originating(country, supplier, as_of):
    """Assumption #2: a CUSMA-country input counts only with a valid, unexpired declaration."""
    if country not in CUSMA:
        return False, "made outside CUSMA"
    if supplier["declaration_status"] != "valid":
        return False, f"no valid declaration ({supplier['declaration_status']})"
    if supplier["declaration_expiry"] and date.fromisoformat(supplier["declaration_expiry"]) < as_of:
        return False, f"declaration expired {supplier['declaration_expiry']}"
    return True, "CUSMA origin with valid declaration"


def _is_excluded(hs):
    heading = int(hs[:4]) if hs[:4].isdigit() else 0
    lo, hi = FR["excluded_headings_range"]
    return lo <= heading <= hi or any(hs.startswith(x) for x in FR["excluded_subheadings"])


def get_materials(sku, as_of=None, changes=None):
    if sku not in SKUS:
        return _err("NOT_FOUND", f"No boot with id {sku}")
    as_of = as_of or date.today()
    lines = []
    for r in (x for x in BOM if x["sku"] == sku):
        line = dict(r, cost_cad=float(r["cost_cad"]))
        for ch in changes or []:                       # used by what_if
            if int(ch["line"]) == int(r["line_no"]):
                if "new_supplier_id" in ch:
                    s = SUPPLIERS[ch["new_supplier_id"]]
                    line.update(supplier_id=s["supplier_id"], origin_country=s["country"])
                if "new_cost_cad" in ch:
                    line["cost_cad"] = float(ch["new_cost_cad"])
                if "origin_country" in ch:
                    line["origin_country"] = ch["origin_country"]
                if "new_hs_code" in ch:
                    line["hs_code"] = ch["new_hs_code"]
                if "new_description" in ch:
                    line["description"] = ch["new_description"]
        orig, why = _is_originating(line["origin_country"], SUPPLIERS[line["supplier_id"]], as_of)
        line.update(originating=orig, origin_reason=why, excluded=_is_excluded(line["hs_code"]))
        lines.append(line)
    return _ok(lines, [f"{len(lines)} material lines for {sku}"])


# ---------------------------------------------------------------- classification
def classify(sku):
    b = SKUS.get(sku)
    if not b:
        return _err("NOT_FOUND", f"No boot with id {sku}")
    lo, hi = RULES["classification"]["knee_review_band_in"]
    review = lambda why: _ok({"hs_code": None, "status": "needs_review", "reason": why}, [why], [why])
    upper = b["upper_class"]
    if upper == "neoprene":                                    # Archetype S6
        return review("Neoprene upper: rubber (6401) or textile (6404) is a judgment call; a person decides")
    if upper in ("textile", "leather") or (upper == "rubber_plastic" and b["construction"] == "stitched"):
        # Stitched upper on a rubber bottom: the material with the largest external surface decides
        if b["toe_type"] == "steel":
            return review("Stitched boot with a metal toe: code not in our rules sheet yet")
        if upper == "textile":
            code = "6404.19.20"
        elif upper == "leather":
            code = "6403.91.60" if b["gender"] == "men" else "6403.91.90"
        else:
            code = "6402.91.50"
        return _ok({"hs_code": code, "status": "classified"},
                   [f"Stitched upper, main material {upper} ({float(b['upper_main_share']):.0%} of surface), "
                    f"gender {b['gender']} -> {code}"])
    # Moulded waterproof rubber/plastic boot: heading 6401
    if b["toe_type"] == "steel":
        return _ok({"hs_code": "6401.10.00", "status": "classified"}, ["Moulded boot with metal toe cap -> 6401.10"])
    if lo <= float(b["height_in"]) <= hi:
        return review(f"Height {b['height_in']} in. is inside the {lo}-{hi} in. knee band (Assumption #8)")
    code = "6401.99.10" if b["covers_knee"] == "yes" else "6401.92.90"
    return _ok({"hs_code": code, "status": "classified"},
               [f"Moulded waterproof boot, non-metal toe, covers knee = {b['covers_knee']} -> {code}"])


# ---------------------------------------------------------------- origin
def check_origin(sku, as_of=None, changes=None):
    m = get_materials(sku, as_of, changes)
    if not m["ok"]:
        return m
    b, lines = SKUS[sku], m["data"]
    vnm = sum(l["cost_cad"] for l in lines if not l["originating"])
    vom = sum(l["cost_cad"] for l in lines if l["originating"])
    nc = vnm + vom + float(b["labour_cad"]) + float(b["overhead_cad"])
    failing = [l for l in lines if l["excluded"] and not l["originating"]]
    excl_share = sum(l["cost_cad"] for l in failing) / nc
    shift_ok = excl_share <= FR["de_minimis_share"]
    rvc = (nc - vnm) / nc
    rvc_waived = vnm / nc <= FR["de_minimis_share"]
    rvc_ok = rvc_waived or rvc >= FR["rvc_threshold"]

    trace = [f"Net cost = materials {vnm + vom:.2f} + labour {b['labour_cad']} + overhead {b['overhead_cad']} = {nc:.2f} CAD"]
    if failing:
        names = ", ".join(f"{l['component']} {l['hs_code']} ({l['origin_country']})" for l in failing)
        trace.append(f"Excluded imported inputs: {names} = {excl_share:.1%} of net cost; "
                     f"limit {FR['de_minimis_share']:.0%} -> tariff shift {'met by de minimis' if shift_ok else 'FAILS'}")
    else:
        trace.append("No excluded imported inputs -> tariff shift met")
    trace.append(f"RVC = ({nc:.2f} - {vnm:.2f}) / {nc:.2f} = {rvc:.1%} vs {FR['rvc_threshold']:.0%}"
                 + (" (waived: all imported inputs within 10%)" if rvc_waived else ""))

    escalate = None
    if not shift_ok:
        verdict = "does_not_qualify"
    elif not rvc_ok:
        verdict = "does_not_qualify"
    elif not rvc_waived and rvc - FR["rvc_threshold"] <= FR["escalation_margin"]:
        verdict, escalate = "needs_review", f"RVC {rvc:.1%} is within {FR['escalation_margin']:.0%} of the threshold"
    else:
        verdict = "qualifies"
    cls = classify(sku)["data"]
    if verdict == "qualifies" and cls["status"] == "needs_review":
        verdict, escalate = "needs_review", "origin test passed, but the tariff code needs a person: " + cls["reason"]
    elif verdict == "needs_review" and cls["status"] == "needs_review":
        escalate += "; the tariff code also needs a person: " + cls["reason"]
    trace.append(f"Verdict: {verdict}" + (f" ({escalate})" if escalate else ""))
    data = {"sku": sku, "hs_code": cls["hs_code"], "verdict": verdict,
            "tariff_shift": {"passed": shift_ok, "excluded_share": round(excl_share, 4),
                             "failing_materials": [l["component"] for l in failing]},
            "rvc": {"value": round(rvc, 4), "threshold": FR["rvc_threshold"], "method": "net_cost",
                    "waived": rvc_waived},
            "net_cost_cad": round(nc, 2), "vnm_cad": round(vnm, 2), "escalate_reason": escalate}
    return _ok(data, trace)


def what_if(sku, changes, as_of=None):
    before, after = check_origin(sku, as_of), check_origin(sku, as_of, changes)
    if not (before["ok"] and after["ok"]):
        return before if not before["ok"] else after
    b, a = before["data"], after["data"]
    return _ok({"before": {"verdict": b["verdict"], "rvc": b["rvc"]["value"]},
                "after": {"verdict": a["verdict"], "rvc": a["rvc"]["value"]},
                "rvc_change": round(a["rvc"]["value"] - b["rvc"]["value"], 4), "changes": changes},
               ["BEFORE: " + before["trace"][-2], "BEFORE: " + before["trace"][-1]]
               + ["AFTER:  " + t for t in after["trace"][1:]])


# ---------------------------------------------------------------- duty
def duty(sku, quantity, unit_price_usd, as_of=None):
    c = classify(sku)
    if not c["ok"]:
        return c
    code = c["data"]["hs_code"]
    if code is None:
        return _err("MISSING_DATA", "Boot not classified yet; a person must decide its tariff code")
    if code not in RULES["us_duty"]:
        return _err("NOT_SUPPORTED", f"No duty rate for {code} in the rules file")
    r = RULES["us_duty"][code]
    o = check_origin(sku, as_of)["data"]
    value = quantity * unit_price_usd
    if_q = value * (r["usmca"] + r["applies_anyway"])
    if_not = value * (r["general"] + r["extra_if_not_qualifying"] + r["applies_anyway"])
    data = {"hs_code": code, "order_value_usd": round(value, 2), "verdict": o["verdict"],
            "duty_if_qualifying_usd": round(if_q, 2), "duty_if_not_qualifying_usd": round(if_not, 2),
            "duty_payable_usd": round(if_q if o["verdict"] == "qualifies" else if_not, 2),
            "rates_as_of": RULES["rules_version"]}
    warn = ["Verdict needs review; duty shown both ways until a person confirms"] if o["verdict"] == "needs_review" else []
    return _ok(data, [f"{code}: general {r['general']:.1%} + extra {r['extra_if_not_qualifying']:.0%} if not qualifying; "
                      f"{r['applies_anyway']:.0%} applies anyway", f"Order value {value:,.2f} USD; origin verdict {o['verdict']}"], warn)
