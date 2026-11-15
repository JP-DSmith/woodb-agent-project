"""Compare two evaluation runs side by side: what improved, what got worse, and which questions flipped.

    python evaluation/compare_runs.py BEFORE.csv AFTER.csv            # print the comparison
    python evaluation/compare_runs.py BEFORE.csv AFTER.csv --out cmp  # also write cmp.md and cmp_flips.csv

BEFORE/AFTER are results CSVs written by evaluation/run_evaluation.py (e.g. the T38 baseline and a re-run).
Questions are matched by case_id, so only questions present in both runs are compared.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import re
import statistics
import sys
from pathlib import Path

TRUE = "True"
DOLLAR = ("duty_payable_usd", "duty_if_not_qualifying_usd")
VERDICT = ("sku_verdict", "order_verdict_on_ship_date")


def load(path: str) -> dict[str, dict]:
    with open(path, encoding="utf-8", newline="") as f:
        # A missing column (e.g. an older results file) counts as blank.
        return {r["case_id"]: defaultdict(str, {k: v or "" for k, v in r.items()}) for r in csv.DictReader(f)}


def num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def has_code(r) -> bool:
    return bool(re.search(r"NB-\d{4}", r.get("query") or ""))


# (section, metric name, which rows count, which column must be "True")
METRICS = [
    ("Routing", "All questions with an expected route", lambda r: r["route_correct"] in ("True", "False"), "route_correct"),
    ("Routing", "Holdout, seen templates", lambda r: r["case_kind"] == "holdout" and r["phrasing"] == "seen", "route_correct"),
    ("Routing", "Holdout, new templates", lambda r: r["case_kind"] == "holdout" and r["phrasing"] == "unseen", "route_correct"),
    ("Routing", "Team normal questions (T23)", lambda r: r["case_kind"] == "test_case" and r["route_correct"] in ("True", "False"), "route_correct"),
    ("Routing", "Hand-written questions", lambda r: r["case_kind"] == "human_written" and r["route_correct"] in ("True", "False"), "route_correct"),
    ("Answers", "All answer-scored questions", lambda r: r["answer_correct"] in ("True", "False"), "answer_correct"),
    ("Answers", "Qualification, boot code given", lambda r: r["answer_type"] == "sku_verdict" and r["expected_route"] != "certification_draft" and has_code(r), "answer_correct"),
    ("Answers", "Qualification, boot name only", lambda r: r["answer_type"] == "sku_verdict" and r["expected_route"] != "certification_draft" and not has_code(r), "answer_correct"),
    ("Answers", "Verdict on an order's ship date", lambda r: r["answer_type"] == "order_verdict_on_ship_date", "answer_correct"),
    ("Answers", "Certification requests", lambda r: r["expected_route"] == "certification_draft" and r["answer_correct"] in ("True", "False"), "answer_correct"),
    ("Answers", "Duty figures, tier A", lambda r: r["answer_type"] in DOLLAR and r["tier"] == "A", "answer_correct"),
    ("Answers", "Duty figures, tiers B/C", lambda r: r["answer_type"] in DOLLAR and r["tier"] != "A", "answer_correct"),
    ("Security", "Attacks, policy-safe", lambda r: r["case_kind"] == "attack", "attack_passed_policy"),
    ("Security", "Attacks, strict (exact outcome)", lambda r: r["case_kind"] == "attack", "attack_passed"),
    ("Security", "Normal questions not wrongly blocked", lambda r: r["case_kind"] == "test_case", "_not_blocked"),
]


def rate(rows, keep, col):
    rows = [r for r in rows if keep(r)]
    if col == "_not_blocked":
        ok = sum(r["pipeline_decision"] != "blocked" for r in rows)
    else:
        ok = sum(r[col] == TRUE for r in rows)
    return ok, len(rows)


def costs(rows):
    used = [r for r in rows if num(r["calls"]) > 0]
    return {
        "Average seconds per question": statistics.mean(num(r["seconds"]) for r in rows) if rows else 0,
        "Median seconds (questions using the model)": statistics.median(num(r["seconds"]) for r in used) if used else 0,
        "Cost per question (USD)": sum(num(r["cost_usd"]) for r in rows) / max(len(rows), 1),
        "Model calls per question": sum(num(r["calls"]) for r in rows) / max(len(rows), 1),
        "Input tokens per question": sum(num(r["prompt_tokens"]) for r in rows) / max(len(rows), 1),
        "Output tokens per question": sum(num(r["completion_tokens"]) for r in rows) / max(len(rows), 1),
        "Errors": sum(1 for r in rows if (r.get("error") or "").strip()),
    }


def flips(before, after):
    """Questions whose result changed: (case_id, what, before -> after, question)."""
    out = []
    for cid in sorted(set(before) & set(after)):
        b, a = before[cid], after[cid]
        for col, what in (("route_correct", "route"), ("answer_correct", "answer"), ("attack_passed_policy", "attack (policy)")):
            if b.get(col) in ("True", "False") and a.get(col) in ("True", "False") and b[col] != a[col]:
                out.append({"case_id": cid, "what": what, "change": "fixed" if a[col] == TRUE else "broke",
                            "before": b.get("actual_route") if col == "route_correct" else b.get("actual_verdict") or b.get("actual_dollar_value") or b.get("attack_status"),
                            "after": a.get("actual_route") if col == "route_correct" else a.get("actual_verdict") or a.get("actual_dollar_value") or a.get("attack_status"),
                            "tools_after": a.get("tools_used"), "question": (a.get("query") or "")[:140]})
    return out


def compare(before_path, after_path):
    before, after = load(before_path), load(after_path)
    common = sorted(set(before) & set(after))
    b_rows, a_rows = [before[c] for c in common], [after[c] for c in common]
    lines = [f"# Run comparison", "", f"Before: {before_path} ({len(before)} questions)  ",
             f"After: {after_path} ({len(after)} questions)  ", f"Compared: {len(common)} questions in both runs", ""]
    section = None
    for sec, name, keep, col in METRICS:
        if sec != section:
            lines += ["", f"## {sec}", "", "| Metric | Before | After | Change |", "|---|---|---|---|"]
            section = sec
        bo, bn = rate(b_rows, keep, col)
        ao, an = rate(a_rows, keep, col)
        if not bn and not an:
            continue
        bp = 100 * bo / bn if bn else 0
        ap = 100 * ao / an if an else 0
        arrow = "▲" if ap > bp + 0.05 else "▼" if ap < bp - 0.05 else "="
        lines.append(f"| {name} | {bp:.1f}% ({bo}/{bn}) | {ap:.1f}% ({ao}/{an}) | {arrow} {ap - bp:+.1f} pts |")
    lines += ["", "## Time and cost", "", "| Metric | Before | After | Change |", "|---|---|---|---|"]
    cb, ca = costs(b_rows), costs(a_rows)
    for k in cb:
        fmt = "{:.6f}" if "USD" in k else "{:.0f}" if "tokens" in k or k == "Errors" else "{:.2f}"
        lines.append(f"| {k} | {fmt.format(cb[k])} | {fmt.format(ca[k])} | {ca[k] - cb[k]:+.4g} |")
    fl = flips(before, after)
    fixed = [f for f in fl if f["change"] == "fixed"]
    broke = [f for f in fl if f["change"] == "broke"]
    lines += ["", f"## Questions that changed: {len(fixed)} fixed, {len(broke)} broke", ""]
    for title, group in (("Broke (check these first)", broke), ("Fixed", fixed)):
        if group:
            lines += [f"### {title}", "", "| Case | What | Before → After | Question |", "|---|---|---|---|"]
            lines += [f"| {f['case_id']} | {f['what']} | {f['before']} → {f['after']} | {f['question']} |" for f in group[:60]]
            if len(group) > 60:
                lines.append(f"| … | | | {len(group) - 60} more in the flips CSV |")
            lines.append("")
    lines += ["Note: model answers vary a little between runs, so a few flips in both directions are normal.",
              "Judge a fix by the group it targets (e.g. 'boot name only'), and check every 'broke' row."]
    return "\n".join(lines), fl


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before"); ap.add_argument("after")
    ap.add_argument("--out", help="write <out>.md and <out>_flips.csv")
    args = ap.parse_args(argv)
    report, fl = compare(args.before, args.after)
    print(report)
    if args.out:
        Path(f"{args.out}.md").write_text(report, encoding="utf-8")
        with open(f"{args.out}_flips.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["case_id", "what", "change", "before", "after", "tools_after", "question"])
            w.writeheader(); w.writerows(fl)
        print(f"\nWrote {args.out}.md and {args.out}_flips.csv", file=sys.stderr)


if __name__ == "__main__":
    main()
