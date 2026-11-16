"""Compare routers on the same questions: accuracy, accuracy per route, time and cost per question.

Only the router is run (no agents, no reviewer), so it is fast and costs nothing except for routers that call a
model. A router is any function that takes the question text and returns a route label (or the trained router's
dict with "predicted_class"). Unknown labels count as escalate_human, as in the pipeline.

Built-in routers:
    nearest_neighbour  the current method (MiniLM embeddings, closest example wins, below the threshold ->
                       escalate_human), rebuilt from data/router_split/train.jsonl so validation is fair
    nearest_neighbour_styled  the same, with the rewritten questions (train_styled.jsonl) added as examples
    live               the router the system uses now (agents/router_agent.py). It was built from ALL training
                       templates, so its validation scores are inflated: compare it on the final test sets only
    keyword            the keyword stand-in (agents/specialists.py)
Your own router: --router NAME=module.path:function, e.g. --router finetuned=fine_tuning.my_router:route

Question sets:
    val_new_wording    templates held out of training (data/router_split): use this to tune and compare
    val_new_style      the same questions rewritten in other writing styles (if paraphrase_questions.py was run)
    val_seen_wording   validation questions from the training templates (for contrast)
  with --final only (run ONCE, at the end, after all tuning is finished):
    holdout_seen, holdout_new, team_ai, team_hand, human_written (if data/test_human_written.jsonl exists)

    python evaluation/compare_routers.py --router nearest_neighbour --router finetuned=fine_tuning.my_router:route
    python evaluation/compare_routers.py ... --final --out evaluation/results/router_comparison

A router that calls a model should go through agents.base.OpenRouterModel (or record usage the same way) so its
tokens and cost are counted; cost falls back to config/model_prices.json when the provider sends none.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SPLIT_DIR = ROOT / "data" / "router_split"
HOLDOUT_FILE = ROOT / "data" / "test_holdout_human_reviewed.jsonl"
HUMAN_FILE = ROOT / "data" / "test_human_written.jsonl"
PRICE_FILE = ROOT / "config" / "model_prices.json"
CACHE_DIR = ROOT / "evaluation" / "results" / "router_cache"
LABELS = ("qualification_check", "rules_info", "sourcing_whatif", "duty_cost", "certification_draft",
          "document_intake", "escalate_human", "out_of_scope")
VALIDATION_SETS = ("val_new_wording", "val_new_style", "val_seen_wording")
FINAL_SETS = ("holdout_seen", "holdout_new", "team_ai", "team_hand", "human_written")


# ---------------------------------------------------------------- question sets
def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _q(row: dict, set_name: str, idx: int) -> dict:
    return {"set": set_name, "id": str(row.get("id") or f"{set_name}-{idx}"),
            "question": str(row.get("query") or row.get("input") or ""),
            "expected": row.get("intent") or row.get("expected_route")}


def load_sets(names) -> dict[str, list[dict]]:
    out = {}
    for name in names:
        if name in VALIDATION_SETS:
            rows = _jsonl(SPLIT_DIR / f"{name}.jsonl")
        elif name in ("holdout_seen", "holdout_new"):
            want = "seen" if name == "holdout_seen" else "unseen"
            rows = [r for r in _jsonl(HOLDOUT_FILE) if r.get("phrasing") == want]
        elif name in ("team_ai", "team_hand"):
            from evaluation.adversarial_suite import TEST_CASES
            hand = name == "team_hand"
            rows = [r for r in TEST_CASES if r.get("expected_route") and (r.get("source") == "hand-written") == hand]
        elif name == "human_written":
            rows = _jsonl(HUMAN_FILE)
        else:
            raise SystemExit(f"Unknown question set: {name}")
        qs = [_q(r, name, i) for i, r in enumerate(rows)]
        qs = [q for q in qs if q["question"] and q["expected"]]
        if qs:
            out[name] = qs
        else:
            print(f"(skipped {name}: no questions found)", file=sys.stderr)
    return out


# ---------------------------------------------------------------- routers
def nearest_neighbour_router(train_file: Path = SPLIT_DIR / "train.jsonl", threshold: float | None = None):
    """The current routing method, with examples from train_file only."""
    import numpy as np
    from agents import router_agent as ra

    threshold = ra.SIMILARITY_THRESHOLD if threshold is None else threshold
    rows = _jsonl(train_file)
    texts, labels = [r["query"] for r in rows], [r["intent"] for r in rows]
    key = hashlib.sha1("\n".join(texts).encode()).hexdigest()[:12]
    cache = CACHE_DIR / f"nn_{key}.npy"
    if cache.exists():
        emb = np.load(cache)
    else:
        emb = np.stack([ra._get_embedding(t) for t in texts]).astype("float32")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache, emb)

    def route(question: str) -> str:
        sims = emb @ ra._get_embedding(question)
        best = int(sims.argmax())
        return labels[best] if float(sims[best]) >= threshold else "escalate_human"

    return route


def load_router(spec: str):
    """'nearest_neighbour' | 'live' | 'keyword' | 'NAME=module.path:function' | 'module.path:function'."""
    name, _, target = spec.partition("=") if "=" in spec else (spec, "", spec)
    if target == "nearest_neighbour":
        return name, nearest_neighbour_router()
    if target == "nearest_neighbour_styled":
        return name, nearest_neighbour_router(SPLIT_DIR / "train_styled.jsonl")
    if target == "live":
        from agents import router_agent
        return name, router_agent.route_user_input
    if target == "keyword":
        from agents.specialists import keyword_route
        return name, keyword_route
    module, _, func = target.partition(":")
    if not func:
        raise SystemExit(f"Router '{spec}': use NAME=module.path:function")
    fn = getattr(importlib.import_module(module), func)
    return (name if "=" in spec else func), fn


def _label(raw) -> str:
    label = raw.get("predicted_class") if isinstance(raw, dict) else raw
    label = str(label or "").strip()
    return label if label in LABELS else "escalate_human"


def _prices() -> dict:
    try:
        return json.loads(PRICE_FILE.read_text(encoding="utf-8")).get("models", {})
    except (OSError, ValueError):
        return {}


def run_router(name: str, fn, sets: dict[str, list[dict]]) -> list[dict]:
    from agents.base import measure_usage
    prices, out = _prices(), []
    for set_name, qs in sets.items():
        for q in qs:
            start = time.perf_counter()
            with measure_usage() as meter:
                try:
                    predicted, error = _label(fn(q["question"])), ""
                except Exception as err:  # a failing call counts as wrong, and is listed
                    predicted, error = "error", f"{type(err).__name__}: {err}"[:200]
            seconds = time.perf_counter() - start
            cost = meter.cost_usd
            if cost is None and meter.calls:
                cost = sum(prices.get(m, {}).get("prompt_usd_per_token", 0) * meter.prompt_tokens +
                           prices.get(m, {}).get("completion_usd_per_token", 0) * meter.completion_tokens
                           for m in meter.models)
            out.append({**q, "router": name, "predicted": predicted, "correct": predicted == q["expected"],
                        "seconds": round(seconds, 4), "model_calls": meter.calls,
                        "cost_usd": round(cost or 0.0, 7), "error": error})
    return out


# ---------------------------------------------------------------- report
def _pct(rows) -> str:
    n = len(rows)
    ok = sum(r["correct"] for r in rows)
    return f"{100 * ok / n:.1f}% ({ok}/{n})" if n else "-"


def report(results: list[dict], set_names: list[str], final: bool) -> str:
    routers = list(dict.fromkeys(r["router"] for r in results))
    by = defaultdict(list)
    for r in results:
        by[(r["router"], r["set"])].append(r)
        by[(r["router"], "*")].append(r)
    lines = ["# Router comparison", "",
             "Final test run: test sets included. Do not tune anything after this run." if final else
             "Validation only. Test sets are left out until the final run (--final).", "",
             "## Accuracy", "", "| Router | " + " | ".join(set_names) + " | Seconds per question | Cost per question (USD) |",
             "|---" * (len(set_names) + 3) + "|"]
    for rt in routers:
        allr = by[(rt, "*")]
        sec = sum(r["seconds"] for r in allr) / max(len(allr), 1)
        cost = sum(r["cost_usd"] for r in allr) / max(len(allr), 1)
        lines.append(f"| {rt} | " + " | ".join(_pct(by[(rt, s)]) for s in set_names) + f" | {sec:.4f} | {cost:.6f} |")
    focus = [s for s in set_names if s not in ("val_seen_wording", "holdout_seen")]
    lines += ["", f"## Accuracy per route ({', '.join(focus)} combined)", "",
              "| Route | " + " | ".join(routers) + " |", "|---" * (len(routers) + 1) + "|"]
    for label in LABELS:
        cells = [_pct([r for r in by[(rt, "*")] if r["set"] in focus and r["expected"] == label]) for rt in routers]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "## Most common mistakes", ""]
    for rt in routers:
        wrong = defaultdict(int)
        for r in by[(rt, "*")]:
            if r["set"] in focus and not r["correct"]:
                wrong[(r["expected"], r["predicted"])] += 1
        top = sorted(wrong.items(), key=lambda kv: -kv[1])[:5]
        lines.append(f"**{rt}:** " + ("; ".join(f"{e} -> {p} ({n})" for (e, p), n in top) or "none"))
        lines.append("")
    errors = [r for r in results if r["error"]]
    if errors:
        lines.append(f"Errors: {len(errors)} questions failed (counted as wrong); see the predictions CSV.")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--router", action="append", required=True, help="router to compare (repeat for several)")
    ap.add_argument("--final", action="store_true", help="also run the test sets: once, at the end, after all tuning")
    ap.add_argument("--limit", type=int, help="only the first N questions of each set (quick try)")
    ap.add_argument("--out", help="write <out>.md and <out>_predictions.csv")
    args = ap.parse_args(argv)

    names = list(VALIDATION_SETS) + (list(FINAL_SETS) if args.final else [])
    sets = load_sets(names)
    if args.limit:
        sets = {k: v[:args.limit] for k, v in sets.items()}
    results = []
    for spec in args.router:
        name, fn = load_router(spec)
        print(f"Routing with {name} ...", file=sys.stderr)
        results += run_router(name, fn, sets)
    text = report(results, list(sets), args.final)
    print(text)
    if args.out:
        Path(f"{args.out}.md").write_text(text + "\n", encoding="utf-8")
        with open(f"{args.out}_predictions.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        print(f"\nWrote {args.out}.md and {args.out}_predictions.csv", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
