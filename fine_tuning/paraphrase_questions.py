"""Rewrite router questions in other writing styles (data augmentation for router training).

The generator's questions are tidy and alike, so a router trained only on them does badly on how people really
write. This asks the model to rewrite each question in a few different styles. The meaning stays the same, so the
route label is kept. The styles are general descriptions, NOT copied from any test question.

    python fine_tuning/paraphrase_questions.py            # needs OPENROUTER_API_KEY; ~10-15 minutes, well under $1

Reads data/router_split/train.jsonl and val_new_wording.jsonl. Writes to data/router_split/:
    train_styled.jsonl       the training questions PLUS 3 rewrites of each (use for training)
    val_new_style.jsonl      2 rewrites of each new-wording validation question (new wording AND new style)
    paraphrase_report.json   counts, rewrites dropped by the checks, model use and cost
and evaluation/results/paraphrase_spot_check.xlsx: 50 random rewrites to approve by eye.

Each rewrite is checked: not empty, not a copy, not an answer, and every code in the original (NB-0413, PO-8013,
SUP-010 ...) still there. Rewrites that fail are dropped and counted. If the run stops (e.g. Colab disconnects),
run it again: finished questions are kept in evaluation/results/paraphrase_cache.jsonl and skipped.
"""
from __future__ import annotations

import argparse
import contextvars
import json
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SPLIT_DIR = ROOT / "data" / "router_split"
RESULTS_DIR = ROOT / "evaluation" / "results"
CACHE_FILE = RESULTS_DIR / "paraphrase_cache.jsonl"
SPOT_CHECK_FILE = RESULTS_DIR / "paraphrase_spot_check.xlsx"
SEED = 894

STYLES = {
    "casual": "casual and short, like a quick chat message to a colleague",
    "rushed": "rushed, all lower case, with one or two small typos and little punctuation",
    "wordy": "long and wordy, with a sentence of background about why they are asking",
    "email": "a polite, formal email with a greeting and a sign-off",
    "plain_english": "simple English from someone whose first language is not English",
    "keywords": "very short, just the key words, not a full sentence",
    "indirect": "indirect: describes the situation and implies the question instead of asking it outright",
    "synonyms": "different words for the same things (e.g. 'USMCA' or 'the trade agreement' for CUSMA, "
                "'footwear' for boots, 'tariff' for duty)",
    "outside_view": "from the point of view of a customer or partner outside the company",
    "french": "in French, as a French-speaking Canadian would write it",
}
CODE = re.compile(r"\b(?:NB|PO|SUP|ESC|DOC|CUS)[\s-]?\d{3,4}(?:[\s-]?V\d)?\b|\b\d{4}\.\d{2}(?:\.\d{2})?\b", re.IGNORECASE)

PROMPT = """You rewrite questions that staff, customers and suppliers send to a Canadian rubber-boot maker's
CUSMA (USMCA) trade assistant. Rewrite the question below once for each style listed.

Rules:
- Keep exactly the same meaning and the same request. Do not answer it, add facts, or change what is asked.
- Keep every code, number, date, product name and company name exactly as written.
- Each rewrite must sound like a real person writing in that style.

Question: {question}

Styles:
{styles}

Reply with JSON only: {{"rewrites": [{{"style": "<style name>", "text": "<rewritten question>"}}, ...]}}"""


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def styles_for(question_id: str, n: int, seed: int = SEED) -> list[str]:
    """The same styles for the same question every time."""
    return sorted(random.Random(f"{seed}:{question_id}").sample(sorted(STYLES), n))


def _codes(text: str) -> set[str]:
    return {re.sub(r"[\s-]", "", c).upper() for c in CODE.findall(text or "")}


def check_rewrite(original: str, rewrite: str) -> str | None:
    """Why a rewrite is rejected, or None if it is kept."""
    text = (rewrite or "").strip()
    if len(text) < 3:
        return "empty"
    if text.casefold() == original.strip().casefold():
        return "copy"
    if not _codes(original) <= _codes(text):
        return "code_lost"
    if len(text) > 4 * max(len(original), 60):
        return "too_long"
    return None


def parse_rewrites(content: str) -> list[dict]:
    content = (content or "").strip()
    if content.startswith("```"):
        content = content.strip("`").split("\n", 1)[-1]
    start, end = content.find("{"), content.rfind("}")
    try:
        data = json.loads(content[start:end + 1])
    except ValueError:
        return []
    return [r for r in data.get("rewrites", []) if isinstance(r, dict)]


def rewrite_one(row: dict, styles: list[str], model) -> dict:
    listed = "\n".join(f"- {s}: {STYLES[s]}" for s in styles)
    reply = model([{"role": "user", "content": PROMPT.format(question=row["query"], styles=listed)}], [])
    got = {r.get("style"): str(r.get("text") or "") for r in parse_rewrites(reply.get("content"))}
    kept, dropped = [], []
    for s in styles:
        why = check_rewrite(row["query"], got.get(s, ""))
        (dropped if why else kept).append({"style": s, "text": got.get(s, ""), "reason": why})
    return {"id": row["id"], "kept": kept, "dropped": dropped}


def run(rows: list[dict], n_styles: int, model, cache: Path = CACHE_FILE, workers: int = 8) -> dict[str, dict]:
    """Rewrite every row (skipping ones already in the cache). Returns {id: result}."""
    done = {r["key"]: r for r in read_jsonl(cache)}
    todo = [r for r in rows if f"{r['id']}|{n_styles}" not in done]
    lock = threading.Lock()
    cache.parent.mkdir(parents=True, exist_ok=True)

    def work(row):
        try:
            res = rewrite_one(row, styles_for(row["id"], n_styles), model)
        except Exception as err:  # one failed call: keep going; it is retried on the next run
            print(f"  {row['id']}: {type(err).__name__}: {str(err)[:120]}", file=sys.stderr)
            return
        res["key"] = f"{row['id']}|{n_styles}"
        with lock:
            done[res["key"]] = res
            with cache.open("a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
            if len(done) % 100 == 0:
                print(f"  {len(done)} questions rewritten", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row in todo:
            pool.submit(contextvars.copy_context().run, work, row)
    return {r["id"]: done[f"{r['id']}|{n_styles}"] for r in rows if f"{r['id']}|{n_styles}" in done}


def styled_rows(rows: list[dict], results: dict[str, dict]) -> list[dict]:
    out = []
    for row in rows:
        for i, rw in enumerate((results.get(row["id"]) or {}).get("kept", [])):
            out.append({**row, "id": f"{row['id']}-s{i + 1}", "query": rw["text"], "style": rw["style"],
                        "paraphrase_of": row["id"], "original_query": row["query"]})
    return out


def write_spot_check(rows: list[dict], path: Path = SPOT_CHECK_FILE, n: int = 50) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    sample = random.Random(SEED).sample(rows, min(n, len(rows)))
    wb = Workbook()
    ws = wb.active
    ws.title = "Spot check"
    ws.append(["Check each rewrite: does it ask the SAME thing as the original, so the same route is still right? "
               "Pick Yes / No / Unsure in column F."])
    ws["A1"].font = Font(bold=True)
    ws.append(["ID", "Route", "Style", "Original question", "Rewrite", "Same meaning?", "Notes"])
    for c in ws[2]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="1F4E78")
    for r in sample:
        ws.append([r["id"], r["intent"], r["style"], r["original_query"], r["query"], "", ""])
    dv = DataValidation(type="list", formula1='"Yes,No,Unsure"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"F3:F{len(sample) + 2}")
    for col, width in zip("ABCDEFG", (16, 20, 14, 55, 55, 14, 30)):
        ws.column_dimensions[col].width = width
    for row in ws.iter_rows(min_row=3):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A3"
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-styles", type=int, default=3)
    ap.add_argument("--val-styles", type=int, default=2)
    ap.add_argument("--limit", type=int, help="only the first N questions of each file (quick try)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    from agents.base import OpenRouterModel, measure_usage
    model = OpenRouterModel(temperature=0.8)
    train = read_jsonl(SPLIT_DIR / "train.jsonl")[:args.limit]
    val = read_jsonl(SPLIT_DIR / "val_new_wording.jsonl")[:args.limit]

    with measure_usage() as meter:
        print(f"Rewriting {len(train)} training questions ...", file=sys.stderr)
        train_res = run(train, args.train_styles, model, workers=args.workers)
        print(f"Rewriting {len(val)} validation questions ...", file=sys.stderr)
        val_res = run(val, args.val_styles, model, workers=args.workers)

    train_new, val_new = styled_rows(train, train_res), styled_rows(val, val_res)
    write_jsonl(SPLIT_DIR / "train_styled.jsonl", train + train_new)
    write_jsonl(SPLIT_DIR / "val_new_style.jsonl", val_new)
    dropped = {}
    for res in list(train_res.values()) + list(val_res.values()):
        for d in res["dropped"]:
            dropped[d["reason"]] = dropped.get(d["reason"], 0) + 1
    report = {"seed": SEED, "styles": STYLES, "train_questions": len(train), "train_rewrites_kept": len(train_new),
              "val_questions": len(val), "val_rewrites_kept": len(val_new), "rewrites_dropped": dropped,
              "questions_not_finished": (len(train) - len(train_res)) + (len(val) - len(val_res)),
              "model_use": meter.as_dict()}
    (SPLIT_DIR / "paraphrase_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                                      encoding="utf-8")
    write_spot_check(train_new + val_new)
    print(json.dumps({k: v for k, v in report.items() if k != "styles"}, indent=2))
    if report["questions_not_finished"]:
        print("Some questions failed: run the script again to finish them.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
