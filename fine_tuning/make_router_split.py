"""Make a router validation set that tests NEW wording (for router tuning and the router comparison).

The question generator writes each question from a template. data/val_dataset.jsonl uses the same templates as
data/train_dataset.jsonl, so a router tuned on it only learns to recognise wording it has already seen, and every
router scores in the high 90s. Here, 3 templates per route are held out: ALL their questions (from both files)
become validation, and none of them is used for training.

    python fine_tuning/make_router_split.py

Writes data/router_split/:
    train.jsonl             training questions from the other templates (use these to train / as examples)
    val_seen_wording.jsonl  validation questions from the training templates (same wording as training)
    val_new_wording.jsonl   every question from the held-out templates (wording the router has never seen)
    held_out_templates.json which templates were held out, and the counts

Only training and validation files are read. The test files (holdout, attacks, team and hand-written
questions) are never touched: they are for the final run only. The split is fixed (seed 894), so anyone who
runs this gets the same files.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = ROOT / "data" / "train_dataset.jsonl"
VAL_FILE = ROOT / "data" / "val_dataset.jsonl"
OUT_DIR = ROOT / "data" / "router_split"
HELD_OUT_PER_ROUTE = 3
SEED = 894


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def choose_held_out(rows: list[dict], per_route: int = HELD_OUT_PER_ROUTE, seed: int = SEED) -> dict[str, list[str]]:
    """Pick `per_route` templates for each route, the same way every time."""
    templates = defaultdict(set)
    for r in rows:
        templates[r["intent"]].add(r["frame_id"])
    rng = random.Random(seed)
    return {route: sorted(rng.sample(sorted(frames), per_route)) for route, frames in sorted(templates.items())}


def make_split(train: list[dict], val: list[dict], per_route: int = HELD_OUT_PER_ROUTE, seed: int = SEED) -> dict:
    held = choose_held_out(train, per_route, seed)
    held_ids = {f for frames in held.values() for f in frames}
    new_train = [r for r in train if r["frame_id"] not in held_ids]
    val_seen = [r for r in val if r["frame_id"] not in held_ids]
    val_new = [{**r, "from_file": "train"} for r in train if r["frame_id"] in held_ids] + \
              [{**r, "from_file": "val"} for r in val if r["frame_id"] in held_ids]
    return {"train": new_train, "val_seen_wording": val_seen, "val_new_wording": val_new, "held_out": held}


def main() -> None:
    split = make_split(read_jsonl(TRAIN_FILE), read_jsonl(VAL_FILE))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val_seen_wording", "val_new_wording"):
        write_jsonl(OUT_DIR / f"{name}.jsonl", split[name])
    counts = {name: {"questions": len(split[name]), "by_route": dict(sorted(Counter(r["intent"] for r in split[name]).items()))}
              for name in ("train", "val_seen_wording", "val_new_wording")}
    (OUT_DIR / "held_out_templates.json").write_text(json.dumps(
        {"seed": SEED, "held_out_per_route": HELD_OUT_PER_ROUTE, "held_out_templates": split["held_out"],
         "counts": counts}, indent=2) + "\n", encoding="utf-8")
    for name, c in counts.items():
        print(f"{name}: {c['questions']} questions")
    print(f"Wrote {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
