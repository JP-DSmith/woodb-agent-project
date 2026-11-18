"""Backup router: fine-tune a small pretrained model to classify questions into the 8 routes.

    python fine_tuning/backup/train_finetuned.py

Trains on data/router_split/<train_file>.jsonl and, after every epoch, measures on the validation sets
(val_new_wording and val_new_style: templates and writing styles it never trains on):
  - accuracy, macro-F1 and per-route F1;
  - how far each part of the network has moved from the pretrained weights (relative change), which shows the
    weights really are being trained (and, with freeze_encoder, that the encoder is not).
The epoch with the best average validation macro-F1 is kept. Test sets are never read.

Settings come from the ROUTER_SETTINGS environment variable (JSON) or the settings passed to train():
  freeze_encoder  True trains only the final classification layer on top of the unchanged pretrained model
                  (the "no fine-tuning" baseline); False fine-tunes every layer.
  train_fraction  use only this share of the training questions (for the training-size comparison).
The model is saved OUTSIDE the repo (model files are too big for GitHub), with training_info.json.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / "data" / "router_split"
LABELS = ["qualification_check", "rules_info", "sourcing_whatif", "duty_cost", "certification_draft",
          "document_intake", "escalate_human", "out_of_scope"]
VALIDATION = ("val_new_wording", "val_new_style")
DEFAULTS = {
    "base_model": "sentence-transformers/all-MiniLM-L6-v2",   # the model the current router uses, untrained
    "train_file": "train_styled",
    "train_fraction": 1.0,
    "freeze_encoder": False,
    "epochs": 4,
    "learning_rate": 5e-5,
    "batch_size": 32,
    "max_length": 64,
    "weight_decay": 0.01,
    "warmup_share": 0.1,
    "seed": 894,
    "model_dir": str(Path.home() / "router_models" / "fine_tuned"),   # outside the repo (model files are large)
}
SETTINGS = {**DEFAULTS, **json.loads(os.environ.get("ROUTER_SETTINGS") or "{}")}


def load(name: str) -> list[dict]:
    with (SPLIT / f"{name}.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def predict(model, tokenizer, texts, device, batch_size=64, max_length=64) -> list[str]:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(texts[i:i + batch_size], truncation=True, max_length=max_length,
                            padding=True, return_tensors="pt").to(device)
            out += [LABELS[p] for p in model(**enc).logits.argmax(-1).tolist()]
    return out


def scores(gold: list[str], pred: list[str]) -> dict:
    """Accuracy, macro-F1 and F1 per route."""
    per = {}
    for label in LABELS:
        tp = sum(g == p == label for g, p in zip(gold, pred))
        fp = sum(p == label != g for g, p in zip(gold, pred))
        fn = sum(g == label != p for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per[label] = round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0
    present = [l for l in LABELS if l in gold]
    return {"accuracy": round(sum(g == p for g, p in zip(gold, pred)) / max(len(gold), 1), 4),
            "macro_f1": round(sum(per[l] for l in present) / max(len(present), 1), 4), "f1_per_route": per}


def layer_group(name: str) -> str:
    """Which part of the network a parameter belongs to (for the weight-change chart)."""
    m = re.search(r"layer\.(\d+)\.", name)
    if m:
        return f"encoder layer {int(m.group(1)) + 1}"
    if "embeddings" in name:
        return "embeddings"
    if "classifier" in name or "pre_classifier" in name:
        return "classification head"
    return "pooler / other"


def weight_change(initial: dict, model) -> dict:
    """Relative change from the starting weights, per group: ||W - W0|| / ||W0||."""
    num, den = {}, {}
    for name, p in model.named_parameters():
        g = layer_group(name)
        w0 = initial[name]
        num[g] = num.get(g, 0.0) + float(((p.detach().cpu() - w0) ** 2).sum())
        den[g] = den.get(g, 0.0) + float((w0 ** 2).sum())
    return {g: round((num[g] / den[g]) ** 0.5 if den[g] else 0.0, 6) for g in num}


def build(base_model: str):
    """Tokenizer and an 8-label classifier on top of the pretrained model."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
        label2id={l: i for i, l in enumerate(LABELS)}, ignore_mismatched_sizes=True)
    return tokenizer, model


def train(settings: dict = SETTINGS, tokenizer=None, model=None) -> dict:
    from transformers import get_linear_schedule_with_warmup
    settings = {**DEFAULTS, **settings}
    random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if tokenizer is None or model is None:
        tokenizer, model = build(settings["base_model"])
    if settings["freeze_encoder"]:
        for name, p in model.named_parameters():
            p.requires_grad = layer_group(name) == "classification head"
    model.to(device)
    initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    train_rows = load(settings["train_file"])
    if settings["train_fraction"] < 1:
        rng = random.Random(settings["seed"])
        train_rows = rng.sample(train_rows, max(1, int(len(train_rows) * settings["train_fraction"])))
    val = {name: load(name) for name in VALIDATION}
    label_id = {l: i for i, l in enumerate(LABELS)}

    def collate(batch):
        enc = tokenizer([r["query"] for r in batch], truncation=True, max_length=settings["max_length"],
                        padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([label_id[r["intent"]] for r in batch])
        return enc

    loader = DataLoader(train_rows, batch_size=settings["batch_size"], shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(settings["seed"]))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    steps = len(loader) * settings["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(steps * settings["warmup_share"]), steps)

    def evaluate(epoch: int, train_loss) -> dict:
        row = {"epoch": epoch, "train_loss": train_loss}
        for name, rows in val.items():
            s = scores([r["intent"] for r in rows],
                       predict(model, tokenizer, [r["query"] for r in rows], device, max_length=settings["max_length"]))
            row[name] = s
        row["mean_macro_f1"] = round(sum(row[n]["macro_f1"] for n in val) / len(val), 4)
        row["weight_change"] = weight_change(initial, model)
        return row

    history = [evaluate(0, None)]            # epoch 0: before any training
    step_losses, best, started = [], None, time.time()
    out_dir = Path(settings["model_dir"])
    print(f"epoch 0 (untrained head): macro-F1 {history[0]['mean_macro_f1']:.1%}   "
          f"trainable {trainable:,} of {total_params:,} weights")
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        total = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            total += loss.item()
            step_losses.append(round(loss.item(), 4))
        row = evaluate(epoch, round(total / len(loader), 4))
        history.append(row)
        print(f"epoch {epoch}: loss {row['train_loss']:.3f}  " +
              "  ".join(f"{n} acc {row[n]['accuracy']:.1%} F1 {row[n]['macro_f1']:.1%}" for n in val))
        if best is None or row["mean_macro_f1"] > best["mean_macro_f1"]:
            best = {"epoch": epoch, "mean_macro_f1": row["mean_macro_f1"],
                    **{n: row[n]["macro_f1"] for n in val}}
            out_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
    info = {"settings": settings, "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "training_seconds": round(time.time() - started, 1), "train_questions": len(train_rows),
            "trainable_weights": trainable, "total_weights": total_params,
            "history": history, "step_losses": step_losses, "best_epoch": best}
    (out_dir / "training_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Best epoch {best['epoch']} saved to {out_dir} ({info['training_seconds']} s)")
    return info


if __name__ == "__main__":
    train()
