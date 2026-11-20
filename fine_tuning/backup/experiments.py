"""The router experiments, each changing one thing at a time, and the report that brings them together.

Training versions (all the same model and settings except the one thing named):
  fine-tuned                  every layer trained, on the original questions plus the style rewrites
  frozen encoder              only the final classification layer trained (20 epochs, as only one small layer learns);
                              the pretrained weights stay as they are. It reads only the first token's output, which
                              MiniLM wasn't trained to use, so the comparison also includes a fairer frozen baseline:
  frozen + classifier         (a router, not a trained version) the averaged word vectors from the unchanged
                              pretrained model, with a simple classifier on top (routers.make_probe_router)
  fine-tuned, original data   every layer trained, on the original questions only (what the rewrites add)
  fine-tuned, 25% / 50% data  every layer trained on a share of the questions (how much data it needs)

Routers compared on the same questions:
  current method (nearest neighbour, no training), the trained versions above, and prompted LLMs in a 2 x 2:
  small and large, each with and without example questions (size and examples each changed on their own).

Before the versions are trained, search() tries combinations of learning rate and batch size on validation, and
the best one (with its best number of epochs) becomes the settings for every version.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from fine_tuning.backup import analysis as an
from fine_tuning.backup import train_finetuned as tf

VERSIONS = {
    "fine-tuned": {},
    "frozen encoder": {"freeze_encoder": True, "learning_rate": 1e-3, "epochs": 20},
    "fine-tuned, original data": {"train_file": "train"},
    "fine-tuned, 25% data": {"train_fraction": 0.25},
    "fine-tuned, 50% data": {"train_fraction": 0.5},
}


SEARCH = {"learning_rate": [2e-5, 5e-5, 1e-4], "batch_size": [16, 32]}


def search(base: dict, out_root: str, grid: dict = SEARCH, epochs: int = 6, build=None) -> list[dict]:
    """Try every combination in grid on validation only. Each run keeps its best epoch (up to `epochs`), so the
    number of epochs is chosen too. Returns one row per combination, best first."""
    import itertools
    keys = list(grid)
    results = []
    for values in itertools.product(*(grid[k] for k in keys)):
        combo = dict(zip(keys, values))
        name = ", ".join(f"{k}={v}" for k, v in combo.items())
        print(f"\n=== search: {name} ===")
        settings = {**base, **combo, "epochs": epochs, "model_dir": str(Path(out_root) / ("search_" + slug(name)))}
        tok, model = (build or tf.build)(settings["base_model"])
        info = tf.train(settings, tokenizer=tok, model=model)
        results.append({**combo, "best_epoch": info["best_epoch"]["epoch"],
                        "mean_macro_f1": info["best_epoch"]["mean_macro_f1"],
                        "val_new_wording": info["best_epoch"]["val_new_wording"],
                        "val_new_style": info["best_epoch"]["val_new_style"],
                        "training_seconds": info["training_seconds"],
                        "curve": [h["mean_macro_f1"] for h in info["history"]]})
    return sorted(results, key=lambda r: -r["mean_macro_f1"])


def best_settings(base: dict, results: list[dict], max_epochs: int | None = None) -> dict:
    """The base settings with the winning learning rate and batch size. With max_epochs, every version then trains for
    up to that many epochs and keeps its own best epoch (fair to versions with less data, which need more epochs);
    without it, the winner's best epoch count is used for all."""
    top = results[0]
    return {**base, **{k: top[k] for k in SEARCH if k in top}, "epochs": max_epochs or top["best_epoch"]}


def slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def train_versions(base: dict, out_root: str, versions: dict = VERSIONS, build=None) -> dict:
    """Train each version into out_root/<name>. A version's own settings (e.g. the frozen version's epochs) override
    the base. build(base_model) can be swapped (for tests)."""
    infos = {}
    for name, extra in versions.items():
        print(f"\n=== {name} ===")
        settings = {**base, **extra, "model_dir": str(Path(out_root) / slug(name))}
        tok, model = (build or tf.build)(settings["base_model"])
        infos[name] = tf.train(settings, tokenizer=tok, model=model)
    return infos


def embedding_check(base_model, tokenizer, finetuned_dir: str, rows: list[dict], path, n: int = 600, seed: int = 894):
    """Vectors from the pretrained model and the fine-tuned one for the same validation questions."""
    sample = random.Random(seed).sample(rows, min(n, len(rows)))
    texts, labels = [r["query"] for r in sample], [r["intent"] for r in sample]
    before = an.embed(base_model, texts, tokenizer=tokenizer)
    after = an.embed(finetuned_dir, texts)
    scores = (an.separation(before, labels), an.separation(after, labels))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    an.chart_embeddings(before, after, labels, scores, path)
    return {"questions": len(sample), "silhouette_before": round(scores[0], 4), "silhouette_after": round(scores[1], 4)}


def write_report(out: Path, infos: dict, rows: list[dict], validation_sets, emb: dict | None, economics_md: str,
                 final: bool = False, live_check: dict | None = None, search_results: list[dict] | None = None) -> str:
    """All tables plus charts, saved under out/. Returns the markdown."""
    out.mkdir(parents=True, exist_ok=True)
    by = an.group_by_router(rows, set(validation_sets))
    ms = {n: an.metrics(r) for n, r in by.items()}
    an.chart_learning({k: v for k, v in infos.items() if "%" not in k}, out / "learning_curves.png")
    an.chart_weight_change({k: infos[k] for k in ("fine-tuned", "frozen encoder") if k in infos}, out / "weight_change.png")
    key = [n for n in by if n in ("current method", "frozen + classifier", "fine-tuned", "llm small + examples", "llm large + examples")]
    an.chart_confusions({n: ms[n] for n in key}, out / "confusion_matrices.png")
    an.chart_f1_per_route(ms, out / "f1_per_route.png")
    an.chart_cost_accuracy([{"name": n, "macro_f1": ms[n]["macro_f1"],
                             "cost_usd": sum(r["cost_usd"] for r in by[n]) / len(by[n]),
                             "seconds": sum(r["seconds"] for r in by[n]) / len(by[n])} for n in by],
                           out / "cost_vs_quality.png")
    sizes = [(infos[n]["train_questions"], infos[n]["best_epoch"]["mean_macro_f1"])
             for n in infos if n == "fine-tuned" or "%" in n]
    if len(sizes) > 1:
        an.chart_training_size(sizes, out / "training_size.png")
    what = "TEST SETS (final run, once)" if final else "validation sets"
    ft = infos.get("fine-tuned")
    lines = [f"# Router comparison: {what}", "",
             f"Question sets: {', '.join(validation_sets)}. 95% confidence intervals from 1,000 bootstrap resamples; "
             "p-values from an exact McNemar test on the same questions.", "",
             "## 1. Does training change the model?", "",
             "The current router doesn't train anything: it compares each question with stored examples using the "
             "pretrained model as it was downloaded. The fine-tuned router changes the model's weights. The frozen "
             "version trains only the final layer, so its encoder is identical to the pretrained one (weight change 0). "
             "Because that version reads only the first token's output, which MiniLM wasn't trained to use, the comparison "
             "also includes 'frozen + classifier': the unchanged pretrained model's averaged word vectors with a simple "
             "classifier on top. That is the fair test of what fine-tuning adds.",
             ""]
    if live_check:
        lines += [f"**The live router is the untrained download.** Of the {live_check['exact_compared']} weight tables the live "
                  f"router's model file stores at full precision, **{live_check['exact_identical']} are identical** to a "
                  f"fresh download of the pretrained model (largest difference {live_check['exact_max_difference']}). "
                  "Fine-tuning would have changed them.", ""]
    lines += [an.training_table(infos), "", "![Weight change](weight_change.png)", ""]
    if emb:
        lines += [f"Route separation in the model's sentence space (silhouette, {emb['questions']} validation questions): "
                  f"**{emb['silhouette_before']:.2f} before fine-tuning → {emb['silhouette_after']:.2f} after** "
                  "(about 0 means the routes are mixed together; towards 1 means clean, separate groups).", "",
                  "![Embeddings before and after](embeddings_before_after.png)", ""]
    if search_results:
        an.chart_search(search_results, out / "hyperparameter_search.png")
        lines += ["## 2. Choosing the training settings (validation only)", "",
                  "Every combination of learning rate and batch size was trained, keeping each run's best epoch. The "
                  "best combination (top row) is used for all the fine-tuned versions below.", "",
                  an.search_table(search_results), "", "![Hyperparameter search](hyperparameter_search.png)", ""]
    if ft:
        lines += ["## 3. How it changed during training (fine-tuned, chosen settings)", "", an.epoch_table(ft), "",
                  "![Learning curves](learning_curves.png)", ""]
    lines += ["## 4. Comparison", "", an.comparison_table(by), "", "### Is the fine-tuned router really different?", "",
              an.significance_table(by, "fine-tuned") if "fine-tuned" in by else "", "",
              "### F1 per route", "", an.per_route_table(by), "", "![F1 per route](f1_per_route.png)", "",
              "![Confusion matrices](confusion_matrices.png)", "", "![Cost vs quality](cost_vs_quality.png)", ""]
    if len(sizes) > 1:
        lines += ["## 5. How much training data it needs", "", "![Training size](training_size.png)", ""]
    lines += ["## 6. Economics", "", economics_md, ""]
    text = "\n".join(lines)
    (out / "router_analysis.md").write_text(text + "\n", encoding="utf-8")
    (out / "training_infos.json").write_text(json.dumps(infos, indent=1), encoding="utf-8")
    return text
