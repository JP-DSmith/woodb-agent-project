"""Analysis for the router comparison: metrics, confidence, significance, embeddings and charts.

Works on the rows evaluation/compare_routers.run_router() returns (one per question per router) and on the
training_info.json files train_finetuned.py writes.

  metrics(rows)                     accuracy, macro-F1, F1 per route, confusion matrix
  bootstrap(rows)                   95% confidence intervals for accuracy and macro-F1 (resampling questions)
  mcnemar(rows_a, rows_b)           paired test: do two routers really differ on the same questions?
  embed(model_or_path, texts)       sentence vectors from a model (mean of the last layer)
  separation(vectors, labels)       how well the routes form separate groups (silhouette score, -1 to 1)
  charts                            learning curves, weight change, confusion matrices, embeddings, F1 per route,
                                    accuracy vs cost, training size
"""
from __future__ import annotations

import math
import random
from collections import defaultdict

LABELS = ["qualification_check", "rules_info", "sourcing_whatif", "duty_cost", "certification_draft",
          "document_intake", "escalate_human", "out_of_scope"]
SHORT = {"qualification_check": "qualif.", "rules_info": "rules", "sourcing_whatif": "sourcing",
         "duty_cost": "duty", "certification_draft": "certif.", "document_intake": "intake",
         "escalate_human": "escalate", "out_of_scope": "out of scope"}


# ---------------------------------------------------------------- metrics
def metrics(rows: list[dict]) -> dict:
    gold, pred = [r["expected"] for r in rows], [r["predicted"] for r in rows]
    per, cm = {}, [[0] * len(LABELS) for _ in LABELS]
    idx = {l: i for i, l in enumerate(LABELS)}
    for g, p in zip(gold, pred):
        if g in idx and p in idx:
            cm[idx[g]][idx[p]] += 1
    for l in LABELS:
        tp = sum(g == p == l for g, p in zip(gold, pred))
        fp = sum(p == l != g for g, p in zip(gold, pred))
        fn = sum(g == l != p for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per[l] = {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                  "support": tp + fn}
    present = [l for l in LABELS if per[l]["support"]]
    return {"n": len(rows), "accuracy": sum(g == p for g, p in zip(gold, pred)) / max(len(rows), 1),
            "macro_f1": sum(per[l]["f1"] for l in present) / max(len(present), 1), "per_route": per, "confusion": cm}


def bootstrap(rows: list[dict], n: int = 1000, seed: int = 894) -> dict:
    """95% intervals by resampling the questions with replacement."""
    rng = random.Random(seed)
    acc, f1 = [], []
    for _ in range(n):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        m = metrics(sample)
        acc.append(m["accuracy"])
        f1.append(m["macro_f1"])
    q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))]
    return {"accuracy": (q(acc, 0.025), q(acc, 0.975)), "macro_f1": (q(f1, 0.025), q(f1, 0.975))}


def mcnemar(rows_a: list[dict], rows_b: list[dict]) -> dict:
    """Exact McNemar test on the questions both routers answered. b: only A right; c: only B right."""
    a = {(r["set"], r["id"]): r["correct"] for r in rows_a}
    b = {(r["set"], r["id"]): r["correct"] for r in rows_b}
    shared = a.keys() & b.keys()
    only_a = sum(a[k] and not b[k] for k in shared)
    only_b = sum(b[k] and not a[k] for k in shared)
    n = only_a + only_b
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2 ** n)
    return {"questions": len(shared), "only_a_right": only_a, "only_b_right": only_b, "p_value": p}


# ---------------------------------------------------------------- is the live router trained?
def live_router_matches_pretrained(onnx_path: str, base_model: str) -> dict:
    """Compare the weights inside the live router's ONNX file with the pretrained model downloaded fresh.
    Weights stored at full precision are compared exactly; weights the ONNX export compressed to 8 bits are
    decompressed and compared (a small rounding difference is expected). If the live router had been trained,
    its weights would differ from the download."""
    import numpy as np
    import onnx
    from onnx import numpy_helper
    from transformers import AutoModel
    pre = {k: v.detach().cpu().numpy() for k, v in AutoModel.from_pretrained(base_model).state_dict().items()}
    inits = {i.name: numpy_helper.to_array(i) for i in onnx.load(onnx_path).graph.initializer}
    exact, compressed = [], []
    for name, arr in inits.items():
        key = name[:-len("_quantized")] if name.endswith("_quantized") else name
        if key not in pre or pre[key].shape != arr.shape:
            continue
        if name.endswith("_quantized"):
            scale = inits.get(key + "_scale")
            zero = inits.get(key + "_zero_point", np.zeros_like(scale) if scale is not None else None)
            if scale is None:
                continue
            restored = (arr.astype(np.float32) - zero.astype(np.float32)) * scale
            corr = float(np.corrcoef(restored.ravel(), pre[key].ravel())[0, 1])
            compressed.append({"weight": key, "correlation": round(corr, 6),
                               "max_difference": float(np.abs(restored - pre[key]).max())})
        else:
            exact.append({"weight": key, "max_difference": float(np.abs(arr - pre[key]).max())})
    return {"exact_compared": len(exact), "exact_identical": sum(e["max_difference"] == 0.0 for e in exact),
            "exact_max_difference": max((e["max_difference"] for e in exact), default=None),
            "compressed_compared": len(compressed),
            "compressed_min_correlation": min((c["correlation"] for c in compressed), default=None),
            "details": exact + compressed}


# ---------------------------------------------------------------- embeddings
def embed(model_or_path, texts: list[str], tokenizer=None, batch_size: int = 64, max_length: int = 64):
    """Mean-pooled last-layer vectors (normalised). Accepts a model name, a saved folder, or a model object."""
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    if isinstance(model_or_path, str):
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_or_path)
        model = AutoModel.from_pretrained(model_or_path)
    else:
        model = getattr(model_or_path, "base_model", model_or_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(texts[i:i + batch_size], truncation=True, max_length=max_length, padding=True,
                            return_tensors="pt").to(device)
            h = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
    return np.concatenate(out)


def separation(vectors, labels: list[str]) -> float:
    """Silhouette score: about 0 = routes mixed together, towards 1 = routes in clean separate groups."""
    from sklearn.metrics import silhouette_score
    return float(silhouette_score(vectors, labels, metric="cosine"))


def pca2(vectors):
    import numpy as np
    x = vectors - vectors.mean(0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


# ---------------------------------------------------------------- charts
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


COLORS = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1", "#9C755F"]


def chart_learning(infos: dict, path) -> None:
    """infos: {name: training_info}. Loss per step, and validation macro-F1 per epoch."""
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for i, (name, info) in enumerate(infos.items()):
        losses = info["step_losses"]
        k = max(1, len(losses) // 60)
        smooth = [sum(losses[j:j + k]) / len(losses[j:j + k]) for j in range(0, len(losses), k)]
        ax[0].plot([j * k for j in range(len(smooth))], smooth, label=name, color=COLORS[i % 8])
        ep = [h["epoch"] for h in info["history"]]
        ax[1].plot(ep, [h["mean_macro_f1"] for h in info["history"]], marker="o", label=name, color=COLORS[i % 8])
    ax[0].set(title="Training loss", xlabel="training step", ylabel="loss")
    ax[1].set(title="Validation macro-F1 (new wording + new style)", xlabel="epoch (0 = before training)",
              ylabel="macro-F1", ylim=(0, 1))
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_weight_change(infos: dict, path) -> None:
    """How far each part of the network moved from the pretrained weights, at the end of training.
    The pretrained layers and the new classification head are shown on separate scales."""
    plt = _plt()
    groups = list(next(iter(infos.values()))["history"][-1]["weight_change"].keys())
    body = sorted([g for g in groups if g != "classification head"], key=lambda g: (g != "embeddings", g))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [len(body), 1.6]})
    width = 0.8 / len(infos)
    for i, (name, info) in enumerate(infos.items()):
        wc = info["history"][-1]["weight_change"]
        ax[0].bar([j + i * width for j in range(len(body))], [wc.get(g, 0) for g in body], width, label=name,
                  color=COLORS[i % 8])
        ax[1].bar([i * width], [wc.get("classification head", 0)], width, label=name, color=COLORS[i % 8])
        for j, g in enumerate(body):
            if wc.get(g, 0) == 0:
                ax[0].text(j + i * width, 0, "0", ha="center", va="bottom", fontsize=7, color=COLORS[i % 8])
    ax[0].set_xticks([j + width * (len(infos) - 1) / 2 for j in range(len(body))], body, rotation=30, ha="right")
    ax[0].set(title="Pretrained layers: change from the downloaded weights", ylabel="||W - W0|| / ||W0||")
    ax[1].set_xticks([width * (len(infos) - 1) / 2], ["classification head\n(new, starts random)"])
    ax[1].set(title="New output layer")
    for a in ax:
        a.grid(alpha=0.3, axis="y")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_confusions(results: dict, path) -> None:
    """results: {router name: metrics()}. Rows = correct route, columns = route chosen; shares of each row."""
    plt = _plt()
    n = len(results)
    cols = min(3, n)
    rows_n = (n + cols - 1) // cols
    fig, grid = plt.subplots(rows_n, cols, figsize=(4.6 * cols, 4.6 * rows_n), squeeze=False)
    axes = [a for row in grid for a in row]
    for a in axes[n:]:
        a.axis("off")
    for k, (ax, (name, m)) in enumerate(zip(axes, results.items())):
        cm = m["confusion"]
        share = [[c / max(sum(row), 1) for c in row] for row in cm]
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(LABELS)), [SHORT[l] for l in LABELS], rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(LABELS)), [SHORT[l] for l in LABELS], fontsize=7)
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                if cm[i][j]:
                    ax.text(j, i, f"{share[i][j]:.0%}", ha="center", va="center", fontsize=6,
                            color="white" if share[i][j] > 0.5 else "black")
        ax.set_title(f"{name}\naccuracy {m['accuracy']:.1%}, macro-F1 {m['macro_f1']:.1%}", fontsize=9)
        ax.set_xlabel("route chosen", fontsize=8)
        if k % cols == 0:
            ax.set_ylabel("correct route", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_embeddings(before, after, labels: list[str], scores: tuple, path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for a, vec, title, s in ((ax[0], before, "Pretrained model (not fine-tuned)", scores[0]),
                             (ax[1], after, "After fine-tuning", scores[1])):
        xy = pca2(vec)
        for i, l in enumerate(LABELS):
            pts = [k for k, lab in enumerate(labels) if lab == l]
            a.scatter(xy[pts, 0], xy[pts, 1], s=8, alpha=0.6, color=COLORS[i], label=SHORT[l])
        a.set(title=f"{title}\nroute separation (silhouette) {s:.2f}", xticks=[], yticks=[])
    ax[1].legend(fontsize=7, markerscale=2, loc="best")
    fig.suptitle("Validation questions (new style), each dot one question, coloured by correct route", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_f1_per_route(results: dict, path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(11, 4))
    width = 0.8 / len(results)
    for i, (name, m) in enumerate(results.items()):
        ax.bar([j + i * width for j in range(len(LABELS))], [m["per_route"][l]["f1"] for l in LABELS], width,
               label=name, color=COLORS[i % 8])
    ax.set_xticks([j + width * (len(results) - 1) / 2 for j in range(len(LABELS))], [SHORT[l] for l in LABELS])
    ax.set(title="F1 per route", ylabel="F1", ylim=(0, 1))
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_cost_accuracy(points: list[dict], path) -> None:
    """points: [{name, macro_f1, cost_usd, seconds}]. Routers that use no AI model are drawn at the left edge;
    each point is named in the legend, with its time per question."""
    plt = _plt()
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, p in enumerate(points):
        ax.scatter(max(p["cost_usd"], 1e-7), p["macro_f1"], s=90, color=COLORS[i % 8], marker=markers[i % len(markers)],
                   edgecolor="black", linewidth=0.4, alpha=0.85,
                   label=f"{p['name']}  ({pct(p['macro_f1'])}, {p['seconds'] * 1000:.0f} ms)")
    ax.set_xscale("log")
    ax.set(title="Quality vs AI cost per question", xlabel="AI cost per question (USD, log scale; no-AI routers at the left edge)",
           ylabel="macro-F1", ylim=(0, 1.02))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5), title="router (macro-F1, time per question)",
              title_fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_training_size(points: list[tuple], path) -> None:
    """points: [(training questions, macro-F1)]"""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    xs, ys = zip(*sorted(points))
    ax.plot(xs, ys, marker="o", color=COLORS[0])
    ax.set(title="Fine-tuned router: validation macro-F1 vs training size", xlabel="training questions",
           ylabel="macro-F1", ylim=(0, 1))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_search(results: list[dict], path) -> None:
    """Heatmap of validation macro-F1 by learning rate and batch size, and each run's curve by epoch."""
    plt = _plt()
    lrs = sorted({r["learning_rate"] for r in results})
    bss = sorted({r["batch_size"] for r in results})
    grid = [[next((r["mean_macro_f1"] for r in results if r["learning_rate"] == lr and r["batch_size"] == bs), float("nan"))
             for lr in lrs] for bs in bss]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    im = ax[0].imshow(grid, cmap="Greens", vmin=min(min(g) for g in grid) - 0.02, vmax=1)
    ax[0].set_xticks(range(len(lrs)), [f"{lr:g}" for lr in lrs])
    ax[0].set_yticks(range(len(bss)), [str(b) for b in bss])
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            ax[0].text(j, i, f"{v:.1%}", ha="center", va="center", fontsize=9)
    ax[0].set(title="Best validation macro-F1", xlabel="learning rate", ylabel="batch size")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    for i, r in enumerate(results):
        ax[1].plot(range(len(r["curve"])), r["curve"], marker=".", color=COLORS[i % 8],
                   label=f"lr {r['learning_rate']:g}, batch {r['batch_size']}")
    ax[1].set(title="Validation macro-F1 by epoch", xlabel="epoch (0 = before training)", ylabel="macro-F1", ylim=(0, 1))
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- tables
def search_table(results: list[dict]) -> str:
    lines = ["| Learning rate | Batch size | Best epoch | New wording macro-F1 | New style macro-F1 | Mean | Training time |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['learning_rate']:g} | {r['batch_size']} | {r['best_epoch']} | {pct(r['val_new_wording'])} | "
                     f"{pct(r['val_new_style'])} | **{pct(r['mean_macro_f1'])}** | {r['training_seconds']:.0f} s |")
    return "\n".join(lines)



def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def comparison_table(by_router: dict[str, list[dict]]) -> str:
    lines = ["| Router | Questions | Accuracy (95% CI) | Macro-F1 (95% CI) | Seconds per question | AI cost per question |",
             "|---|---|---|---|---|---|"]
    for name, rows in by_router.items():
        m, ci = metrics(rows), bootstrap(rows)
        sec = sum(r["seconds"] for r in rows) / len(rows)
        cost = sum(r["cost_usd"] for r in rows) / len(rows)
        lines.append(f"| {name} | {m['n']} | {pct(m['accuracy'])} ({pct(ci['accuracy'][0])}–{pct(ci['accuracy'][1])}) | "
                     f"{pct(m['macro_f1'])} ({pct(ci['macro_f1'][0])}–{pct(ci['macro_f1'][1])}) | {sec:.4f} | ${cost:.6f} |")
    return "\n".join(lines)


def significance_table(by_router: dict[str, list[dict]], against: str) -> str:
    lines = [f"| {against} vs | Questions | Only {against} right | Only other right | p-value | Real difference? |",
             "|---|---|---|---|---|---|"]
    for name, rows in by_router.items():
        if name == against:
            continue
        t = mcnemar(by_router[against], rows)
        lines.append(f"| {name} | {t['questions']} | {t['only_a_right']} | {t['only_b_right']} | {t['p_value']:.4f} | "
                     f"{'yes (p < 0.05)' if t['p_value'] < 0.05 else 'not shown'} |")
    return "\n".join(lines)


def per_route_table(by_router: dict[str, list[dict]]) -> str:
    names = list(by_router)
    ms = {n: metrics(r) for n, r in by_router.items()}
    lines = ["| Route | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for l in LABELS:
        lines.append(f"| {l} | " + " | ".join(pct(ms[n]["per_route"][l]["f1"]) for n in names) + " |")
    return "\n".join(lines)


def training_table(infos: dict) -> str:
    lines = ["| Version | Training questions | Weights trained | Best epoch | New wording macro-F1 | New style macro-F1 | "
             "Training time |", "|---|---|---|---|---|---|---|"]
    for name, info in infos.items():
        b = info["best_epoch"]
        lines.append(f"| {name} | {info['train_questions']:,} | {info['trainable_weights']:,} of {info['total_weights']:,} | "
                     f"{b['epoch']} | {pct(b['val_new_wording'])} | {pct(b['val_new_style'])} | {info['training_seconds']:.0f} s |")
    return "\n".join(lines)


def epoch_table(info: dict) -> str:
    lines = ["| Epoch | Training loss | New wording acc. | New wording macro-F1 | New style acc. | New style macro-F1 | "
             "Encoder weight change |", "|---|---|---|---|---|---|---|"]
    for h in info["history"]:
        enc = [v for g, v in h["weight_change"].items() if g.startswith("encoder")]
        lines.append(f"| {h['epoch']} | {h['train_loss'] if h['train_loss'] is not None else '–'} | "
                     f"{pct(h['val_new_wording']['accuracy'])} | {pct(h['val_new_wording']['macro_f1'])} | "
                     f"{pct(h['val_new_style']['accuracy'])} | {pct(h['val_new_style']['macro_f1'])} | "
                     f"{(sum(enc) / len(enc) if enc else 0):.4f} |")
    return "\n".join(lines)


def group_by_router(rows: list[dict], sets=None) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for r in rows:
        if sets is None or r["set"] in sets:
            out[r["router"]].append(r)
    return dict(out)
