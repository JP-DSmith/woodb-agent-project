"""Backup routers for the comparison: the fine-tuned model, and prompted LLMs (a small and a large one).

Each router is a function: question text in, one of the 8 route labels out, as evaluation/compare_routers.py
expects.

    make_finetuned_router(model_dir)          a model trained by train_finetuned.py
    finetuned_route(question)                 the one in MODEL_DIR
    make_llm_router("openai/gpt-4o-mini")     a prompted LLM router (any OpenRouter model)

The LLM prompt is built only from the route descriptions in config/request_types_and_personas.json
("description" and "not_this"). It uses none of that file's example sentences and no test questions. With
few_shot > 0 it adds that many example questions per route, taken from the TRAINING split only.
"""
from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABELS = ["qualification_check", "rules_info", "sourcing_whatif", "duty_cost", "certification_draft",
          "document_intake", "escalate_human", "out_of_scope"]
MODEL_DIR = os.environ.get("ROUTER_MODEL_DIR", str(Path.home() / "router_models" / "fine_tuned"))

# ---------------------------------------------------------------- fine-tuned
def make_finetuned_router(model_dir: str = MODEL_DIR):
    """Router for a model saved by train_finetuned.py (loaded on first use)."""
    state = {}

    def route(question: str) -> str:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if not state:
            info = Path(model_dir) / "training_info.json"
            state["tok"] = AutoTokenizer.from_pretrained(model_dir)
            state["model"] = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
            state["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            state["model"].to(state["device"])
            state["max_length"] = json.loads(info.read_text())["settings"]["max_length"] if info.exists() else 64
        enc = state["tok"](question, truncation=True, max_length=state["max_length"], return_tensors="pt").to(state["device"])
        with torch.no_grad():
            return LABELS[int(state["model"](**enc).logits.argmax(-1))]

    route.__name__ = f"finetuned_{Path(model_dir).name}"
    return route


_default = {}


def finetuned_route(question: str) -> str:
    """The model in MODEL_DIR (the default fine-tuned router)."""
    if "route" not in _default:
        _default["route"] = make_finetuned_router(MODEL_DIR)
    return _default["route"](question)


def reset_finetuned() -> None:
    """Forget the loaded default model (after retraining)."""
    _default.clear()


# ---------------------------------------------------------------- frozen encoder + simple classifier
def make_probe_router(base_model: str, train_file: str = "train_styled", max_length: int = 64, seed: int = 894):
    """The fair "no fine-tuning" baseline: the pretrained model is left exactly as downloaded; each question becomes the
    average of its word vectors (as the current router uses), and only a simple classifier (logistic regression) is
    trained on top. Trains on first use. route.info has the training size and time."""
    state = {}

    def fit():
        import time
        from sklearn.linear_model import LogisticRegression
        from transformers import AutoModel, AutoTokenizer
        from fine_tuning.backup import analysis as an
        started = time.time()
        rows = [json.loads(l) for l in (ROOT / "data" / "router_split" / f"{train_file}.jsonl").open(encoding="utf-8")]
        state["tok"] = AutoTokenizer.from_pretrained(base_model)
        state["enc"] = AutoModel.from_pretrained(base_model)          # loaded once, never trained
        x = an.embed(state["enc"], [r["query"] for r in rows], tokenizer=state["tok"], max_length=max_length)
        state["clf"] = LogisticRegression(max_iter=3000, C=10.0, random_state=seed).fit(x, [r["intent"] for r in rows])
        route.info = {"train_questions": len(rows), "training_seconds": round(time.time() - started, 1),
                      "weights_trained": int(state["clf"].coef_.size + state["clf"].intercept_.size)}

    def route(question: str) -> str:
        from fine_tuning.backup import analysis as an
        if not state:
            fit()
        return str(state["clf"].predict(an.embed(state["enc"], [question], tokenizer=state["tok"], max_length=max_length))[0])

    route.info = {}
    route.__name__ = "frozen_probe"
    return route


# ---------------------------------------------------------------- prompted LLM
def route_descriptions() -> str:
    cfg = json.loads((ROOT / "config" / "request_types_and_personas.json").read_text(encoding="utf-8"))
    lines = []
    for rt in cfg["request_types"]:
        line = f"- {rt['id']}: {rt['description']}"
        if rt.get("not_this"):
            line += f" (Not this: {rt['not_this']})"
        lines.append(line)
    return "\n".join(lines)


def few_shot_examples(per_route: int, seed: int = 894) -> str:
    if per_route <= 0:
        return ""
    rows = [json.loads(l) for l in (ROOT / "data" / "router_split" / "train.jsonl").open(encoding="utf-8")]
    rng, out = random.Random(seed), []
    for label in LABELS:
        for r in rng.sample([r for r in rows if r["intent"] == label], per_route):
            out.append(f"Question: {r['query']}\nRoute: {label}")
    return "\n\nExamples:\n\n" + "\n\n".join(out)


PROMPT = """You route questions sent to a CUSMA (USMCA) rules-of-origin assistant for Northfield Rubber Footwear,
a Canadian maker of rubber boots. Choose the ONE route that fits the question best.

Routes:
{routes}
{examples}

Answer with the route name only, exactly as written above, and nothing else."""


def make_llm_router(model_id: str, few_shot: int = 0, temperature: float = 0.0, cache_path=None):
    """A prompted LLM router. With cache_path, every answer is saved to that JSON file (keyed by model, number of
    examples and question) with its tokens, cost and time, so running the comparison again doesn't pay for the same
    questions twice and still reports what they cost. route.seconds[question] holds the original time of a reused
    answer (see restore_seconds); route.unpriced counts reused answers saved without their cost."""
    import time
    from agents import base
    system = PROMPT.format(routes=route_descriptions(), examples=few_shot_examples(few_shot))
    model, cache = {}, {}
    cache_file = Path(cache_path) if cache_path else None
    if cache_file and cache_file.exists():
        cache.update(json.loads(cache_file.read_text(encoding="utf-8")))

    def route(question: str) -> str:
        key = f"{model_id}|{few_shot}|{temperature}|{question}"
        meter = base._meter.get()
        hit = cache.get(key)
        if hit is not None:
            if isinstance(hit, str):                 # saved by an older version: answer only
                text = hit
                route.unpriced += 1
            else:
                text = hit["text"]
                if meter is not None:                # count the original call's tokens and cost again
                    meter.add(model_id, hit.get("prompt_tokens", 0), hit.get("completion_tokens", 0), hit.get("cost_usd"))
                route.seconds[question] = hit.get("seconds")
        else:
            if "m" not in model:
                model["m"] = base.OpenRouterModel(model=model_id, temperature=temperature)
            before = meter.as_dict() if meter is not None else None
            started = time.perf_counter()
            reply = model["m"]([{"role": "system", "content": system},
                                {"role": "user", "content": f"Question: {question}\nRoute:"}], [])
            text = (reply.get("content") or "").strip()
            if cache_file:
                entry = {"text": text, "seconds": round(time.perf_counter() - started, 4)}
                if meter is not None:
                    after = meter.as_dict()
                    entry |= {"prompt_tokens": after["prompt_tokens"] - before["prompt_tokens"],
                              "completion_tokens": after["completion_tokens"] - before["completion_tokens"],
                              "cost_usd": None if after["cost_usd"] is None else
                              round(after["cost_usd"] - (before["cost_usd"] or 0.0), 8)}
                cache[key] = entry
                if len(cache) % 25 == 0:
                    route.save()
        found = [l for l in LABELS if re.search(rf"\b{l}\b", text.lower())]
        return found[0] if len(found) == 1 else "escalate_human"

    def save():
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    route.save = save
    route.seconds = {}
    route.unpriced = 0
    route.__name__ = f"llm_{model_id.split('/')[-1]}"
    return route


def restore_seconds(rows: list[dict], name: str, route) -> int:
    """For answers reused from the cache, put back the time the original call took (a reused answer takes ~0 s,
    which would make the LLM look instant). Returns how many rows were changed."""
    changed = 0
    for r in rows:
        if r["router"] == name and route.seconds.get(r["question"]) is not None:
            r["seconds"] = route.seconds[r["question"]]
            changed += 1
    return changed
