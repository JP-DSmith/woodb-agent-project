"""Router economics: monthly cost at different volumes, and when fine-tuning pays off.

Uses the measured cost per question and accuracy of each router (from the comparison) and a few stated
assumptions (the INPUTS below; change them in the notebook). For each router and monthly volume:

    monthly cost = fixed monthly cost + volume x (cost per question + error rate x cost of a misroute)
                   + one-time cost spread over the amortisation period

A misroute costs staff time: the question goes to the wrong specialist and a person has to redirect it.
Break-even volume (fine-tuned vs an LLM router) is the monthly volume where the fine-tuned router's lower running
cost pays for its one-time and fixed costs.
"""
from __future__ import annotations

INPUTS = {
    "gpu_usd_per_hour": 0.50,          # cloud GPU for training (Colab's free T4 costs nothing; this is a paid one)
    "people_hours_one_time": 12,       # building, tuning and checking the fine-tuned model
    "people_usd_per_hour": 60,
    "retrains_per_year": 4,            # new boots, suppliers or rules mean retraining
    "people_hours_per_retrain": 3,
    "hosting_usd_per_month": 15,       # a small always-on container for the fine-tuned model
    "cpu_usd_per_second": 0.000029,    # pay-per-use container time for each question (fine-tuned model)
    "misroute_usd": 5.00,              # about 5 minutes of staff time to redirect a wrongly routed question (an assumption)
    "amortise_months": 12,
}
VOLUMES = (1_000, 100_000, 1_000_000)
MISROUTE_LEVELS = (0.50, 5.00)        # misroute costs shown in the break-even table (a sensitivity check)


def one_time_usd(training_seconds: float, inputs: dict = INPUTS) -> float:
    return training_seconds / 3600 * inputs["gpu_usd_per_hour"] + inputs["people_hours_one_time"] * inputs["people_usd_per_hour"]


def monthly_fixed_usd(kind: str, inputs: dict = INPUTS) -> float:
    if kind != "finetuned":
        return 0.0
    retrain = inputs["retrains_per_year"] * inputs["people_hours_per_retrain"] * inputs["people_usd_per_hour"] / 12
    return inputs["hosting_usd_per_month"] + retrain


def per_question_usd(router: dict, inputs: dict = INPUTS, misroutes: bool = True) -> float:
    """Running cost per question, plus (unless misroutes=False) the expected cost of misroutes."""
    run = router["cost_usd"] if router["kind"] != "finetuned" else router["seconds"] * inputs["cpu_usd_per_second"]
    return run + ((1 - router["accuracy"]) * inputs["misroute_usd"] if misroutes else 0.0)


def monthly_usd(router: dict, volume: int, training_seconds: float = 0.0, inputs: dict = INPUTS,
                misroutes: bool = True) -> float:
    fixed = monthly_fixed_usd(router["kind"], inputs)
    if router["kind"] == "finetuned":
        fixed += one_time_usd(training_seconds, inputs) / inputs["amortise_months"]
    return fixed + volume * per_question_usd(router, inputs, misroutes)


def break_even_volume(ft: dict, llm: dict, training_seconds: float, inputs: dict = INPUTS,
                      misroutes: bool = True) -> float | None:
    """Monthly questions at which the fine-tuned router becomes cheaper than the LLM router (None: never).
    misroutes=False compares running costs only, as if both were equally accurate."""
    saving = per_question_usd(llm, inputs, misroutes) - per_question_usd(ft, inputs, misroutes)
    if saving <= 0:
        return None
    fixed = monthly_fixed_usd("finetuned", inputs) + one_time_usd(training_seconds, inputs) / inputs["amortise_months"]
    return fixed / saving


def table(routers: list[dict], training_seconds: float, inputs: dict = INPUTS) -> str:
    """routers: [{"name", "kind" ("finetuned" | "llm" | "current"), "accuracy" (0-1), "cost_usd", "seconds"}]

    Three tables: what it costs to run each router; how many questions it would misroute (whose cost is an
    assumption, so it is kept apart); and the volume at which fine-tuning pays off, with and without misroutes."""
    vols = " | ".join(f"{v:,}/month" for v in VOLUMES)
    sep = "|---" * (3 + len(VOLUMES)) + "|"
    fixed = monthly_fixed_usd("finetuned", inputs) + one_time_usd(training_seconds, inputs) / inputs["amortise_months"]
    lines = ["**A. Running cost per month** (what it costs to run the router itself)", "",
             f"| Router | Accuracy | Cost per question | {vols} |", sep]
    for r in routers:
        run = per_question_usd(r, inputs, misroutes=False)
        cells = [f"${monthly_usd(r, v, training_seconds, inputs, misroutes=False):,.2f}" for v in VOLUMES]
        lines.append(f"| {r['name']} | {r['accuracy']:.1%} | ${run:.6f} | " + " | ".join(cells) + " |")
    lines += ["", f"The fine-tuned routers have a fixed cost of about ${fixed:,.0f} a month whatever the volume: hosting "
              f"(${inputs['hosting_usd_per_month']:,.0f}), {inputs['retrains_per_year']} retrains a year and the one-time build "
              f"({inputs['people_hours_one_time']} hours at ${inputs['people_usd_per_hour']:,.0f}) spread over "
              f"{inputs['amortise_months']} months. The prompted LLMs have no fixed cost, only a fee per question.", "",
              f"**B. Misrouted questions per month** (from each router's error rate on the validation questions)", "",
              f"| Router | Error rate | {vols} | Cost at ${inputs['misroute_usd']:.2f} each, {VOLUMES[1]:,}/month |", sep]
    for r in routers:
        err = 1 - r["accuracy"]
        lines.append(f"| {r['name']} | {err:.1%} | " + " | ".join(f"{err * v:,.0f}" for v in VOLUMES) +
                     f" | ${err * VOLUMES[1] * inputs['misroute_usd']:,.0f} |")
    lines += ["", f"A misroute sends the question to the wrong specialist, so someone has to redirect or re-ask it. Its cost "
              f"(${inputs['misroute_usd']:.2f}, about five minutes of staff time) is an assumption, not a measurement, and at high "
              "volume it outweighs the cost of running any router. These error rates are from the validation questions, "
              "which are deliberately hard (new wording and new writing styles); everyday questions may be easier."]
    ft = next((r for r in routers if r["kind"] == "finetuned" and r["name"] == "fine-tuned"),
              next((r for r in routers if r["kind"] == "finetuned"), None))
    if ft:
        fmt = lambda v: f"about {v:,.0f}" if v is not None else "never"
        head = " | ".join(f"Including misroutes at ${m:.2f}" for m in MISROUTE_LEVELS)
        lines += ["", f"**C. When fine-tuning pays off** (monthly questions above which the '{ft['name']}' router is cheaper)", "",
                  f"| Against | Running cost only | {head} |", "|---" * (2 + len(MISROUTE_LEVELS)) + "|"]
        for r in routers:
            if r["kind"] == "llm":
                cells = [fmt(break_even_volume(ft, r, training_seconds, inputs, misroutes=False))]
                cells += [fmt(break_even_volume(ft, r, training_seconds, {**inputs, "misroute_usd": m})) for m in MISROUTE_LEVELS]
                lines.append(f"| {r['name']} | " + " | ".join(cells) + " |")
        lines += ["", "'Never' means the fine-tuned router is not cheaper per question at that setting, so no volume pays "
                  "for its fixed costs: on running cost alone, because the LLM's fee is lower than the fine-tuned router's "
                  "compute; with misroutes, because the LLM's higher accuracy saves more than its fee costs."]
    return "\n".join(lines)
