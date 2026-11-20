# Router comparison: prompted LLM router (gpt-4o-mini) vs ONNX router

Router only (no agents). Made with `evaluation/compare_routers.py`: routers `llm=agents.llm_router:route` (the LLM router on its own, no fallback) and `onnx=live` (agents/router_agent.py). Validation sets first; test sets run once for reporting. Per-question predictions (ids only): `router_comparison_llm_vs_onnx_predictions.csv`.

| Figure used in the README | LLM router | ONNX router |
|---|---|---|
| Test sets, all (653 questions) | 92.3% | 79.8% |
| Hand-written questions (70) | 80.0% | 54.3% |
| Validation, new wording + new style (1623) | 97.4% | 96.3% (built from all templates, so inflated) |
| Seconds per question | 0.515 | 0.023 |
| Cost per question (USD) | 0.000089 | 0.000000 |

# Router comparison

Final test run: test sets included. Do not tune anything after this run.

## Accuracy

| Router | val_new_wording | val_new_style | val_seen_wording | holdout_seen | holdout_new | team_ai | team_hand | human_written | Seconds per question | Cost per question (USD) |
|---|---|---|---|---|---|---|---|---|---|---|
| llm | 98.7% (539/546) | 96.7% (1041/1077) | 97.6% (368/377) | 97.2% (242/249) | 98.4% (247/251) | 65.0% (39/60) | 82.6% (19/23) | 80.0% (56/70) | 0.5148 | 0.000089 |
| onnx | 99.3% (542/546) | 94.8% (1021/1077) | 97.9% (369/377) | 97.2% (242/249) | 78.9% (198/251) | 56.7% (34/60) | 39.1% (9/23) | 54.3% (38/70) | 0.0233 | 0.000000 |

## Accuracy per route (val_new_wording, val_new_style, holdout_new, team_ai, team_hand, human_written combined)

| Route | llm | onnx |
|---|---|---|
| qualification_check | 99.3% (425/428) | 94.6% (405/428) |
| rules_info | 89.0% (202/227) | 82.8% (188/227) |
| sourcing_whatif | 97.2% (315/324) | 90.4% (293/324) |
| duty_cost | 95.9% (283/295) | 95.3% (281/295) |
| certification_draft | 98.0% (242/247) | 96.4% (238/247) |
| document_intake | 94.1% (207/220) | 89.5% (197/220) |
| escalate_human | 87.1% (128/147) | 86.4% (127/147) |
| out_of_scope | 100.0% (139/139) | 81.3% (113/139) |

## Most common mistakes

**llm:** escalate_human -> qualification_check (16); rules_info -> out_of_scope (12); duty_cost -> qualification_check (11); sourcing_whatif -> qualification_check (9); rules_info -> qualification_check (7)

**onnx:** out_of_scope -> escalate_human (21); rules_info -> escalate_human (18); sourcing_whatif -> escalate_human (17); document_intake -> sourcing_whatif (12); qualification_check -> escalate_human (11)
