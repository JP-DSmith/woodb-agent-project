# Router study: fine-tuning vs prompted models (brief section 2)

**The subtask.** Routing: reading one question and choosing which of the 8 specialist routes should answer it
(`qualification_check`, `rules_info`, `sourcing_whatif`, `duty_cost`, `certification_draft`, `document_intake`,
`escalate_human`, `out_of_scope`). It is narrow, has one right answer per question, is the first step of every
request, and a wrong route means a wrong or missing answer. So it is the part of the system that suits a small,
fine-tuned model.

**The question.** Is fine-tuning a small model worth it here, compared with the router the system started with
and with prompted small and large LLMs? Every comparison changes one thing at a time, on the same questions.

## Data (no test data used for training or tuning)

| Use | Files | Questions |
|---|---|---|
| Training | `data/router_split/train.jsonl` (+ style rewrites in `train_styled.jsonl`) | 1,577 (6,264 with rewrites) |
| Validation (all tuning and choices) | `val_new_wording.jsonl` (3 held-out templates per route), `val_new_style.jsonl` (the same questions in other writing styles) | 546 + 1,077 |
| Test (run once, for reporting) | holdout (seen and new templates), team questions (AI-assisted and by hand), hand-written questions | 249 + 251 + 60 + 23 + 70 = 653 |

How the split and the rewrites were made: `fine_tuning/README_router.md`.

## What was done

1. **Is the starting router trained?** No. All 62 full-precision weight tables in `models/router/model.onnx`
   are identical to a fresh download of `all-MiniLM-L6-v2`: it compares questions with stored examples, so it
   recalls familiar wording well and generalises poorly.
2. **Fine-tuning.** `all-MiniLM-L6-v2` (the same base model) with an 8-route classification head, every layer
   trained (`fine_tuning/backup/train_finetuned.py`). Each epoch records loss, accuracy, macro-F1 and F1 per
   route on validation, and how far each layer's weights moved from the pretrained ones; the best epoch on
   validation is kept.
3. **Settings search** (validation only): learning rate 2e-5 / 5e-5 / 1e-4 × batch size 16 / 32, up to 6
   epochs each. Chosen: **5e-5, batch 32, epoch 2** (mean macro-F1 96.1%).
4. **Controlled versions:** frozen encoder (only the head trained), frozen encoder + logistic regression (the
   fair no-fine-tuning baseline), original questions only, 25% and 50% of the data, two other base models
   (DistilBERT, multilingual MiniLM).
5. **Prompted LLM routers**, on the same questions: gpt-4o-mini (small) and gpt-4o (large), each with and
   without example questions (examples from training only). The prompt holds only the route definitions.
6. **Statistics:** 95% bootstrap confidence intervals, exact McNemar tests against the fine-tuned router,
   confusion matrices, F1 per route, the sentence space before and after (PCA, silhouette).
7. **Economics:** monthly cost at 1k / 100k / 1M questions, break-even volume, with and without the cost of
   misroutes (`fine_tuning/backup/economics.py`).
8. **Full system:** the same 784 evaluation cases run through the whole pipeline three times, changing only the
   router (starting ONNX router, fine-tuned, gpt-4o-mini).

## Results

**Fine-tuning works** (validation, `results/router_study/validation/`): the weights move (most in the top
layers), route separation in the sentence space goes from 0.14 to 0.74, and macro-F1 goes from 8.8% before
training to 98.2% (new wording) / 94.0% (new style) at epoch 2. Training only the head reaches 65.0% / 63.4%.

**Test sets, run once** (`results/router_study/test/`):

| Router | Accuracy (95% CI) | Hand-written questions | Seconds / question | AI cost / question |
|---|---|---|---|---|
| Starting router (ONNX nearest neighbour) | 79.8% (76.7–82.7) | 54.3% | 0.006 | $0 |
| Frozen encoder + classifier | 83.0% (80.1–85.9) | – | 0.006 | $0 |
| **Fine-tuned MiniLM** | **88.2% (85.6–90.7)** | **62.9%** | 0.004 | $0 |
| gpt-4o-mini, prompted | 92.6% (90.5–94.6) | 81.4% | 0.71 | $0.000077 |
| gpt-4o, prompted | 95.3% (93.4–96.9) | 88.6% | 0.95 | $0.0013 |

- Fine-tuning beats the starting router (p < 0.001) and the frozen baseline (p < 0.001), but both prompted
  models beat it (p < 0.001), and the gap is largest on questions people wrote (63% vs 81%): the training data
  comes from templates, and the fine-tuned model learned the templates' wording. More data, the style
  rewrites and bigger base models made no significant difference on the test sets.
- Adding example questions to the prompts did not help reliably (worse on validation, slightly better on test,
  within the margin of error).

**Full system** (`results/router_study/full_system/`, same 784 cases): routing 79.9% (starting router) → 88.4%
(fine-tuned) → 93.3% (gpt-4o-mini); answers 64.6% → 65.7% → 67.4%; about +0.6 s and +$0.00009 per question for
gpt-4o-mini.

## When does fine-tuning pay off?

The fine-tuned router costs almost nothing per question but has fixed costs (hosting, retraining, the build).
On running cost alone it is cheaper than gpt-4o-mini only above about **1.7 million questions a month**
(about 104,000 against gpt-4o). Once misroutes are costed it never catches up, because the prompted models
are more accurate. It is still the right choice when speed (about 0.004 s vs 0.7 s), data residency (questions
never leave the server) or very high volume matter, and the plan would be to fine-tune on real, logged user
questions (labelled by the LLM, checked by a person) rather than templates.

**Conclusion (a negative result for fine-tuning at this volume):** a prompted gpt-4o-mini router is more
accurate, generalises better to how people write, and is cheaper at Northfield's volume. The fine-tuned model
is kept as the offline option. The prompted router is in `agents/llm_router.py` (PR #72), with the ONNX router as
fallback; its own committed evidence is `evaluation/evidence/router_comparison_llm_vs_onnx.md`.

## Reproduce

- One training run: `python fine_tuning/train_small_model.py` (settings via `ROUTER_SETTINGS`, JSON).
- The whole study: `fine_tuning/backup/router_study.ipynb` (VS Code or Colab, a GPU, an OpenRouter key for the
  LLM routers; about an hour and $8 on the full validation sets). Step 11 runs the test sets and is run once.
- Comparing any routers: `python evaluation/evaluate_models.py --router live --router keyword`.
- Training needs PyTorch and transformers (`requirements-router-build.txt`); the app does not (it runs the
  ONNX router). The study's tests (`testing/test_backup_router.py`) skip the training tests without PyTorch.
