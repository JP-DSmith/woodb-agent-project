# Router tuning and comparison

## The question sets

`data/router_split/` (made by `fine_tuning/make_router_split.py`, fixed seed, same result every time):

| File | What it is | Use it for |
|---|---|---|
| `train.jsonl` | Questions from the training templates (1,577) | Training a model, or as router examples |
| `val_new_wording.jsonl` | Every question from 3 held-out templates per route (546) | Tuning and comparing routers. Wording no router has trained on |
| `val_seen_wording.jsonl` | Validation questions from the training templates (377) | Contrast only: shows the gap between familiar and new wording |
| `train_styled.jsonl` | The training questions plus 3 AI rewrites of each in other writing styles | Training, so the router learns more ways people write |
| `val_new_style.jsonl` | The new-wording questions rewritten in 2 other styles | Tuning: new wording AND new style, closest to real users |
| `paraphrase_report.json` | Counts, rewrites dropped by the checks, cost | Report |

`data/val_dataset.jsonl` uses the same templates as training, so on its own it can't show how a router copes
with new wording. `val_new_wording` can: the current method scores about 98% on familiar wording and about 81%
on new wording here, close to what the final test showed.

The styled files come from `fine_tuning/paraphrase_questions.py`. It asks gpt-4o-mini to rewrite each question in
general styles (casual, rushed with typos, wordy, formal email, simple English, key words only, indirect,
synonyms, outside point of view, French). The meaning is kept, so the route label is kept. Rewrites that lose a
code (NB-0413, PO-8013 ...), copy the question or come back empty are dropped. The paraphrase script generates `evaluation/results/paraphrase_spot_check.xlsx` with 50 randomly selected rewrites for human spot-checking before the augmented data is treated as validated evidence. The styles are general descriptions, not copied from any test question.

Never train or tune on the test sets (holdout, attacks, team questions, hand-written questions). They are used
once, in the final run.

## Comparing routers

A router is a function: question text in, route label out. The labels are qualification_check, rules_info,
sourcing_whatif, duty_cost, certification_draft, document_intake, escalate_human and out_of_scope; anything else
counts as escalate_human.
Put it in any module, then:

```
python evaluation/compare_routers.py --router nearest_neighbour --router finetuned=fine_tuning.my_router:route --out evaluation/results/router_val
```

Built-in routers: `nearest_neighbour` (the current method, rebuilt from `train.jsonl`),
`nearest_neighbour_styled` (the same with the rewrites added), `live` (the router the
system uses now: it trained on all templates, so judge it on the final test sets only) and `keyword`.

The report gives accuracy per set, accuracy per route, the most common mistakes, seconds per question and cost
per question. A router that calls an LLM should use `agents.base.OpenRouterModel` so its tokens and cost are
counted.

## Final comparison (once)

When tuning is finished, add `--final`. This adds the holdout (familiar and new templates), the team-written
questions (AI-assisted and hand-written separately) and the hand-written question file if present. Report these
numbers; don't change the routers after seeing them. If anything is changed afterwards, log it in the prompt
change log (T35).
