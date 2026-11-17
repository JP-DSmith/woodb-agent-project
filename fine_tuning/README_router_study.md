# Router fine-tuning study (brief section 2)

Is fine-tuning a small model worth it for routing (choosing which of the 8 specialists answers a question),
compared with prompted small and large models? This folder holds the setup; the results and the conclusion are
in the report (§5 and Appendix D).

| File | What it does |
|---|---|
| `backup/train_finetuned.py` | Fine-tunes `all-MiniLM-L6-v2` with an 8-route head; logs validation scores and weight change per epoch; keeps the best epoch on validation |
| `backup/experiments.py` | Settings search (learning rate × batch size, validation only) and the controlled versions (frozen encoder, less data, other base models) |
| `backup/routers.py` | The routers compared: fine-tuned, frozen encoder + classifier, prompted LLMs (route definitions only; examples from training only) |
| `backup/analysis.py` | Accuracy, macro-F1, per-route F1, confusion matrices, bootstrap confidence intervals, McNemar tests, charts |
| `backup/economics.py` | Monthly cost at 1k / 100k / 1M questions and break-even volume |

**Data.** Training and tuning use `data/router_split/` only (see `README_router.md`). The final test sets are
run once, after tuning, through `evaluation/compare_routers.py --final`. `testing/test_backup_router.py`
checks that no validation or final question (including the team, hand-written and attack sets) appears in
the training data.

**Install and run.** `pip install -r requirements.txt -r requirements-research.txt` (a GPU is recommended for
training). One training run: `python fine_tuning/backup/train_finetuned.py`, with settings as JSON in
`ROUTER_SETTINGS`. Comparing routers: `python evaluation/compare_routers.py --router live
--router finetuned=fine_tuning.backup.routers:finetuned_route`. Tests: `pytest testing/test_backup_router.py`
(the training, chart and ONNX tests skip unless the research packages are installed).

**Economics are scenarios, not measurements.** `economics.py` combines each router's measured accuracy, time
and cost per question with stated assumptions (labour rate, build and retraining hours, hosting, cost of a
misroute, amortisation; see `INPUTS`). Its break-even volumes are conditional on those assumptions.
