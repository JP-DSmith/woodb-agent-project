"""Compare routing models on the same questions: accuracy (overall and per route), time and cost.

    python evaluation/evaluate_models.py --router live --router keyword
    python evaluation/evaluate_models.py --router finetuned=fine_tuning.backup.routers:finetuned_route
    python evaluation/evaluate_models.py ... --final --out evaluation/results/router_final   # test sets, once

A thin entry point to evaluation/compare_routers.py (see its help for all options and built-in routers).
Validation sets by default; --final adds the test sets and is run once, after all tuning. The router study's
results (fine-tuned vs prompted small and large models) are in fine_tuning/results/router_study/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation import compare_routers  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(compare_routers.main(sys.argv[1:] or ["--router", "live", "--router", "keyword"]))
