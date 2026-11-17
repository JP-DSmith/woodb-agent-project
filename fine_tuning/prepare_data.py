"""Prepare the router training and validation data.

    python fine_tuning/prepare_data.py               # the template-based split (no model calls)
    python fine_tuning/paraphrase_questions.py       # optional: the style rewrites (calls gpt-4o-mini)

Builds data/router_split/ with fine_tuning/make_router_split.py: training templates, held-out templates for
validation (new wording), and validation questions from the training templates (for contrast). The split is
fixed (same seed, same result every time). See fine_tuning/README_router.md for what each file is for.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fine_tuning import make_router_split  # noqa: E402

if __name__ == "__main__":
    make_router_split.main()
