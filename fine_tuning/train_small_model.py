"""Fine-tune the small router model (all-MiniLM-L6-v2 with an 8-route classification head).

    python fine_tuning/train_small_model.py

Runs fine_tuning/backup/train_finetuned.py with its default settings (the ones the settings search chose:
learning rate 5e-5, batch size 32, best epoch kept on validation). Change them with the ROUTER_SETTINGS
environment variable, e.g. ROUTER_SETTINGS='{"epochs": 6, "model_dir": "/tmp/router"}'. Needs PyTorch and
transformers (requirements-router-build.txt) and ideally a GPU. Trains on data/router_split only; the test
sets are never read. The full study (settings search, controlled comparisons, LLM routers, economics) is in
fine_tuning/README_router_study.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fine_tuning.backup import train_finetuned  # noqa: E402

if __name__ == "__main__":
    train_finetuned.train()
