import json
import time
from pathlib import Path

import numpy as np

from test_lightweight_router import route


ROOT = Path(__file__).resolve().parent.parent

VAL_FILE = (
    ROOT
    / "data"
    / "val_dataset.jsonl"
)


rows = [
    json.loads(line)
    for line in VAL_FILE.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]


correct = 0
escalated = 0
mistakes = []

start = time.perf_counter()


for row in rows:

    result = route(
        row["query"]
    )

    predicted = result[
        "predicted_class"
    ]

    expected = row[
        "intent"
    ]

    if predicted == expected:
        correct += 1
    else:
        mistakes.append(
            (
                expected,
                predicted,
                result["score"],
                row["query"],
            )
        )

    if predicted == "escalate_human":
        escalated += 1


elapsed = (
    time.perf_counter()
    - start
)

accuracy = (
    correct / len(rows)
)


print()
print("=" * 65)
print("LIGHTWEIGHT ONNX VALIDATION")
print("=" * 65)

print(
    "Validation records:",
    len(rows),
)

print(
    "Correct:",
    correct,
)

print(
    "Accuracy:",
    f"{accuracy:.1%}",
)

print(
    "Escalated:",
    escalated,
)

print(
    "Total runtime:",
    f"{elapsed:.3f} seconds",
)

print(
    "Average/question:",
    f"{elapsed / len(rows) * 1000:.1f} ms",
)


print()
print("First mistakes:")

for (
    expected,
    predicted,
    score,
    question,
) in mistakes[:10]:

    print(
        f"wanted {expected:20s} "
        f"got {predicted:20s} "
        f"score {score:.2f} | "
        f"{question[:90]}"
    )


print()
print("=" * 65)

if accuracy < 0.95:

    raise SystemExit(
        "FAIL: Lightweight router accuracy "
        "fell below 95%."
    )

print(
    "PASS: Lightweight router accuracy "
    "remains above 95%."
)
