import collections
import json
import time
from pathlib import Path

from test_lightweight_router import route


ROOT = Path(__file__).resolve().parent.parent

TEST_FILE = (
    ROOT
    / "data"
    / "test_holdout_human_reviewed.jsonl"
)


rows = [
    json.loads(line)
    for line in TEST_FILE.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]


correct = collections.Counter()
total = collections.Counter()
wrong = []

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

    total[expected] += 1

    if predicted == expected:
        correct[expected] += 1
    else:
        wrong.append(
            (
                expected,
                predicted,
                result["score"],
                row["query"],
            )
        )


elapsed = time.perf_counter() - start

total_correct = sum(
    correct.values()
)

accuracy = (
    total_correct / len(rows)
)


print()
print("=" * 70)
print("LIGHTWEIGHT ONNX HUMAN-REVIEWED HOLDOUT")
print("=" * 70)

print(
    f"Overall: {total_correct}/{len(rows)} "
    f"= {accuracy:.1%}"
)

print()

for label in sorted(total):

    label_accuracy = (
        correct[label]
        / total[label]
    )

    print(
        f"{label:22s} "
        f"{correct[label]:4d}/"
        f"{total[label]:<4d} "
        f"{label_accuracy:.0%}"
    )


print()
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
) in wrong[:10]:

    print(
        f"wanted {expected:20s} "
        f"got {predicted:20s} "
        f"score {score:.2f} | "
        f"{question[:90]}"
    )


print()
print("=" * 70)

if accuracy < 0.80:
    raise SystemExit(
        "FAIL: Holdout accuracy fell below 80%."
    )

print(
    "PASS: Lightweight router retains "
    "acceptable holdout accuracy."
)
