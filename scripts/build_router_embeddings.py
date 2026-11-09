import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = (
    ROOT
    / "config"
    / "request_types_and_personas.json"
)

TRAIN_PATH = (
    ROOT
    / "data"
    / "train_dataset.jsonl"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "router_reference_embeddings.npz"
)

MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


def load_examples():
    texts = []
    labels = []

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    for request_type in config.get(
        "request_types",
        [],
    ):
        label = request_type["id"]

        examples = request_type.get(
            "examples",
            {},
        )

        for language in ("en", "fr"):
            for example in examples.get(
                language,
                [],
            ):
                texts.append(example)
                labels.append(label)

    with TRAIN_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if not line.strip():
                continue

            row = json.loads(line)

            texts.append(row["query"])
            labels.append(row["intent"])

    return texts, labels


def mean_pool(
    token_embeddings,
    attention_mask,
):
    expanded_mask = (
        attention_mask
        .unsqueeze(-1)
        .expand(token_embeddings.size())
        .float()
    )

    summed = torch.sum(
        token_embeddings * expanded_mask,
        dim=1,
    )

    counts = torch.clamp(
        expanded_mask.sum(dim=1),
        min=1e-9,
    )

    return summed / counts


texts, labels = load_examples()

print(
    "Reference examples:",
    len(texts),
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModel.from_pretrained(
    MODEL_NAME
)

model.eval()

all_embeddings = []

BATCH_SIZE = 64

with torch.no_grad():
    for start in range(
        0,
        len(texts),
        BATCH_SIZE,
    ):
        batch = texts[
            start:start + BATCH_SIZE
        ]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        output = model(**inputs)

        embeddings = mean_pool(
            output[0],
            inputs["attention_mask"],
        )

        all_embeddings.append(
            embeddings.cpu().numpy()
        )

        print(
            "Embedded:",
            min(
                start + BATCH_SIZE,
                len(texts),
            ),
            "/",
            len(texts),
        )


matrix = np.vstack(
    all_embeddings
).astype("float32")


# Normalize once so runtime cosine similarity
# becomes a fast dot product.
norms = np.linalg.norm(
    matrix,
    axis=1,
    keepdims=True,
)

matrix = (
    matrix
    / np.maximum(
        norms,
        1e-12,
    )
)


np.savez_compressed(
    OUTPUT_PATH,
    embeddings=matrix,
    texts=np.asarray(
        texts,
        dtype=str,
    ),
    labels=np.asarray(
        labels,
        dtype=str,
    ),
)


print()
print(
    "Saved:",
    OUTPUT_PATH,
)

print(
    "Shape:",
    matrix.shape,
)

print(
    "Size:",
    OUTPUT_PATH.stat().st_size,
    "bytes",
)
