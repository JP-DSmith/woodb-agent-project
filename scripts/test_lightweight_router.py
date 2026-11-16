from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT / "models" / "router" / "model.onnx"
)

TOKENIZER_PATH = (
    ROOT / "models" / "router" / "tokenizer.json"
)

REFERENCE_PATH = (
    ROOT
    / "data"
    / "router_reference_embeddings.npz"
)

THRESHOLD = 0.505


# ------------------------------------------------------------
# Load precomputed references
# ------------------------------------------------------------

reference_data = np.load(
    REFERENCE_PATH
)

reference_embeddings = (
    reference_data["embeddings"]
)

reference_texts = (
    reference_data["texts"]
)

reference_labels = (
    reference_data["labels"]
)


# ------------------------------------------------------------
# Load tokenizer and ONNX model
# ------------------------------------------------------------

tokenizer = Tokenizer.from_file(
    str(TOKENIZER_PATH)
)

tokenizer.enable_truncation(
    max_length=256
)

tokenizer.enable_padding(
    pad_id=0,
    pad_token="[PAD]",
)


options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1

session = ort.InferenceSession(
    str(MODEL_PATH),
    sess_options=options,
    providers=["CPUExecutionProvider"],
)


# ------------------------------------------------------------
# Generate one question embedding
# ------------------------------------------------------------

def embed(text: str) -> np.ndarray:

    encoding = tokenizer.encode(text)

    input_ids = np.asarray(
        [encoding.ids],
        dtype=np.int64,
    )

    attention_mask = np.asarray(
        [encoding.attention_mask],
        dtype=np.int64,
    )

    available_inputs = {
        item.name
        for item in session.get_inputs()
    }

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    if "token_type_ids" in available_inputs:
        inputs["token_type_ids"] = np.asarray(
            [encoding.type_ids],
            dtype=np.int64,
        )

    token_embeddings = session.run(
        None,
        inputs,
    )[0]

    mask = attention_mask[
        ..., None
    ].astype(np.float32)

    embedding = np.sum(
        token_embeddings * mask,
        axis=1,
    )

    embedding /= np.maximum(
        np.sum(mask, axis=1),
        1e-9,
    )

    embedding /= np.maximum(
        np.linalg.norm(
            embedding,
            axis=1,
            keepdims=True,
        ),
        1e-12,
    )

    return embedding[0]


# ------------------------------------------------------------
# Route one question
# ------------------------------------------------------------

def route(text: str):

    query_embedding = embed(text)

    # Reference vectors are already normalized.
    similarities = (
        reference_embeddings
        @ query_embedding
    )

    best_index = int(
        np.argmax(similarities)
    )

    score = float(
        similarities[best_index]
    )

    matched_label = str(
        reference_labels[best_index]
    )

    predicted = (
        "escalate_human"
        if score < THRESHOLD
        else matched_label
    )

    return {
        "predicted_class": predicted,
        "score": score,
        "matched_class": matched_label,
        "matched_example": str(
            reference_texts[best_index]
        ),
    }


# ------------------------------------------------------------
# Smoke tests
# ------------------------------------------------------------

TESTS = [
    (
        "Does NB-0101 qualify under CUSMA?",
        "qualification_check",
    ),
    (
        "How much duty for order PO-8017?",
        "duty_cost",
    ),
    (
        "Tell me a joke about boots.",
        "escalate_human",
    ),
    (
        "asdkjh qwe zzz",
        "escalate_human",
    ),
]


print("=" * 65)
print("LIGHTWEIGHT ROUTER TEST")
print("=" * 65)

correct = 0

for question, expected in TESTS:

    result = route(question)

    passed = (
        result["predicted_class"]
        == expected
    )

    correct += passed

    print()
    print(
        "PASS" if passed else "FAIL",
        "|",
        result["predicted_class"],
        f"| score {result['score']:.4f}",
    )

    print("Question:", question)
    print(
        "Matched:",
        result["matched_example"][:90],
    )


print()
print(
    f"Result: {correct}/{len(TESTS)} correct"
)
