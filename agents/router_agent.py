"""Lightweight semantic router for the CUSMA Trade Assistant.

Runtime design:
- ONNX Runtime instead of PyTorch
- precomputed reference embeddings
- one embedding generated per incoming question
- confidence-based human escalation
"""

import os
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


# ============================================================
# Repository paths
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT
    / "models"
    / "router"
    / "model.onnx"
)

TOKENIZER_PATH = (
    ROOT
    / "models"
    / "router"
    / "tokenizer.json"
)

REFERENCE_PATH = (
    ROOT
    / "data"
    / "router_reference_embeddings.npz"
)


# ============================================================
# Configuration
# ============================================================

SIMILARITY_THRESHOLD = float(
    os.getenv(
        "ROUTER_SIMILARITY_THRESHOLD",
        "0.505",
    )
)


AGENT_MAPPING = {
    "qualification_check": (
        "Qualification Agent "
        "(3a_qualification_agent)"
    ),
    "rules_info": (
        "Qualification Agent "
        "(3a_qualification_agent)"
    ),
    "sourcing_whatif": (
        "Sourcing Agent "
        "(3b_sourcing_agent)"
    ),
    "duty_cost": (
        "Duty and Cost Agent "
        "(3c_duty_cost_agent)"
    ),
    "certification_draft": (
        "Certification Agent "
        "(3d_certification_agent)"
    ),
    "document_intake": (
        "Intake Agent "
        "(1_intake_agent)"
    ),
    "escalate_human": (
        "Human Escalate (human)"
    ),
    "out_of_scope": (
        "Polite Decline (polite_decline)"
    ),
}


# ============================================================
# Lazy runtime state
# ============================================================

_tokenizer = None
_session = None

_reference_embeddings = None
_reference_texts = None
_reference_labels = None

_runtime_loaded = False

_load_lock = threading.RLock()


# ============================================================
# File validation
# ============================================================

def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            "Required router file is missing: "
            f"{path}"
        )


# ============================================================
# Lightweight runtime initialization
# ============================================================

def _load_runtime() -> None:
    global _tokenizer
    global _session
    global _reference_embeddings
    global _reference_texts
    global _reference_labels
    global _runtime_loaded

    if _runtime_loaded:
        return

    with _load_lock:
        if _runtime_loaded:
            return

        _require_file(MODEL_PATH)
        _require_file(TOKENIZER_PATH)
        _require_file(REFERENCE_PATH)

        # --------------------------------------------
        # Precomputed references
        # --------------------------------------------

        reference_data = np.load(
            REFERENCE_PATH
        )

        _reference_embeddings = (
            reference_data[
                "embeddings"
            ].astype(
                np.float32,
                copy=False,
            )
        )

        _reference_texts = (
            reference_data[
                "texts"
            ]
        )

        _reference_labels = (
            reference_data[
                "labels"
            ]
        )

        if (
            len(_reference_embeddings)
            == 0
        ):
            raise RuntimeError(
                "Router reference embeddings "
                "are empty."
            )

        if not (
            len(_reference_embeddings)
            == len(_reference_texts)
            == len(_reference_labels)
        ):
            raise RuntimeError(
                "Router reference data "
                "is not aligned."
            )

        # --------------------------------------------
        # Tokenizer
        # --------------------------------------------

        _tokenizer = (
            Tokenizer.from_file(
                str(TOKENIZER_PATH)
            )
        )

        _tokenizer.enable_truncation(
            max_length=256
        )

        _tokenizer.enable_padding(
            pad_id=0,
            pad_token="[PAD]",
        )

        # --------------------------------------------
        # ONNX Runtime
        # --------------------------------------------

        options = ort.SessionOptions()

        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1

        _session = ort.InferenceSession(
            str(MODEL_PATH),
            sess_options=options,
            providers=[
                "CPUExecutionProvider"
            ],
        )

        _runtime_loaded = True


# ============================================================
# Compatibility wrapper
#
# Existing tests import reference_texts and call len().
# ============================================================

class _LazyReferenceTexts(
    Sequence[str]
):

    def __len__(self) -> int:
        _load_runtime()

        return len(
            _reference_texts
        )

    def __getitem__(
        self,
        index,
    ):
        _load_runtime()

        return str(
            _reference_texts[
                index
            ]
        )

    def __iter__(
        self,
    ) -> Iterator[str]:
        _load_runtime()

        return (
            str(value)
            for value
            in _reference_texts
        )


reference_texts = (
    _LazyReferenceTexts()
)


# ============================================================
# ONNX embedding
# ============================================================

def _get_embedding(
    text: str,
) -> np.ndarray:

    _load_runtime()

    encoding = (
        _tokenizer.encode(
            text
        )
    )

    input_ids = np.asarray(
        [encoding.ids],
        dtype=np.int64,
    )

    attention_mask = np.asarray(
        [
            encoding.attention_mask
        ],
        dtype=np.int64,
    )

    available_inputs = {
        item.name
        for item
        in _session.get_inputs()
    }

    inputs = {
        "input_ids": input_ids,
        "attention_mask": (
            attention_mask
        ),
    }

    if (
        "token_type_ids"
        in available_inputs
    ):
        inputs[
            "token_type_ids"
        ] = np.asarray(
            [encoding.type_ids],
            dtype=np.int64,
        )

    outputs = _session.run(
        None,
        inputs,
    )

    token_embeddings = (
        outputs[0]
    )

    mask = attention_mask[
        ...,
        None,
    ].astype(
        np.float32
    )

    summed = np.sum(
        token_embeddings * mask,
        axis=1,
    )

    counts = np.maximum(
        np.sum(
            mask,
            axis=1,
        ),
        1e-9,
    )

    embedding = (
        summed / counts
    )

    norm = np.linalg.norm(
        embedding,
        axis=1,
        keepdims=True,
    )

    embedding = (
        embedding
        / np.maximum(
            norm,
            1e-12,
        )
    )

    return embedding[0]


# ============================================================
# Escalation result
# ============================================================

def _escalation_result(
    message: str,
    score: float = 0.0,
    matched_example: str = "",
    matched_class: str | None = None,
) -> dict:

    return {
        "message": message,
        "predicted_class": (
            "escalate_human"
        ),
        "target_agent": (
            AGENT_MAPPING[
                "escalate_human"
            ]
        ),
        "similarity_score": round(
            float(score),
            4,
        ),
        "matched_example": (
            matched_example
        ),
        "matched_class": (
            matched_class
        ),
        "router": "router_agent",
    }


# ============================================================
# Router
# ============================================================

def route_message_few_shot(
    message: str,
) -> dict:

    if not isinstance(
        message,
        str,
    ):
        raise TypeError(
            "Router message must "
            "be a string."
        )

    message = message.strip()

    if not message:
        return _escalation_result(
            message=""
        )
    query_embedding = (
        _get_embedding(
            message
        )
    )

    # Reference embeddings are
    # already normalized.
    similarities = (
        _reference_embeddings
        @ query_embedding
    )

    best_index = int(
        np.argmax(
            similarities
        )
    )

    best_score = float(
        similarities[
            best_index
        ]
    )

    best_label = str(
        _reference_labels[
            best_index
        ]
    )

    matched_example = str(
        _reference_texts[
            best_index
        ]
    )

    if (
        best_score
        < SIMILARITY_THRESHOLD
    ):
        return _escalation_result(
            message=message,
            score=best_score,
            matched_example=(
                matched_example
            ),
            matched_class=(
                best_label
            ),
        )

    return {
        "message": message,
        "predicted_class": (
            best_label
        ),
        "target_agent": (
            AGENT_MAPPING.get(
                best_label,
                AGENT_MAPPING[
                    "escalate_human"
                ],
            )
        ),
        "similarity_score": round(
            best_score,
            4,
        ),
        "matched_example": (
            matched_example
        ),
        "matched_class": (
            best_label
        ),
        "router": "router_agent",
    }


def route_user_input(
    message: str,
) -> dict:

    return route_message_few_shot(
        message
    )
