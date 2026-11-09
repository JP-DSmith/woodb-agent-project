import json
import os
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


# ============================================================
# Repository paths
# ============================================================

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


# ============================================================
# Router configuration
# ============================================================

MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

SIMILARITY_THRESHOLD = float(
    os.getenv(
        "ROUTER_SIMILARITY_THRESHOLD",
        "0.52",
    )
)


# ============================================================
# Agent mapping
# ============================================================

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
# Lazy router state
# ============================================================

_reference_texts = []
_reference_labels = []

_tokenizer = None
_embedding_model = None
_reference_embeddings = None
_device = None

_examples_loaded = False
_router_loaded = False

# Re-entrant because _load_router() calls
# _load_reference_examples().
_load_lock = threading.RLock()


# ============================================================
# Required-file validation
# ============================================================

def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            "Required router file is missing: "
            f"{path}"
        )


# ============================================================
# Lazy example loading
# ============================================================

def _load_reference_examples() -> None:
    global _examples_loaded

    if _examples_loaded:
        return

    with _load_lock:

        if _examples_loaded:
            return

        _require_file(CONFIG_PATH)
        _require_file(TRAIN_PATH)

        # --------------------------------------------
        # Request-type examples
        # --------------------------------------------

        with CONFIG_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            config_data = json.load(file)

        for request_type in config_data.get(
            "request_types",
            [],
        ):
            label = request_type["id"]

            examples = request_type.get(
                "examples",
                {},
            )

            for language in (
                "en",
                "fr",
            ):
                for example in examples.get(
                    language,
                    [],
                ):
                    _reference_texts.append(
                        example
                    )

                    _reference_labels.append(
                        label
                    )

        # --------------------------------------------
        # Synthetic training dataset
        # --------------------------------------------

        with TRAIN_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line_number, line in enumerate(
                file,
                start=1,
            ):

                if not line.strip():
                    continue

                try:
                    item = json.loads(line)

                    query = item["query"]
                    intent = item["intent"]

                except (
                    json.JSONDecodeError,
                    KeyError,
                    TypeError,
                ) as error:

                    raise ValueError(
                        "Invalid router training "
                        "record at "
                        f"{TRAIN_PATH}:"
                        f"{line_number}"
                    ) from error

                _reference_texts.append(
                    query
                )

                _reference_labels.append(
                    intent
                )

        if not _reference_texts:
            raise RuntimeError(
                "Router loaded zero "
                "reference examples."
            )

        if (
            len(_reference_texts)
            != len(_reference_labels)
        ):
            raise RuntimeError(
                "Router reference texts and "
                "labels are not aligned."
            )

        _examples_loaded = True


# ============================================================
# Compatibility wrapper
#
# Existing tests import reference_texts and call len().
# This allows that behavior without loading the model.
# ============================================================

class _LazyReferenceTexts(
    Sequence[str]
):

    def __len__(self) -> int:
        _load_reference_examples()

        return len(
            _reference_texts
        )

    def __getitem__(
        self,
        index,
    ):
        _load_reference_examples()

        return _reference_texts[
            index
        ]

    def __iter__(
        self,
    ) -> Iterator[str]:
        _load_reference_examples()

        return iter(
            _reference_texts
        )


reference_texts = (
    _LazyReferenceTexts()
)


# ============================================================
# Embeddings
# ============================================================

def _get_embeddings(
    texts: list[str],
) -> np.ndarray:

    if (
        _tokenizer is None
        or _embedding_model is None
        or _device is None
    ):
        raise RuntimeError(
            "Router model has not "
            "been initialized."
        )

    inputs = _tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(_device)

    with torch.no_grad():
        outputs = (
            _embedding_model(
                **inputs
            )
        )

    attention_mask = (
        inputs[
            "attention_mask"
        ]
    )

    token_embeddings = (
        outputs[0]
    )

    expanded_mask = (
        attention_mask
        .unsqueeze(-1)
        .expand(
            token_embeddings.size()
        )
        .float()
    )

    sum_embeddings = (
        torch.sum(
            token_embeddings
            * expanded_mask,
            dim=1,
        )
    )

    sum_mask = torch.clamp(
        expanded_mask.sum(
            dim=1
        ),
        min=1e-9,
    )

    embeddings = (
        sum_embeddings
        / sum_mask
    )

    return (
        embeddings
        .cpu()
        .numpy()
    )


# ============================================================
# Lazy model loading
# ============================================================

def _load_router() -> None:
    global _tokenizer
    global _embedding_model
    global _reference_embeddings
    global _device
    global _router_loaded

    if _router_loaded:
        return

    with _load_lock:

        if _router_loaded:
            return

        _load_reference_examples()

        _device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        _tokenizer = (
            AutoTokenizer
            .from_pretrained(
                MODEL_NAME
            )
        )

        _embedding_model = (
            AutoModel
            .from_pretrained(
                MODEL_NAME
            )
        )

        _embedding_model = (
            _embedding_model.to(
                _device
            )
        )

        _embedding_model.eval()

        _reference_embeddings = (
            _get_embeddings(
                _reference_texts
            )
        )

        _router_loaded = True


# ============================================================
# Human escalation result
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
# Few-shot router
# =============================================

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

    # Model and embeddings are loaded
    # only when routing is first needed.
    _load_router()

    if (
        _reference_embeddings is None
        or len(
            _reference_embeddings
        ) == 0
    ):
        raise RuntimeError(
            "Router reference embeddings "
            "were not created."
        )

    query_embedding = (
        _get_embeddings(
            [message]
        )
    )

    dot_product = np.dot(
        query_embedding,
        _reference_embeddings.T,
    )

    query_norm = np.linalg.norm(
        query_embedding,
        axis=1,
        keepdims=True,
    )

    reference_norms = (
        np.linalg.norm(
            _reference_embeddings,
            axis=1,
            keepdims=True,
        ).T
    )

    denominator = np.maximum(
        query_norm
        * reference_norms,
        1e-12,
    )

    similarities = (
        dot_product
        / denominator
    )

    best_index = int(
        np.argmax(
            similarities[0]
        )
    )

    best_score = float(
        similarities[0][
            best_index
        ]
    )

    best_label = (
        _reference_labels[
            best_index
        ]
    )

    matched_example = (
        _reference_texts[
            best_index
        ]
    )

    # --------------------------------------------
    # Not sure / low-confidence behavior
    # --------------------------------------------

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


# ============================================================
# Public router entry point
# ============================================================

def route_user_input(
    message: str,
) -> dict:

    return route_message_few_shot(
        message
    )
