"""Chat-service integration for the CUSMA Trade Assistant."""

import re

from agents.pipeline import answer as pipeline_answer


SKU_PATTERN = re.compile(
    r"\bNB-\d{4}(?:-V\d+)?\b",
    re.IGNORECASE,
)

OTHER_PRODUCT_ID_PATTERN = re.compile(
    r"\b[A-Z]+-\d{3,}(?:-V\d+)?\b",
    re.IGNORECASE,
)


def answer_question(
    message: str,
    *,
    original: str | None = None,
    language: str = "en",
    tier: str | None = None,
) -> dict:
    """Process one user question through the complete agent pipeline."""

    user_message = message.strip()

    if not user_message:
        return {
            "response": "Please enter a question.",
            "result": None,
            "router": None,
            "label": None,
        }

    # Preserve the established behavior for product IDs that are
    # clearly outside the supported NB-series test catalogue.
    product_match = OTHER_PRODUCT_ID_PATTERN.search(
        user_message
    )

    nb_match = SKU_PATTERN.search(
        user_message
    )

    if product_match and not nb_match:
        return {
            "response": (
                "I can currently evaluate CUSMA qualification "
                "for products in the NB-series test world. "
                "Please include a SKU such as NB-0101, "
                "NB-0413, or NB-0414 in your question."
            ),
            "result": None,
            "router": None,
            "label": None,
        }

    return pipeline_answer(
        user_message,
        tier,
        original=original,
        language=language,
    )
