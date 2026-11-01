import re

from prototypes.nerissa.Engine.rules_engine import check_origin


SKU_PATTERN = re.compile(
    r"\bNB-\d{4}(?:-V\d+)?\b",
    re.IGNORECASE,
)


def answer_question(message: str) -> str:
    """Answer supported CUSMA qualification questions.

    The conversational layer does not calculate origin qualification.
    It identifies the SKU and delegates the determination to the
    deterministic rules engine.
    """

    user_message = message.strip()

    if not user_message:
        return "Please enter a question."

    sku_match = SKU_PATTERN.search(user_message)

    if not sku_match:
        return (
            "I can currently evaluate CUSMA qualification for products "
            "in the NB-series test world. Please include a SKU such as "
            "NB-0101, NB-0413, or NB-0414 in your question."
        )

    sku = sku_match.group(0).upper()

    result = check_origin(sku)

    if not result["ok"]:
        error = result["error"]

        return (
            f"I could not evaluate {sku}. "
            f"{error['message']}"
        )

    data = result["data"]

    verdict = data["verdict"]
    rvc = data["rvc"]["value"]
    threshold = data["rvc"]["threshold"]
    tariff_shift_passed = data["tariff_shift"]["passed"]
    hs_code = data["hs_code"]
    rules_version = result["rules_version"]

    if verdict == "qualifies":
        introduction = (
            f"{sku} qualifies under the current CUSMA rules."
        )

    elif verdict == "needs_review":
        introduction = (
            f"{sku} requires human review before CUSMA "
            "qualification can be confirmed."
        )

    else:
        introduction = (
            f"{sku} does not qualify under the current CUSMA rules."
        )

    response_parts = [
        introduction,
        f"HS code: {hs_code or 'Needs review'}.",
        f"Regional value content: {rvc:.1%}.",
        f"Required RVC: {threshold:.0%}.",
        (
            "Tariff shift: Passed."
            if tariff_shift_passed
            else "Tariff shift: Failed."
        ),
    ]

    if data["escalate_reason"]:
        response_parts.append(
            f"Review reason: {data['escalate_reason']}."
        )

    response_parts.append(
        f"Rules version: {rules_version}."
    )

    return " ".join(response_parts)