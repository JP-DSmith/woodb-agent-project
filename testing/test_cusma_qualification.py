from prototypes.cusma_qualification.Engine.rules_engine import check_origin
from web.chat_services import answer_question


def test_nb0101_qualifies():
    """NB-0101 should qualify under the current deterministic rules."""
    result = check_origin("NB-0101")

    assert result["ok"] is True
    assert result["data"]["verdict"] == "qualifies"
    assert result["data"]["hs_code"] == "6401.99.10"
    assert result["data"]["tariff_shift"]["passed"] is True
    assert result["data"]["rvc"]["value"] == 0.8164


def test_nb0413_requires_human_review():
    """NB-0413 should escalate because its RVC is close to the threshold."""
    result = check_origin("NB-0413")

    assert result["ok"] is True
    assert result["data"]["verdict"] == "needs_review"
    assert result["data"]["tariff_shift"]["passed"] is True
    assert result["data"]["rvc"]["value"] == 0.5512
    assert result["data"]["escalate_reason"] is not None


def test_nb0414_does_not_qualify():
    """NB-0414 should fail qualification because tariff shift fails."""
    result = check_origin("NB-0414")

    assert result["ok"] is True
    assert result["data"]["verdict"] == "does_not_qualify"
    assert result["data"]["rvc"]["value"] == 0.574
    assert result["data"]["tariff_shift"]["passed"] is False


def test_unsupported_boot_does_not_create_qualification_result():
    """Unsupported product IDs must not receive a fabricated determination."""
    answer = answer_question(
        "Does BOOT-001 qualify under CUSMA?"
    )

    assert answer["result"] is None
    assert "NB-series" in answer["response"]