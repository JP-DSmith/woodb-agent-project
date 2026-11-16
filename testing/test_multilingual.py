import pytest

from services.translation_service import (
    TranslationService,
    UnsupportedLanguageError,
)


@pytest.mark.parametrize(
    ("detected", "expected"),
    [
        ("en", "en"),
        ("en-CA", "en"),
        ("fr", "fr-CA"),
        ("fr-CA", "fr-CA"),
        ("es", "es"),
        ("es-MX", "es"),
        ("zh", "zh-CN"),
        ("zh-CN", "zh-CN"),
        ("zh-Hans", "zh-CN"),
        ("pa", "pa"),
        ("ar", "ar"),
        ("hi", "hi"),
        ("fil", "fil"),
        ("tl", "fil"),
        ("yue", "yue"),
        ("it", "it"),
        ("de", "de"),
        ("th", "th"),
        ("vi", "vi"),
    ],
)
def test_supported_language_normalization(
    detected,
    expected,
):
    assert (
        TranslationService.normalize_language(detected)
        == expected
    )


@pytest.mark.parametrize(
    "language_code",
    [
        "pl",
        "ja",
        "ko",
        "pt",
        "ru",
    ],
)
def test_unsupported_languages_are_not_normalized(
    language_code,
):
    assert (
        TranslationService.normalize_language(
            language_code
        )
        is None
    )


def test_english_input_is_not_translated():
    class FakeLanguage:
        language_code = "en"

    class FakeDetectionResponse:
        languages = [FakeLanguage()]

    class FakeClient:
        def detect_language(self, request):
            return FakeDetectionResponse()

        def translate_text(self, request):
            raise AssertionError(
                "English input must not be translated."
            )

    service = TranslationService(
        project_id="test-project",
        client=FakeClient(),
    )

    context = service.prepare_input(
        "Does NB-0101 qualify under CUSMA?"
    )

    assert context.detected_language == "en"
    assert context.response_language == "en"
    assert context.translation_applied is False
    assert (
        context.processing_text
        == "Does NB-0101 qualify under CUSMA?"
    )


def test_supported_non_english_input_translates_to_english():
    class FakeLanguage:
        language_code = "de"

    class FakeDetectionResponse:
        languages = [FakeLanguage()]

    class FakeTranslation:
        translated_text = (
            "Does NB-0101 qualify under CUSMA?"
        )

    class FakeTranslationResponse:
        translations = [FakeTranslation()]

    class FakeClient:
        def detect_language(self, request):
            return FakeDetectionResponse()

        def translate_text(self, request):
            assert (
                request["target_language_code"]
                == "en"
            )

            return FakeTranslationResponse()

    service = TranslationService(
        project_id="test-project",
        client=FakeClient(),
    )

    context = service.prepare_input(
        "Erfüllt NB-0101 die Voraussetzungen "
        "des CUSMA?"
    )

    assert context.detected_language == "de"
    assert context.response_language == "de"
    assert context.translation_applied is True
    assert (
        context.processing_text
        == "Does NB-0101 qualify under CUSMA?"
    )


def test_unsupported_language_is_blocked():
    class FakeLanguage:
        language_code = "pl"

    class FakeDetectionResponse:
        languages = [FakeLanguage()]

    class FakeClient:
        def detect_language(self, request):
            return FakeDetectionResponse()

    service = TranslationService(
        project_id="test-project",
        client=FakeClient(),
    )

    with pytest.raises(
        UnsupportedLanguageError
    ) as error:
        service.prepare_input(
            "Czy NB-0101 kwalifikuje się "
            "w ramach CUSMA?"
        )

    assert error.value.detected_language == "pl"