import os
from dataclasses import dataclass

from google.cloud import translate_v3


SUPPORTED_LANGUAGES = {
    "en": "English",
    "fr-CA": "French (Canada)",
    "es": "Spanish",
    "zh-CN": "Mandarin Chinese",
    "pa": "Punjabi",
    "ar": "Arabic",
    "hi": "Hindi",
    "fil": "Filipino / Tagalog",
    "yue": "Cantonese",
    "it": "Italian",
    "de": "German",
    "th": "Thai",
    "vi": "Vietnamese",
}


LANGUAGE_VARIANTS = {
    # English
    "en": "en",
    "en-CA": "en",
    "en-US": "en",
    "en-GB": "en",
    "en-AU": "en",

    # French
    "fr": "fr-CA",
    "fr-CA": "fr-CA",

    # Spanish
    "es": "es",
    "es-419": "es",
    "es-MX": "es",
    "es-US": "es",
    "es-ES": "es",

    # Mandarin Chinese
    "zh": "zh-CN",
    "zh-CN": "zh-CN",
    "zh-Hans": "zh-CN",

    # Punjabi
    "pa": "pa",

    # Arabic
    "ar": "ar",

    # Hindi
    "hi": "hi",

    # Filipino / Tagalog
    "fil": "fil",
    "tl": "fil",

    # Cantonese
    "yue": "yue",

    # Italian
    "it": "it",

    # German
    "de": "de",

    # Thai
    "th": "th",

    # Vietnamese
    "vi": "vi",
}


@dataclass(frozen=True)
class TranslationContext:
    original_text: str
    detected_language: str
    response_language: str | None
    processing_text: str
    translation_applied: bool
    supported: bool


class UnsupportedLanguageError(Exception):
    """Raised when the portal does not support a detected language."""

    def __init__(self, detected_language):
        self.detected_language = detected_language

        supported_names = ", ".join(
            SUPPORTED_LANGUAGES.values()
        )

        super().__init__(
            "That language is not currently supported. "
            "Please submit your question in one of the "
            f"following languages: {supported_names}."
        )


class TranslationService:
    """Translate portal communication around the CUSMA engine."""

    def __init__(
        self,
        project_id=None,
        client=None,
    ):
        self.project_id = (
            project_id
            or os.getenv("GOOGLE_CLOUD_PROJECT")
        )

        if not self.project_id:
            raise RuntimeError(
                "GOOGLE_CLOUD_PROJECT is not configured."
            )

        self.parent = (
            f"projects/{self.project_id}/locations/global"
        )

        self.client = (
            client
            or translate_v3.TranslationServiceClient()
        )

    @staticmethod
    def normalize_language(language_code):
        """Map detected language variants to portal languages."""

        if not language_code:
            return None

        if language_code in LANGUAGE_VARIANTS:
            return LANGUAGE_VARIANTS[language_code]

        base_code = language_code.split("-")[0]

        if base_code in LANGUAGE_VARIANTS:
            return LANGUAGE_VARIANTS[base_code]

        return None

    def detect_language(self, text):
        """Detect the language of user-provided text."""

        response = self.client.detect_language(
            request={
                "parent": self.parent,
                "content": text,
                "mime_type": "text/plain",
            }
        )

        if not response.languages:
            return None

        return response.languages[0].language_code

    def translate(
        self,
        text,
        target_language,
        source_language=None,
    ):
        """Translate text without altering structured CUSMA data."""

        if not text:
            return text

        request = {
            "parent": self.parent,
            "contents": [text],
            "mime_type": "text/plain",
            "target_language_code": target_language,
        }

        if source_language:
            request["source_language_code"] = (
                source_language
            )

        response = self.client.translate_text(
            request=request
        )

        if not response.translations:
            return text

        return response.translations[0].translated_text

    def prepare_input(self, text):
        """
        Detect the input language and convert supported
        non-English input to English for internal processing.
        """

        detected_language = self.detect_language(text)

        normalized_language = self.normalize_language(
            detected_language
        )

        if normalized_language is None:
            raise UnsupportedLanguageError(
                detected_language
            )

        if normalized_language == "en":
            return TranslationContext(
                original_text=text,
                detected_language=detected_language,
                response_language="en",
                processing_text=text,
                translation_applied=False,
                supported=True,
            )

        processing_text = self.translate(
            text=text,
            target_language="en",
            source_language=detected_language,
        )

        return TranslationContext(
            original_text=text,
            detected_language=detected_language,
            response_language=normalized_language,
            processing_text=processing_text,
            translation_applied=True,
            supported=True,
        )

    def translate_response(
        self,
        text,
        response_language,
    ):
        """
        Translate an English conversational response back
        to the user's supported language.
        """

        if response_language == "en":
            return text

        return self.translate(
            text=text,
            target_language=response_language,
            source_language="en",
        )


def supported_language_names():
    """Return the portal's approved language names."""

    return list(SUPPORTED_LANGUAGES.values())