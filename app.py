from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from web.chat_services import answer_question

from services.translation_service import (
    TranslationService,
    UnsupportedLanguageError,
)


app = FastAPI(
    title="CUSMA Trade Assistant",
    description=(
        "Development environment for the "
        "CUSMA conversational agent"
    ),
    version="0.6.0",
)


class ChatRequest(BaseModel):
    message: str
    preferred_language: str | None = None


class QualificationResult(BaseModel):
    sku: str
    verdict: str
    hs_code: str | None
    rvc: float
    rvc_threshold: float
    tariff_shift_passed: bool
    review_reason: str | None
    rules_version: str


class ChatResponse(BaseModel):
    response: str
    environment: str
    version: str
    result: QualificationResult | None = None
    detected_language: str | None = None
    response_language: str = "en"
    translation_applied: bool = False


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "environment": "dev",
        "version": "0.6.0",
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    translation_service = TranslationService()

    try:
        translation = (
            translation_service.prepare_input(
                request.message
            )
        )

    except UnsupportedLanguageError as error:
        return ChatResponse(
            response=str(error),
            environment="dev",
            version="0.6.0",
            result=None,
            detected_language=(
                error.detected_language
            ),
            response_language="en",
            translation_applied=False,
        )

    # By default, respond in the language detected
    # from the user's question.
    response_language = (
        translation.response_language
        or "en"
    )

    # If the user explicitly selected a supported
    # response language, use that for presentation.
    #
    # This does not modify the English processing
    # text or the deterministic CUSMA result.
    if request.preferred_language:
        preferred_language = (
            translation_service.normalize_language(
                request.preferred_language
            )
        )

        if preferred_language is None:
            return ChatResponse(
                response=(
                    "The selected response language "
                    "is not currently supported."
                ),
                environment="dev",
                version="0.6.0",
                result=None,
                detected_language=(
                    translation.detected_language
                ),
                response_language="en",
                translation_applied=False,
            )

        response_language = (
            preferred_language
        )

    # The deterministic CUSMA engine always receives
    # English processing text.
    answer = answer_question(
        translation.processing_text
    )

    # Only the conversational explanation may be
    # translated. answer["result"] is never translated
    # or modified.
    response_text = answer["response"]

    translation_applied = (
        response_language != "en"
    )

    if translation_applied:
        response_text = (
            translation_service.translate_response(
                text=response_text,
                response_language=(
                    response_language
                ),
            )
        )

    return ChatResponse(
        response=response_text,
        environment="dev",
        version="0.6.0",
        result=answer["result"],
        detected_language=(
            translation.detected_language
        ),
        response_language=(
            response_language
        ),
        translation_applied=(
            translation_applied
        ),
    )


@app.get("/", response_class=HTMLResponse)
def home():
    with open(
        "web/index.html",
        encoding="utf-8",
    ) as file:
        return file.read()