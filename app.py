from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from web.chat_services import answer_question


app = FastAPI(
    title="CUSMA Boots Agent",
    description="Development environment for the CUSMA conversational agent",
    version="0.4.0",
)


class ChatRequest(BaseModel):
    message: str

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


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "environment": "dev",
        "version": "0.4.0",
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    answer = answer_question(request.message)

    return ChatResponse(
        response=answer["response"],
        environment="dev",
        version="0.4.0",
        result=answer["result"],
    )


@app.get("/", response_class=HTMLResponse)
def home():
    with open("web/index.html", encoding="utf-8") as file:
        return file.read()