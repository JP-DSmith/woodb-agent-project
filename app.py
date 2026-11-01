from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from web.chat_services import answer_question


app = FastAPI(
    title="CUSMA Boots Agent",
    description="Development environment for the CUSMA conversational agent",
    version="0.3.0",
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    environment: str
    version: str


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "environment": "dev",
        "version": "0.3.0",
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    return ChatResponse(
        response=answer_question(request.message),
        environment="dev",
        version="0.3.0",
    )


@app.get("/", response_class=HTMLResponse)
def home():
    with open("web/index.html", encoding="utf-8") as file:
        return file.read()