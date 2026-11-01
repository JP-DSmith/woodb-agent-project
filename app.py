from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(
    title="CUSMA Boots Agent",
    description="Development environment for the CUSMA conversational agent",
    version="0.1.1",
)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "environment": "dev"
    }


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>CUSMA Boots Agent</title>
        </head>
        <body style="
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
        ">
            <h1>CUSMA Boots Agent</h1>

            <h2>Development Environment</h2>

            <p>
                The CUSMA conversational agent development
                environment is running successfully.
            </p>

            <p>
                Agent functionality will be added as the team's
                components are integrated.
            </p>

            <hr>

            <p>
                <strong>Status:</strong> Online
            </p>

            <p>
                <strong>Environment:</strong> DEV
            </p>

            <p>
                <strong>Version:</strong> 0.1.1
            </p>
        </body>
    </html>
    """
