from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import traceback
import logging
from typing import Optional

# Import your agent wrapper
from create_formula import generate_formula

app = FastAPI(
    title="Cosmetics Formula Generator",
    version="0.1",
    description="Accepts a user request and returns a JSON cosmetic formula produced by the agent."
)

logger = logging.getLogger("uvicorn.error")


# -----------------------------
# Pydantic models
# -----------------------------
class GenerateRequest(BaseModel):
    message: str
    save_file: bool = False


class GenerateResponse(BaseModel):
    ok: bool
    result: Optional[dict] = None
    error: Optional[dict] = None


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/", tags=["health"])
def root():
    """
    Health check / root endpoint.
    """
    return {"status": "ok", "message": "Cosmetics microservice is running"}


@app.post("/generate", response_model=GenerateResponse, tags=["generation"])
def generate(req: GenerateRequest):
    """
    Generate a cosmetic formula using the agent wrapper.
    Example request body:
      { "message": "Lightweight daytime moisturizer for oily skin with SPF 30", "save_file": false }
    """
    try:
        result = generate_formula(req.message, save_file=req.save_file)
        return GenerateResponse(ok=True, result=result)
    except Exception as exc:
        tb = traceback.format_exc()
        logger.exception("Error while generating formula")
        return GenerateResponse(ok=False, error={"message": str(exc), "trace": tb})


# -----------------------------
# Run app locally
# -----------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
