"""Cloud application entry point."""
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime, timezone

app = FastAPI(title="Dental Edge AI API")

class Prediction(BaseModel):
    image_id: str
    predicted_class: str
    confidence: float
    model_version: str

@app.get("/health")
def health():
    return {"status": "ok", "service": "dental-edge-ai"}

@app.post("/predictions")
def create_prediction(prediction: Prediction):
    return {
        "status": "received",
        "prediction": prediction.model_dump(),
        "received_at": datetime.now(timezone.utc).isoformat()
    }
