
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from cloud import database
from cloud.schemas import Prediction


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()
    yield


app = FastAPI(
    title="Dental Edge AI API",
    description="Research prototype for dental image analysis.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "service": "dental-edge-ai"}


@app.post("/predictions", status_code=201)
def create_prediction(prediction: Prediction):
    received_at = datetime.now(timezone.utc).isoformat()
    prediction_data = prediction.model_dump()

    record_id = database.save_prediction(
        prediction_data,
        created_at=received_at,
    )

    return {
        "id": record_id,
        "status": "saved",
        "prediction": prediction_data,
        "created_at": received_at,
    }
