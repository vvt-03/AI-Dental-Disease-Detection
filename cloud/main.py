
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from cloud import database
from cloud.schemas import Prediction
from uuid import uuid4
from fastapi import UploadFile, File, HTTPException
from cloud.inference import run_inference


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
    

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    allowed_types = {"image/jpeg", "image/png"}

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail="Please upload a JPEG or PNG image.",
        )

    image_bytes = await file.read(MAX_IMAGE_BYTES + 1)

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Image is empty.")

    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB upload limit.",
        )

    image_id = str(uuid4())

    result = run_inference(image_bytes)

    return {
        "image_id": image_id,
        "filename": file.filename,
        "status": result["status"],
        "inference": result,
    }
    
