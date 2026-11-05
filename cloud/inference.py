
from pathlib import Path
from functools import lru_cache
from io import BytesIO

from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "finetuned_800_best.pt"
)

IMAGE_SIZE = 800
CONFIDENCE_THRESHOLD = 0.25
DEVICE = "cpu"

EXPECTED_CLASSES = {
    0: "Impacted",
    1: "Caries",
    2: "Periapical Lesion",
    3: "Deep Caries",
}


@lru_cache(maxsize=1)
def get_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = YOLO(str(MODEL_PATH))

    if model.task != "detect":
        raise RuntimeError(
            f"Expected a detection model, got {model.task}"
        )

    actual_classes = {
        int(key): str(value).strip().casefold().replace("_", " ")
        for key, value in model.names.items()
    }
    expected_classes = {
        key: value.casefold() for key, value in EXPECTED_CLASSES.items()
    }

    if actual_classes != expected_classes:
        raise RuntimeError(
            f"Class mismatch! Expected {expected_classes}, "
            f"got {actual_classes}"
        )

    return model


def run_inference(image_bytes: bytes) -> dict:
    if not image_bytes:
        raise ValueError("Image data is empty.")

    try:
        with Image.open(BytesIO(image_bytes)) as uploaded_image:
            uploaded_image.verify()

        with Image.open(BytesIO(image_bytes)) as uploaded_image:
            image = uploaded_image.convert("RGB")

    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise ValueError("Uploaded file is not a valid or safe image.")

    model = get_model()

    results = model.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        device=DEVICE,
        verbose=False,
    )

    detections = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            coordinates = box.xyxy[0].tolist()

            detections.append({
                "class_id": class_id,
                "class_name": EXPECTED_CLASSES[class_id],
                "confidence": round(confidence, 4),
                "bbox_xyxy": [
                    round(float(value), 2)
                    for value in coordinates
                ],
            })

    return {
        "status": "completed" if detections else "no_detections",
        "model_version": MODEL_PATH.name,
        "model_architecture": "YOLO11m",
        "image_size": IMAGE_SIZE,
        "device": DEVICE,
        "image_width": image.width,
        "image_height": image.height,
        "detections": detections,
    }
