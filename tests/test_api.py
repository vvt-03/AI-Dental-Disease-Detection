
from fastapi.testclient import TestClient

from cloud import database
from cloud.main import app
from io import BytesIO
from PIL import Image

def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_prediction_is_saved(tmp_path, monkeypatch):
    # Use a temporary database so tests don't modify your real database.
    test_db = tmp_path / "test_predictions.db"
    monkeypatch.setattr(database, "DB_PATH", test_db)

    payload = {
        "image_id": "test-image-001",
        "predicted_class": "caries",
        "confidence": 0.91,
        "model_version": "test-v1",
    }

    with TestClient(app) as client:
        response = client.post("/predictions", json=payload)

    assert response.status_code == 201
    result = response.json()
    assert result["status"] == "saved"
    assert result["prediction"] == payload
    assert result["id"] == 1

    with database.get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM predictions WHERE id = ?",
            (result["id"],),
        ).fetchone()

    assert row is not None
    assert row["image_id"] == "test-image-001"
    assert row["confidence"] == 0.91


def test_invalid_confidence_is_rejected():
    payload = {
        "image_id": "test-image-002",
        "predicted_class": "caries",
        "confidence": 1.5,
        "model_version": "test-v1",
    }

    with TestClient(app) as client:
        response = client.post("/predictions", json=payload)

    assert response.status_code == 422


def test_analyze_rejects_unsupported_file_type():
    with TestClient(app) as client:
        response = client.post(
            "/analyze",
            files={
                "file": (
                    "notes.txt",
                    b"not an image",
                    "text/plain",
                )
            },
        )

    assert response.status_code == 415


def test_analyze_image_with_mocked_inference(monkeypatch):
    # Mock inference so the test doesn't need the actual model.
    def fake_inference(image_bytes):
        return {
            "status": "completed",
            "model_version": "test-model",
            "model_architecture": "YOLO11m",
            "image_size": 800,
            "device": "cpu",
            "image_width": 100,
            "image_height": 100,
            "detections": [
                {
                    "class_id": 1,
                    "class_name": "Caries",
                    "confidence": 0.95,
                    "bbox_xyxy": [10, 10, 50, 50],
                }
            ],
        }

    monkeypatch.setattr("cloud.main.run_inference", fake_inference)

    # Create a real PNG image for the upload.
    image = Image.new("RGB", (100, 100), color="white")
    image_buffer = BytesIO()
    image.save(image_buffer, format="PNG")
    image_buffer.seek(0)

    with TestClient(app) as client:
        response = client.post(
            "/analyze",
            files={
                "file": (
                    "sample.png",
                    image_buffer.getvalue(),
                    "image/png",
                )
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert len(data["detections"]) == 1
    assert data["detections"][0]["class_name"] == "Caries"