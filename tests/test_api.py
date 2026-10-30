
from fastapi.testclient import TestClient

from cloud import database
from cloud.main import app


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
