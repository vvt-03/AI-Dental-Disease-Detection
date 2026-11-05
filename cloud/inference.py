
def run_inference(image_bytes: bytes) -> dict:
    """
    Placeholder for the trained dental detection model.

    Later, this function will preprocess the image, run model inference,
    and return actual detections. It must not invent predictions.
    """
    if not image_bytes:
        raise ValueError("Image data is empty.")

    return {
        "status": "model_not_connected",
        "message": (
            "Image received successfully. "
            "Waiting for the trained model to be integrated."
        ),
        "detections": [],
        "model_version": None,
    }
