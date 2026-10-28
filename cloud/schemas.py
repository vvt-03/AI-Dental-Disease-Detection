"""Request and response schemas for the cloud API."""

from pydantic import BaseModel, Field


class Prediction(BaseModel):
    image_id: str = Field(min_length=1, max_length=200)
    predicted_class: str = Field(min_length=1, max_length=100)
    confidence: float = Field(ge=0, le=1)
    model_version: str = Field(min_length=1, max_length=50)
