from typing import Any, Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(
        ...,
        description="Student features required by the trained model"
    )


class PredictionResponse(BaseModel):
    success: bool
    prediction: str
    confidence: Optional[float] = None
    probabilities: Optional[dict[str, float]] = None
