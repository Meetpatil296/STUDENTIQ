from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.service import (
    load_model,
    predict_student,
    get_model_info
)

app = FastAPI(
    title="STUDENTIQ API",
    description=(
        "Student Performance Prediction System "
        "using Random Forest Classifier"
    ),
    version="1.0.0"
)


class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(
        ...,
        description="Student details using the model's exact feature names"
    )


@app.get("/")
def home():
    return {
        "project": "STUDENTIQ",
        "message": "Student Performance Prediction API is running",
        "algorithm": "Random Forest Classifier",
        "documentation": "/docs"
    }


@app.get("/health")
def health():
    try:
        load_model()
        return {
            "status": "healthy",
            "model_status": "loaded"
        }
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=str(error)
        )


@app.get("/model-info")
def model_info():
    try:
        return get_model_info()
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        result = predict_student(request.features)

        return {
            "success": True,
            "project": "STUDENTIQ",
            "result": result
        }

    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
