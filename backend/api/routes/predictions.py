from fastapi import APIRouter, HTTPException

from backend.schemas.prediction import (
    PredictionRequest,
    PredictionResponse
)
from backend.services.model_service import predict

router = APIRouter(prefix="/api/v1", tags=["Predictions"])


@router.post("/predict", response_model=PredictionResponse)
def predict_performance(request: PredictionRequest):
    try:
        result = predict(request.features)

        return {
            "success": True,
            **result
        }

    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
