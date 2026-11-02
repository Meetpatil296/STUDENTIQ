from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import APP_NAME, APP_VERSION
from backend.api.routes.predictions import router as prediction_router

app = FastAPI(
    title=f"{APP_NAME} Backend API",
    description="Student Performance Prediction using Random Forest",
    version=APP_VERSION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(prediction_router)


@app.get("/")
def home():
    return {
        "project": APP_NAME,
        "message": "Backend API is running",
        "version": APP_VERSION
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "STUDENTIQ Backend"
    }
