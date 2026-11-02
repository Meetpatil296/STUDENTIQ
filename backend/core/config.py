from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_model.pkl"
DATASET_PATH = PROJECT_ROOT / "dataset" / "student_performance.csv"

APP_NAME = "STUDENTIQ"
APP_VERSION = "1.0.0"
DEBUG = True
