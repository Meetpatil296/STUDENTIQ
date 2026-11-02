"""
common.py
Shared constants and helper functions used by generate_dataset.py,
train_model.py, app.py and the analysis notebook.
Keeping them in one place guarantees that training and prediction
use exactly the same column names, ranges and cleaning rules.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Paths (relative to this file, so the project works on any computer)
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "student_performance.csv"
MODEL_PATH = BASE_DIR / "models" / "random_forest_model.pkl"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
REPORT_DIR = BASE_DIR / "reports"

# ----------------------------------------------------------------------
# Column definitions
# ----------------------------------------------------------------------
ID_COL = "Student_ID"
TARGET = "Performance_Category"
CLASS_ORDER = ["Poor", "Average", "Good"]          # low -> high

NUMERIC_FEATURES = [
    "Study_Hours_Per_Day",
    "Attendance_Percentage",
    "Previous_Exam_Marks",
    "Assignment_Scores",
    "Internal_Assessment_Marks",
    "Sleep_Hours",
]
CATEGORICAL_FEATURES = ["Extracurricular_Activity", "Internet_Availability"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Human readable names for the dashboard
LABELS = {
    "Study_Hours_Per_Day": "Study hours per day",
    "Attendance_Percentage": "Attendance (%)",
    "Previous_Exam_Marks": "Previous exam marks",
    "Assignment_Scores": "Assignment scores",
    "Internal_Assessment_Marks": "Internal assessment marks",
    "Sleep_Hours": "Sleep hours",
    "Extracurricular_Activity": "Extracurricular activity",
    "Internet_Availability": "Internet availability",
}

# Allowed (min, max) for each numeric input - used for validation
VALID_RANGES = {
    "Study_Hours_Per_Day": (0.0, 14.0),
    "Attendance_Percentage": (0.0, 100.0),
    "Previous_Exam_Marks": (0.0, 100.0),
    "Assignment_Scores": (0.0, 100.0),
    "Internal_Assessment_Marks": (0.0, 100.0),
    "Sleep_Hours": (2.0, 14.0),
}
VALID_CHOICES = {
    "Extracurricular_Activity": ["Yes", "No"],
    "Internet_Availability": ["Yes", "No"],
}


# ----------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------
def load_dataset(path=DATA_PATH):
    """Read the raw CSV file. Raises FileNotFoundError with a clear hint."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Run 'python generate_dataset.py' first."
        )
    return pd.read_csv(path)


def clean_dataset(df):
    """
    Row-level cleaning that is safe to do BEFORE the train/test split:
      * remove exact duplicate records
      * drop rows whose target label is missing (cannot learn from them)
      * make sure numeric columns really are numeric
    NOTE: filling missing FEATURE values is deliberately NOT done here.
    It is done inside the sklearn Pipeline (fitted on training data only)
    so that no information from the test set leaks into training.
    """
    df = df.drop_duplicates().copy()
    df = df.dropna(subset=[TARGET])
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def load_metrics(path=METRICS_PATH):
    """Load the evaluation results saved by train_model.py."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Metrics file not found at '{path}'. Run 'python train_model.py' first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(path=MODEL_PATH):
    """Load the saved preprocessing + Random Forest pipeline."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at '{path}'. Run 'python train_model.py' first."
        )
    return joblib.load(path)


# ----------------------------------------------------------------------
# Prediction helpers
# ----------------------------------------------------------------------
def validate_input(values, allow_missing=False):
    """
    Check a dictionary of user inputs. Returns a list of error messages
    (an empty list means the input is valid).
    """
    errors = []
    for col, (low, high) in VALID_RANGES.items():
        v = values.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            if not allow_missing:
                errors.append(f"{LABELS[col]} is required.")
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            errors.append(f"{LABELS[col]} must be a number.")
            continue
        if not (low <= v <= high):
            errors.append(f"{LABELS[col]} must be between {low:g} and {high:g}.")
    for col, choices in VALID_CHOICES.items():
        v = values.get(col)
        if v is None:
            if not allow_missing:
                errors.append(f"{LABELS[col]} is required.")
        elif v not in choices:
            errors.append(f"{LABELS[col]} must be one of {choices}.")
    if not errors:
        study, sleep = values.get("Study_Hours_Per_Day"), values.get("Sleep_Hours")
        if study is not None and sleep is not None and study + sleep > 24:
            errors.append("Study hours + sleep hours cannot exceed 24 hours a day.")
    return errors


def predict_student(model, values):
    """
    Run ONE student through the saved pipeline (imputation + encoding +
    Random Forest). Returns (predicted_label, {class: probability}).
    Missing values (None) are filled by the pipeline's imputers.
    """
    row = pd.DataFrame([{col: values.get(col, np.nan) for col in FEATURES}])
    for col in NUMERIC_FEATURES:
        row[col] = pd.to_numeric(row[col], errors="coerce")
    label = model.predict(row)[0]
    proba = model.predict_proba(row)[0]
    probs = {cls: float(p) for cls, p in zip(model.classes_, proba)}
    return label, probs


def explain_prediction(values, df, importances, top_n=3):
    """
    Build a simple, human readable explanation. Each numeric input is
    compared with the typical (median) student in the dataset.
    This does NOT decide the prediction - it only describes the input.
    """
    strengths, concerns = [], []
    for col in NUMERIC_FEATURES:
        v = values.get(col)
        if v is None:
            continue
        median, std = df[col].median(), df[col].std()
        if std == 0 or np.isnan(std):
            continue
        z = (float(v) - median) / std
        text = f"{LABELS[col]}: {float(v):g} (typical student: {median:.1f})"
        if z >= 0.5:
            strengths.append(text)
        elif z <= -0.5:
            concerns.append(text)
    top_features = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return {"strengths": strengths, "concerns": concerns, "top_features": top_features}
