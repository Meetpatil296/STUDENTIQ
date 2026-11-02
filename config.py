"""
config.py
---------
Central configuration shared by generate_dataset.py, train_model.py,
predictor.py, app.py and the notebook.

Keeping column names, valid ranges and file paths in ONE place guarantees
that the training code and the web application always agree with each other.
All paths are relative to the project folder, so the project runs on any
computer (Windows / macOS / Linux) without editing.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# File paths (relative to this file, so the project is fully portable)
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "student_performance.csv"
MODEL_PATH = BASE_DIR / "models" / "random_forest_model.pkl"
METRICS_PATH = BASE_DIR / "models" / "model_metrics.json"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

# ----------------------------------------------------------------------------
# Reproducibility / training settings
# ----------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20          # 80 % training, 20 % testing
CV_FOLDS = 5

# ----------------------------------------------------------------------------
# Column names
# ----------------------------------------------------------------------------
ID_COL = "Student_ID"
TARGET = "Performance_Category"

NUMERIC_FEATURES = [
    "Study_Hours_Per_Day",
    "Attendance_Percentage",
    "Previous_Exam_Marks",
    "Assignment_Scores",
    "Internal_Assessment_Marks",
    "Sleep_Hours",
]
CATEGORICAL_FEATURES = [
    "Extracurricular_Activity",
    "Internet_Availability",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Order used everywhere for display (best -> worst)
CLASS_ORDER = ["Good", "Average", "Poor"]
PALETTE = {"Good": "#2E9E5B", "Average": "#F2A93B", "Poor": "#D9534F"}

# ----------------------------------------------------------------------------
# Valid ranges used for (a) data cleaning, (b) input validation in the app
# ----------------------------------------------------------------------------
FEATURE_SPEC = {
    "Study_Hours_Per_Day": dict(
        label="Study Hours Per Day", min=0.0, max=12.0, default=4.0, step=0.5,
        unit="hrs", help="Average hours the student studies each day (0 - 12)."),
    "Attendance_Percentage": dict(
        label="Attendance Percentage", min=0.0, max=100.0, default=80.0, step=1.0,
        unit="%", help="Percentage of classes attended (0 - 100)."),
    "Previous_Exam_Marks": dict(
        label="Previous Examination Marks", min=0.0, max=100.0, default=60.0, step=1.0,
        unit="/100", help="Marks obtained in the previous examination, out of 100."),
    "Assignment_Scores": dict(
        label="Assignment Scores", min=0.0, max=100.0, default=65.0, step=1.0,
        unit="/100", help="Average assignment score, out of 100."),
    "Internal_Assessment_Marks": dict(
        label="Internal Assessment Marks", min=0.0, max=50.0, default=30.0, step=1.0,
        unit="/50", help="Internal assessment marks, out of 50."),
    "Sleep_Hours": dict(
        label="Sleep Hours", min=3.0, max=12.0, default=7.0, step=0.5,
        unit="hrs", help="Average hours of sleep per night (3 - 12)."),
}
CATEGORY_OPTIONS = {
    "Extracurricular_Activity": ["Yes", "No"],
    "Internet_Availability": ["Yes", "No"],
}
FEATURE_LABELS = {
    **{k: v["label"] for k, v in FEATURE_SPEC.items()},
    "Extracurricular_Activity": "Extracurricular Activity",
    "Internet_Availability": "Internet Availability",
}

COLUMN_DESCRIPTIONS = [
    (ID_COL, "Text (identifier)", "Unique student identifier, e.g. STU0001 (not used for learning)"),
    ("Study_Hours_Per_Day", "Float", "Average daily self-study hours (0.5 - 10)"),
    ("Attendance_Percentage", "Float", "Percentage of classes attended (40 - 100)"),
    ("Previous_Exam_Marks", "Float", "Marks in the previous examination out of 100"),
    ("Assignment_Scores", "Float", "Average assignment score out of 100"),
    ("Internal_Assessment_Marks", "Float", "Internal assessment marks out of 50"),
    ("Sleep_Hours", "Float", "Average sleep hours per night (4 - 10)"),
    ("Extracurricular_Activity", "Categorical (Yes/No)", "Whether the student takes part in extracurricular activities"),
    ("Internet_Availability", "Categorical (Yes/No)", "Whether the student has internet access at home"),
    (TARGET, "Categorical (target)", "Good / Average / Poor - the value the model predicts"),
]
