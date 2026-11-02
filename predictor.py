"""
predictor.py
------------
All prediction logic used by the Streamlit app, kept separate from the user
interface so that it is easy to read, test and explain in a viva.

The saved model is a scikit-learn Pipeline (imputer + encoder + Random Forest),
so a new student record automatically goes through EXACTLY the same
preprocessing that was used during training.
"""
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

import config as cfg


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------
def load_model(path=cfg.MODEL_PATH):
    """Load the trained pipeline. Raises a clear error if it does not exist."""
    if not path.exists():
        raise FileNotFoundError(
            "Trained model not found. Please run:  python generate_dataset.py  "
            "and then  python train_model.py")
    return joblib.load(path)


def compute_reference(df: pd.DataFrame) -> Dict[str, dict]:
    """
    Reference values from the dataset: the 'typical student' (median / most common
    value) and the min / max seen in the data. Used for explanations and warnings.
    """
    ref = {}
    for col in cfg.NUMERIC_FEATURES:
        ref[col] = {"typical": float(df[col].median()),
                    "min": float(df[col].min()), "max": float(df[col].max())}
    for col in cfg.CATEGORICAL_FEATURES:
        ref[col] = {"typical": df[col].mode().iloc[0]}
    return ref


# -----------------------------------------------------------------------------
# Input handling
# -----------------------------------------------------------------------------
def validate_inputs(values: dict, unknown: List[str],
                    reference: Optional[dict] = None) -> Tuple[List[str], List[str]]:
    """
    Check the user's input. Returns (errors, warnings).
    Errors block the prediction; warnings are only shown to the user.
    """
    errors, warnings = [], []

    for col in cfg.NUMERIC_FEATURES:
        if col in unknown:
            continue
        label = cfg.FEATURE_LABELS[col]
        spec = cfg.FEATURE_SPEC[col]
        value = values.get(col)
        try:
            value = float(value)
        except (TypeError, ValueError):
            errors.append(f"{label}: please enter a valid number.")
            continue
        if np.isnan(value):
            errors.append(f"{label}: value is missing. Enter a number or mark it as unknown.")
        elif not spec["min"] <= value <= spec["max"]:
            errors.append(f"{label}: must be between {spec['min']:g} and {spec['max']:g}.")
        elif reference and not reference[col]["min"] <= value <= reference[col]["max"]:
            warnings.append(
                f"{label} = {value:g} is outside the range seen during training "
                f"({reference[col]['min']:g} - {reference[col]['max']:g}); the prediction may be less reliable.")

    for col in cfg.CATEGORICAL_FEATURES:
        if col not in unknown and values.get(col) not in cfg.CATEGORY_OPTIONS[col]:
            errors.append(f"{cfg.FEATURE_LABELS[col]}: choose Yes or No.")

    # Logical consistency check (a day only has 24 hours)
    study, sleep = values.get("Study_Hours_Per_Day"), values.get("Sleep_Hours")
    if not errors and "Study_Hours_Per_Day" not in unknown and "Sleep_Hours" not in unknown:
        if float(study) + float(sleep) > 24:
            errors.append("Study hours + sleep hours cannot be more than 24 hours per day.")
        elif float(study) + float(sleep) > 20:
            warnings.append("Study + sleep hours leave very little free time; please re-check the values.")

    if len(unknown) >= 4:
        warnings.append("Many fields are marked unknown; they are filled with typical values, "
                        "so the prediction will be less reliable.")
    return errors, warnings


def build_input_frame(values: dict, unknown: List[str]) -> pd.DataFrame:
    """One-row DataFrame in the exact column order used for training.
    Unknown values become NaN and are filled by the pipeline's imputers."""
    row = {}
    for col in cfg.NUMERIC_FEATURES:
        row[col] = np.nan if col in unknown else float(values[col])
    for col in cfg.CATEGORICAL_FEATURES:
        row[col] = np.nan if col in unknown else values[col]
    frame = pd.DataFrame([row], columns=cfg.ALL_FEATURES)
    for col in cfg.CATEGORICAL_FEATURES:
        frame[col] = frame[col].astype("object")
    return frame


# -----------------------------------------------------------------------------
# Prediction and explanation
# -----------------------------------------------------------------------------
def predict(model, frame: pd.DataFrame) -> Tuple[str, Dict[str, float]]:
    """Return the predicted category and the probability of every category."""
    proba = model.predict_proba(frame)[0]
    by_class = dict(zip(model.classes_, proba))
    ordered = {c: float(by_class.get(c, 0.0)) for c in cfg.CLASS_ORDER}
    label = str(model.predict(frame)[0])
    return label, ordered


def explain(model, frame: pd.DataFrame, reference: dict, label: str) -> pd.DataFrame:
    """
    Simple, model-based explanation ("what-if" test).

    For each feature we replace the student's value with the value of a TYPICAL
    student and measure how much the probability of the predicted category changes.
        influence > 0  -> this feature supports the prediction
        influence < 0  -> this feature pushes against the prediction
    """
    base = model.predict_proba(frame)[0][list(model.classes_).index(label)]
    rows = []
    for col in cfg.ALL_FEATURES:
        altered = frame.copy()
        altered[col] = altered[col].astype("object") if col in cfg.CATEGORICAL_FEATURES else altered[col]
        altered.at[0, col] = reference[col]["typical"]
        new = model.predict_proba(altered)[0][list(model.classes_).index(label)]
        shown = frame.at[0, col]
        rows.append({
            "Feature": cfg.FEATURE_LABELS[col],
            "Your value": "unknown (typical value used)" if pd.isna(shown) else
                          (f"{shown:g}" if col in cfg.NUMERIC_FEATURES else str(shown)),
            "Typical student": (f"{reference[col]['typical']:g}" if col in cfg.NUMERIC_FEATURES
                                else str(reference[col]["typical"])),
            "Influence (pp)": round(float(base - new) * 100, 1),
        })
    return pd.DataFrame(rows).sort_values("Influence (pp)", ascending=False).reset_index(drop=True)


ADVICE = {
    "Good": "The student is on a strong track. Keep up the consistent study routine and attendance, "
            "and consider peer mentoring or advanced material to stay challenged.",
    "Average": "The student is doing reasonably well but has room to improve. Focus on the factors "
               "marked below as pulling the prediction down: regular attendance, timely assignments "
               "and a steady daily study schedule can move the student into the Good category.",
    "Poor": "The student may need additional academic support. Recommended actions: a mentor or "
            "counsellor meeting, a structured study plan, remedial classes, closer attendance "
            "monitoring and regular feedback on assignments.",
}


def explanation_text(label: str, probs: Dict[str, float], influence: pd.DataFrame) -> str:
    """Convert the numbers into a short plain-English explanation."""
    top = influence[influence["Influence (pp)"] > 0.5].head(3)
    against = influence[influence["Influence (pp)"] < -0.5].tail(2)
    text = f"The Random Forest assigns **{probs[label] * 100:.1f}%** probability to **{label}**."
    if top.empty and against.empty:
        text += (" The student's values are close to those of a typical student, "
                 "so no single factor stands out.")
    if not top.empty:
        parts = [f"{r['Feature']} ({r['Your value']})" for _, r in top.iterrows()]
        text += " Factors that support this result: " + ", ".join(parts) + "."
    if not against.empty:
        parts = [f"{r['Feature']} ({r['Your value']})" for _, r in against.iterrows()]
        text += " Factors pulling against it: " + ", ".join(parts) + "."
    return text
