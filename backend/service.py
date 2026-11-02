from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "random_forest_model.pkl"
DATASET_PATH = BASE_DIR / "dataset" / "student_performance.csv"

_model = None


def load_model():
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}"
            )

        _model = joblib.load(MODEL_PATH)

    return _model


def get_expected_features(model):
    # Prefer the exact feature names stored by the trained model.
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    # Fall back to dataset columns if model feature names are unavailable.
    if DATASET_PATH.exists():
        df = pd.read_csv(DATASET_PATH)

        ignored = {
            "student id",
            "student_id",
            "performance category",
            "performance_category",
            "performance",
            "target",
            "label"
        }

        return [
            column for column in df.columns
            if column.strip().lower() not in ignored
        ]

    raise ValueError(
        "Cannot identify model input columns. "
        "Check your model and dataset."
    )


def predict_student(features: dict):
    model = load_model()
    expected_features = get_expected_features(model)

    missing = [
        name for name in expected_features
        if name not in features
    ]

    if missing:
        raise ValueError(
            f"Missing input features: {missing}"
        )

    input_data = pd.DataFrame(
        [[features[name] for name in expected_features]],
        columns=expected_features
    )

    prediction = model.predict(input_data)[0]

    result = {
        "predicted_category": str(prediction)
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        classes = model.classes_

        result["confidence"] = round(
            float(max(probabilities)) * 100, 2
        )

        result["probabilities"] = {
            str(label): round(float(probability) * 100, 2)
            for label, probability in zip(classes, probabilities)
        }

    return result


def get_model_info():
    model = load_model()

    return {
        "model_name": "Random Forest Classifier",
        "model_file": str(MODEL_PATH.name),
        "input_features": get_expected_features(model)
    }
