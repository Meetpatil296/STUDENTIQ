import joblib
import pandas as pd

from backend.core.config import MODEL_PATH

_model = None


def get_model():
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found: {MODEL_PATH}"
            )

        _model = joblib.load(MODEL_PATH)

    return _model


def get_model_features():
    model = get_model()

    if not hasattr(model, "feature_names_in_"):
        raise ValueError(
            "Model does not expose feature names. "
            "Check the original training pipeline."
        )

    return list(model.feature_names_in_)


def predict(features: dict):
    model = get_model()
    expected = get_model_features()

    missing = [name for name in expected if name not in features]
    extra = [name for name in features if name not in expected]

    if missing:
        raise ValueError(f"Missing features: {missing}")

    if extra:
        raise ValueError(f"Unexpected features: {extra}")

    data = pd.DataFrame(
        [[features[name] for name in expected]],
        columns=expected
    )

    prediction = str(model.predict(data)[0])

    confidence = None
    probabilities = None

    if hasattr(model, "predict_proba"):
        values = model.predict_proba(data)[0]
        classes = model.classes_

        confidence = round(float(max(values)) * 100, 2)
        probabilities = {
            str(label): round(float(value) * 100, 2)
            for label, value in zip(classes, values)
        }

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities
    }
