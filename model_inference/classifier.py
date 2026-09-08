"""
Layer 3 - Model Inference.

Wraps the trained cloud classifier (a scikit-learn RandomForestClassifier).
The API layer never touches joblib/sklearn directly - it goes through this
class, so the model implementation could later be swapped (e.g. for a
gradient-boosted tree) without touching the rest of the pipeline.
"""
import joblib
import numpy as np

from feature_extraction import FEATURE_NAMES


class PhishingClassifier:
    def __init__(self, model_path: str):
        self._model = joblib.load(model_path)
        # feature_importances_ drives the response layer's plain-English
        # explanation, so it's computed once here rather than per-request.
        self.feature_importance = dict(zip(FEATURE_NAMES, self._model.feature_importances_.tolist()))

    def predict(self, feature_vector: list[float]) -> float:
        """Returns the probability (0-1) that the URL is phishing."""
        x = np.array(feature_vector, dtype=float).reshape(1, -1)
        return float(self._model.predict_proba(x)[0][1])
