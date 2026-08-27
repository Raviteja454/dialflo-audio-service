import numpy as np

from app.audio.features import extract_features
from app.config import settings
from app.inference.base import AudioClassifier


class HeuristicAudioClassifier(AudioClassifier):
    """
    Temporary baseline classifier.

    This is only for validating the inference pipeline.
    It is not a production demographic classifier.
    """

    def predict(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> dict:

        features = extract_features(
            audio,
            sample_rate,
        )

        rms = features["rms"]
        centroid = features["spectral_centroid"]
        bandwidth = features["spectral_bandwidth"]

        if rms <= 0:
            return self._unknown()

        # Temporary heuristic only.
        if centroid < 180:
            gender_prediction = "unknown"
            gender_confidence = 0.0
        elif centroid < 260:
            gender_prediction = "female"
            gender_confidence = 0.61
        else:
            gender_prediction = "male"
            gender_confidence = 0.61

        if bandwidth < 150:
            age_prediction = "46-60"
            age_confidence = 0.60
        elif bandwidth < 250:
            age_prediction = "31-45"
            age_confidence = 0.60
        else:
            age_prediction = "18-30"
            age_confidence = 0.60

        return {
            "gender": self._apply_threshold(
                gender_prediction,
                gender_confidence,
            ),
            "age_bracket": self._apply_threshold(
                age_prediction,
                age_confidence,
            ),
        }

    @staticmethod
    def _apply_threshold(
        prediction: str,
        confidence: float,
    ) -> dict:
        if (
            prediction == "unknown"
            or confidence < settings.confidence_threshold
        ):
            return {
                "prediction": "unknown",
                "confidence": 0.0,
            }

        return {
            "prediction": prediction,
            "confidence": confidence,
        }

    @staticmethod
    def _unknown() -> dict:
        return {
            "gender": {
                "prediction": "unknown",
                "confidence": 0.0,
            },
            "age_bracket": {
                "prediction": "unknown",
                "confidence": 0.0,
            },
        }
