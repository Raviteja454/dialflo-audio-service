import numpy as np

from app.inference.classifier import HeuristicAudioClassifier


def create_audio(
    frequency: float = 220,
    amplitude: float = 0.15,
    duration: float = 2.0,
    sample_rate: int = 16000,
) -> tuple[np.ndarray, int]:

    t = np.linspace(
        0,
        duration,
        int(sample_rate * duration),
        endpoint=False,
    )

    audio = (
        amplitude
        * np.sin(2 * np.pi * frequency * t)
    ).astype(np.float32)

    return audio, sample_rate


def test_classifier_returns_expected_structure():
    audio, sample_rate = create_audio()

    classifier = HeuristicAudioClassifier()

    result = classifier.predict(
        audio,
        sample_rate,
    )

    assert "gender" in result
    assert "age_bracket" in result

    assert "prediction" in result["gender"]
    assert "confidence" in result["gender"]

    assert "prediction" in result["age_bracket"]
    assert "confidence" in result["age_bracket"]


def test_classifier_confidence_is_valid():
    audio, sample_rate = create_audio()

    classifier = HeuristicAudioClassifier()

    result = classifier.predict(
        audio,
        sample_rate,
    )

    assert 0.0 <= result["gender"]["confidence"] <= 1.0
    assert 0.0 <= result["age_bracket"]["confidence"] <= 1.0


def test_silent_audio_returns_unknown():
    sample_rate = 16000

    audio = np.zeros(
        sample_rate * 2,
        dtype=np.float32,
    )

    classifier = HeuristicAudioClassifier()

    result = classifier.predict(
        audio,
        sample_rate,
    )

    assert result["gender"]["prediction"] == "unknown"
    assert result["gender"]["confidence"] == 0.0

    assert result["age_bracket"]["prediction"] == "unknown"
    assert result["age_bracket"]["confidence"] == 0.0
from app.inference.classifier import HeuristicAudioClassifier


def test_classifier_below_threshold_returns_unknown(monkeypatch):
    classifier = HeuristicAudioClassifier()

    monkeypatch.setattr(
        "app.inference.classifier.settings.confidence_threshold",
        0.70,
    )

    result = classifier._apply_threshold(
        "female",
        0.61,
    )

    assert result == {
        "prediction": "unknown",
        "confidence": 0.0,
    }


def test_classifier_at_threshold_returns_prediction(monkeypatch):
    classifier = HeuristicAudioClassifier()

    monkeypatch.setattr(
        "app.inference.classifier.settings.confidence_threshold",
        0.60,
    )

    result = classifier._apply_threshold(
        "female",
        0.60,
    )

    assert result == {
        "prediction": "female",
        "confidence": 0.60,
    }
