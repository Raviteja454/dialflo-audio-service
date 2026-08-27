import numpy as np

from app.audio.features import extract_features


def test_extract_features():
    sample_rate = 16000

    t = np.linspace(
        0,
        2,
        sample_rate * 2,
        endpoint=False,
    )

    audio = (
        0.15 * np.sin(2 * np.pi * 180 * t)
    ).astype(np.float32)

    features = extract_features(
        audio,
        sample_rate,
    )

    assert features["rms"] > 0
    assert features["zero_crossing_rate"] >= 0
    assert features["spectral_centroid"] > 0
    assert features["spectral_bandwidth"] >= 0
