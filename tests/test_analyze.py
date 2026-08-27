from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)

SAMPLE_AUDIO = Path("tests/sample.wav")


def test_analyze_audio():
    with SAMPLE_AUDIO.open("rb") as audio:
        response = client.post(
            "/analyze",
            files={
                "file": (
                    "sample.wav",
                    audio,
                    "audio/wav",
                )
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert "contact_id" in body
    assert "gender" in body
    assert "age_bracket" in body
    assert "processing_ms" in body
    assert "audio_quality" in body

    assert body["gender"]["prediction"] in {
        "male",
        "female",
        "unknown",
    }

    assert 0.0 <= body["gender"]["confidence"] <= 1.0

    assert body["age_bracket"]["prediction"] in {
        "18-30",
        "31-45",
        "46-60",
        "60+",
        "unknown",
    }

    assert 0.0 <= body["age_bracket"]["confidence"] <= 1.0

    assert body["processing_ms"] >= 0

    assert body["audio_quality"] in {
        "good",
        "degraded",
        "insufficient",
    }


def test_analyze_empty_audio():
    response = client.post(
        "/analyze",
        files={
            "file": (
                "empty.wav",
                b"",
                "audio/wav",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Audio file is empty"


def test_analyze_without_file():
    response = client.post("/analyze")

    assert response.status_code == 422
import io
import wave
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)

SAMPLE_AUDIO = Path("tests/sample.wav")


def make_wav(duration_seconds: float) -> bytes:
    sample_rate = 16000
    samples = int(sample_rate * duration_seconds)

    audio = np.zeros(samples, dtype=np.int16)

    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio.tobytes())

    return buffer.getvalue()


def test_analyze_audio_too_short():
    audio = make_wav(0.5)

    response = client.post(
        "/analyze",
        files={
            "file": (
                "short.wav",
                audio,
                "audio/wav",
            )
        },
    )

    assert response.status_code == 400
    assert "too short" in response.json()["detail"]


def test_analyze_audio_too_long():
    audio = make_wav(31)

    response = client.post(
        "/analyze",
        files={
            "file": (
                "long.wav",
                audio,
                "audio/wav",
            )
        },
    )

    assert response.status_code == 400
    assert "too long" in response.json()["detail"]


def test_analyze_file_too_large():
    oversized_audio = b"x" * (15 * 1024 * 1024 + 1)

    response = client.post(
        "/analyze",
        files={
            "file": (
                "large.wav",
                oversized_audio,
                "audio/wav",
            )
        },
    )

    assert response.status_code == 413
    assert "exceeds maximum size" in response.json()["detail"]
