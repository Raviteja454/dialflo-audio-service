from dataclasses import dataclass

import numpy as np

from app.config import settings


@dataclass
class AudioQualityResult:
    quality: str
    duration_seconds: float
    rms: float
    peak: float
    clipping_ratio: float
    estimated_snr_db: float


def calculate_rms(audio: np.ndarray) -> float:
    """Calculate RMS signal energy."""

    if len(audio) == 0:
        return 0.0

    return float(np.sqrt(np.mean(np.square(audio))))


def calculate_peak(audio: np.ndarray) -> float:
    """Calculate maximum absolute amplitude."""

    if len(audio) == 0:
        return 0.0

    return float(np.max(np.abs(audio)))


def calculate_clipping_ratio(audio: np.ndarray) -> float:
    """Calculate percentage of samples close to digital clipping."""

    if len(audio) == 0:
        return 0.0

    clipped = np.abs(audio) >= 0.99

    return float(np.mean(clipped))


def estimate_snr(audio: np.ndarray) -> float:
    """
    Lightweight SNR proxy.

    This isn't a true speech/noise separation model.
    It provides a useful signal-quality heuristic.
    """

    if len(audio) == 0:
        return 0.0

    magnitude = np.abs(audio)

    signal = np.percentile(magnitude, 90)
    noise = np.percentile(magnitude, 10)

    if noise <= 1e-8:
        return 60.0

    return float(
        20 * np.log10(
            max(signal, 1e-8) / noise
        )
    )


def analyze_quality(
    audio: np.ndarray,
    sample_rate: int,
) -> AudioQualityResult:

    duration = len(audio) / sample_rate

    rms = calculate_rms(audio)
    peak = calculate_peak(audio)
    clipping_ratio = calculate_clipping_ratio(audio)
    snr_db = estimate_snr(audio)

    # Completely silent / almost silent audio.
    if rms < 0.005:
        quality = "insufficient"

    # Too little audio to make meaningful analysis.
    elif duration < settings.min_audio_duration_seconds:
        quality = "insufficient"

    # Heavy clipping or very poor signal.
    elif clipping_ratio > 0.05 or snr_db < 6:
        quality = "degraded"

    # Weak but potentially usable signal.
    elif rms < 0.015 or snr_db < 12:
        quality = "degraded"

    else:
        quality = "good"

    return AudioQualityResult(
        quality=quality,
        duration_seconds=duration,
        rms=rms,
        peak=peak,
        clipping_ratio=clipping_ratio,
        estimated_snr_db=snr_db,
    )