import numpy as np


class AudioFeatureError(Exception):
    """Raised when audio features cannot be extracted."""


def extract_features(
    audio: np.ndarray,
    sample_rate: int,
) -> dict:
    """
    Extract lightweight acoustic features from normalized mono audio.

    Uses NumPy directly instead of librosa for lower latency.
    """

    if audio.size == 0:
        raise AudioFeatureError("Audio contains no samples")

    try:
        audio = np.asarray(audio, dtype=np.float32)

        rms = float(np.sqrt(np.mean(np.square(audio))))

        signs = np.signbit(audio)
        zero_crossings = np.count_nonzero(signs[1:] != signs[:-1])
        zero_crossing_rate = float(
            zero_crossings / max(len(audio) - 1, 1)
        )

        # Use a bounded segment for spectral calculations.
        # This keeps inference fast even for longer recordings.
        max_samples = min(len(audio), sample_rate * 10)
        signal = audio[:max_samples]

        # Hann window
        window = np.hanning(len(signal))
        windowed = signal * window

        # Real FFT
        spectrum = np.abs(np.fft.rfft(windowed))
        frequencies = np.fft.rfftfreq(
            len(signal),
            d=1.0 / sample_rate,
        )

        power = spectrum ** 2
        power_sum = np.sum(power)

        if power_sum <= 1e-12:
            spectral_centroid = 0.0
            spectral_bandwidth = 0.0
        else:
            spectral_centroid = float(
                np.sum(frequencies * power) / power_sum
            )

            spectral_bandwidth = float(
                np.sqrt(
                    np.sum(
                        ((frequencies - spectral_centroid) ** 2)
                        * power
                    )
                    / power_sum
                )
            )

        return {
            "rms": rms,
            "zero_crossing_rate": zero_crossing_rate,
            "spectral_centroid": spectral_centroid,
            "spectral_bandwidth": spectral_bandwidth,
        }

    except Exception as exc:
        raise AudioFeatureError(
            "Unable to extract audio features"
        ) from exc
