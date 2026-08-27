import subprocess
from io import BytesIO

import numpy as np
import soundfile as sf

from app.config import settings


class AudioProcessingError(Exception):
    """Raised when audio cannot be decoded or normalized."""


def decode_audio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Decode arbitrary audio bytes using FFmpeg and return
    mono float32 PCM audio at the configured sample rate.
    """

    if not audio_bytes:
        raise AudioProcessingError("Audio payload is empty")

    command = [
        settings.ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-f",
        "wav",
        "-ac",
        "1",
        "-ar",
        str(settings.target_sample_rate),
        "pipe:1",
    ]

    try:
        process = subprocess.run(
            command,
            input=audio_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioProcessingError(
            "Audio decoding timed out"
        ) from exc

    if process.returncode != 0:
        error = process.stderr.decode(
            "utf-8",
            errors="replace",
        )

        raise AudioProcessingError(
            f"Unable to decode audio: {error[:500]}"
        )

    try:
        audio, sample_rate = sf.read(
            BytesIO(process.stdout),
            dtype="float32",
        )
    except Exception as exc:
        raise AudioProcessingError(
            "Unable to read decoded WAV"
        ) from exc

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if len(audio) == 0:
        raise AudioProcessingError(
            "Decoded audio contains no samples"
        )

    return audio, sample_rate


def audio_duration(
    audio: np.ndarray,
    sample_rate: int,
) -> float:
    """Return audio duration in seconds."""

    return len(audio) / sample_rate