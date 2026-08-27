import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Dialflo Audio Analysis Service"

    max_audio_size_mb: int = 15
    min_audio_duration_seconds: float = 1.0
    max_audio_duration_seconds: float = 30.0

    target_sample_rate: int = 16_000
    confidence_threshold: float = 0.60

    ffmpeg_path: str = os.getenv("FFMPEG_PATH", "ffmpeg")


settings = Settings()