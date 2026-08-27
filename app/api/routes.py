import time
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.audio.processor import (
    AudioProcessingError,
    decode_audio,
)
from app.audio.quality import analyze_quality
from app.config import settings
from app.inference.classifier import HeuristicAudioClassifier

router = APIRouter()

classifier = HeuristicAudioClassifier()


@router.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
):
    started = time.perf_counter()
    contact_id = str(uuid.uuid4())

    try:
        audio_bytes = await file.read()

        if not audio_bytes:
            raise HTTPException(
                status_code=400,
                detail="Audio file is empty",
            )

        max_size = settings.max_audio_size_mb * 1024 * 1024

        if len(audio_bytes) > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"Audio file exceeds maximum size of {settings.max_audio_size_mb} MB",
            )

        audio, sample_rate = decode_audio(
            audio_bytes
        )

        duration = len(audio) / sample_rate

        if duration < settings.min_audio_duration_seconds:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Audio duration is too short. "
                    f"Minimum duration is "
                    f"{settings.min_audio_duration_seconds} seconds"
                ),
            )

        if duration > settings.max_audio_duration_seconds:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Audio duration is too long. "
                    f"Maximum duration is "
                    f"{settings.max_audio_duration_seconds} seconds"
                ),
            )

        quality = analyze_quality(
            audio,
            sample_rate,
        )

        predictions = classifier.predict(
            audio,
            sample_rate,
        )

        processing_ms = round(
            (time.perf_counter() - started) * 1000,
            2,
        )

        return {
            "contact_id": contact_id,
            "gender": predictions["gender"],
            "age_bracket": predictions["age_bracket"],
            "processing_ms": processing_ms,
            "audio_quality": quality.quality,
        }

    except AudioProcessingError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Audio analysis failed",
        ) from exc
