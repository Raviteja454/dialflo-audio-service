from typing import Literal

from pydantic import BaseModel, Field


Prediction = Literal["male", "female", "unknown"]
AgeBracket = Literal["18-30", "31-45", "46-60", "60+", "unknown"]
AudioQuality = Literal["good", "degraded", "insufficient"]


class AttributePrediction(BaseModel):
    prediction: str
    confidence: float = Field(ge=0.0, le=1.0)


class AnalyzeResponse(BaseModel):
    contact_id: str

    gender: AttributePrediction

    age_bracket: AttributePrediction

    processing_ms: float

    audio_quality: AudioQuality