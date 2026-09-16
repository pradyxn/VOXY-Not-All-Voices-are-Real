from pydantic import BaseModel, Field
from typing import Literal

RiskLevel = Literal["LOW", "MODERATE", "HIGH", "CRITICAL"]

class AnalysisResponse(BaseModel):
    status: str
    synthetic_score: float = Field(ge=0, le=100)
    risk_score: int = Field(ge=0, le=100)
    risk_level: RiskLevel
    windows_analyzed: int
    audio_quality: str
    mode: Literal["pretrained", "demo"]
    detector: str = "AASIST pretrained anti-spoofing model"
    model_message: str | None = None
    timeline: list[float] = []

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    mode: Literal["pretrained", "demo"]
    detector: str
    device: str
    ffmpeg_available: bool
